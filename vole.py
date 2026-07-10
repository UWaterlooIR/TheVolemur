from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI, OpenAIError
except ModuleNotFoundError:
    OpenAI = None
    OpenAIError = Exception


HOST = "127.0.0.1"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_MODEL_ENV_VAR = "OPENAI_MODEL"
DEFAULT_MODEL = "gpt-5.5"
INSTRUCTIONS_PATH = Path("instructions.txt")
RESULTS_DIR = Path("results")
MAX_DOCUMENT_JUDGMENTS = 50
MAX_MODEL_CALLS = 100
MAX_DUPLICATE_SKIPS = 100
MODEL_ATTEMPTS = 2


class VoleError(RuntimeError):
    pass


class TopicStop(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass
class RunState:
    topic_id: str
    model_calls: int = 0
    judged_relevance: dict[str, int] = field(default_factory=dict)

    @property
    def document_judgments(self) -> int:
        return len(self.judged_relevance)

    @property
    def relevance_3(self) -> int:
        return sum(1 for relevance in self.judged_relevance.values() if relevance == 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "document_judgments": self.document_judgments,
            "relevance_3": self.relevance_3,
            "max_document_judgments": MAX_DOCUMENT_JUDGMENTS,
            "max_model_calls": MAX_MODEL_CALLS,
        }


def stop_reason(state: RunState) -> str | None:
    if state.document_judgments >= MAX_DOCUMENT_JUDGMENTS:
        return "max_document_judgments_reached"
    if state.model_calls >= MAX_MODEL_CALLS:
        return "max_model_calls_reached"
    return None


def normalize_docno(docno: str) -> str:
    docno = docno.strip()
    if len(docno) >= 2 and docno[0] == docno[-1] == '"':
        return docno[1:-1]
    return docno


def load_instructions(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise VoleError(f"Instructions file not found: {path}") from exc


def load_topics(path: Path) -> list[tuple[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as input_file:
            reader = csv.reader(input_file, delimiter="\t")
            topics = []
            for line_number, row in enumerate(reader, start=1):
                if not row or not any(field.strip() for field in row):
                    continue
                if row[0].lstrip().startswith("#"):
                    continue
                if len(row) < 2:
                    raise VoleError(f"Topic row {line_number} has no information need.")

                topic_id = row[0].strip()
                information_need = "\n".join(field.strip() for field in row[1:] if field.strip())
                if not topic_id:
                    raise VoleError(f"Topic row {line_number} has an empty id.")
                if not information_need:
                    raise VoleError(f"Topic row {line_number} has an empty information need.")
                validate_topic_id(topic_id, line_number)
                topics.append((topic_id, information_need))
    except FileNotFoundError as exc:
        raise VoleError(f"Topics file not found: {path}") from exc

    if not topics:
        raise VoleError(f"Topics file has no usable topics: {path}")
    return topics


def validate_topic_id(topic_id: str, line_number: int) -> None:
    if topic_id in {".", ".."} or "/" in topic_id or "\\" in topic_id:
        raise VoleError(f"Topic row {line_number} has an unsafe id: {topic_id!r}")


def result_paths(topic_id: str) -> tuple[Path, Path, Path]:
    log_path = RESULTS_DIR / f"{topic_id}.log"
    meta_path = RESULTS_DIR / f"{topic_id}.meta"
    conversation_path = RESULTS_DIR / f"{topic_id}.conv"
    return log_path, meta_path, conversation_path


def working_paths() -> tuple[Path, Path, Path]:
    return (
        RESULTS_DIR / "working.log",
        RESULTS_DIR / "working.meta",
        RESULTS_DIR / "working.conv",
    )


def reset_working_logs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in working_paths():
        with path.open("w", encoding="utf-8"):
            pass


def results_exist(topic_id: str) -> bool:
    return any(path.exists() for path in result_paths(topic_id))


def initialize_logs(topic_id: str) -> tuple[Path, Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in result_paths(topic_id):
        if path.exists():
            raise VoleError(f"Refusing to overwrite existing log: {path}")

    log_path, meta_path, conversation_path = working_paths()
    for path in (log_path, meta_path, conversation_path):
        if not path.exists():
            path.touch()
    return log_path, meta_path, conversation_path


def finalize_logs(
    topic_id: str,
    log_path: Path,
    meta_path: Path,
    conversation_path: Path,
) -> tuple[Path, Path, Path]:
    final_log_path, final_meta_path, final_conversation_path = result_paths(topic_id)
    for path in (final_log_path, final_meta_path, final_conversation_path):
        if path.exists():
            raise VoleError(f"Refusing to overwrite existing log: {path}")
    final_log_path.write_bytes(log_path.read_bytes())
    final_meta_path.write_bytes(meta_path.read_bytes())
    final_conversation_path.write_bytes(conversation_path.read_bytes())
    for path in (log_path, meta_path, conversation_path):
        with path.open("w", encoding="utf-8"):
            pass
    return final_log_path, final_meta_path, final_conversation_path


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()


def append_runner_event(
    path: Path,
    event: str,
    state: RunState,
    **fields: Any,
) -> None:
    payload = {
        "actor": "runner",
        "event": event,
        "topic_id": state.topic_id,
        **state.as_dict(),
    }
    payload.update(fields)
    append_event(path, payload)


def read_log(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(
    instructions_template: str,
    information_need: str,
    log_text: str,
) -> str:
    instructions = instructions_template.replace("{information_need}", information_need)
    return f"{instructions.rstrip()}\n\nAPPEND-ONLY CONVERSATION (JSONL):\n{log_text}"


def append_conversation_runner_request(conversation_path: Path, state: RunState) -> None:
    append_event(
        conversation_path,
        {"request": {"judgments": f"{state.document_judgments}/{MAX_DOCUMENT_JUDGMENTS}"}},
    )


def build_openai_client() -> OpenAI:
    if OpenAI is None:
        raise VoleError(
            "Missing required package: openai. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        )

    api_key = os.getenv(OPENAI_API_KEY_ENV_VAR)
    if not api_key:
        raise VoleError(f"Missing {OPENAI_API_KEY_ENV_VAR}. Set it in your shell environment.")

    return OpenAI(api_key=api_key)


def ask_model(
    client: OpenAI,
    model: str,
    instructions_template: str,
    information_need: str,
    log_path: Path,
    meta_path: Path,
    conversation_path: Path,
    state: RunState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error = "unknown model error"
    for attempt in range(1, MODEL_ATTEMPTS + 1):
        if state.model_calls >= MAX_MODEL_CALLS:
            raise TopicStop("max_model_calls_reached")
        state.model_calls += 1
        model_call = state.model_calls
        prompt = build_prompt(
            instructions_template,
            information_need,
            read_log(conversation_path),
        )
        append_event(
            log_path,
            {
                "actor": "model",
                "event": "request",
                "topic_id": state.topic_id,
                "model_call": model_call,
                "attempt": attempt,
                "model": model,
                "prompt_characters": len(prompt),
                **state.as_dict(),
            },
        )
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                text={"format": {"type": "json_object"}},
            )
        except OpenAIError as exc:
            last_error = f"OpenAI request failed: {exc}"
            log_error_event = {
                "actor": "runner",
                "event": "error",
                "topic_id": state.topic_id,
                "model_call": model_call,
                "attempt": attempt,
                "phase": "model_request",
                "error": last_error,
                **state.as_dict(),
            }
            meta_error_event = {
                "type": "driver_error",
                "topic_id": state.topic_id,
                "model_call": model_call,
                "attempt": attempt,
                "phase": "model_request",
                "error": last_error,
            }
            append_event(log_path, log_error_event)
            append_event(meta_path, meta_error_event)
            raise VoleError(last_error) from exc

        output_text = response.output_text
        usage = extract_usage(response)
        append_event(
            meta_path,
            {
                "type": "model_response",
                "topic_id": state.topic_id,
                "step": model_call,
                "model_call": model_call,
                "attempt": attempt,
                "model": model,
                "usage": usage,
                "output_characters": len(output_text),
                "prompt_characters": len(prompt),
            },
        )
        try:
            payload = json.loads(output_text)
            decision = validate_decision(payload)
            append_event(
                log_path,
                {
                    "actor": "model",
                    "event": "response",
                    "topic_id": state.topic_id,
                    "model_call": model_call,
                    "attempt": attempt,
                    "model": model,
                    "usage": usage,
                    "output_characters": len(output_text),
                    "decision": decision,
                },
            )
            return decision, usage
        except (json.JSONDecodeError, VoleError) as exc:
            last_error = str(exc)
            append_event(
                log_path,
                {
                    "actor": "model",
                    "event": "response",
                    "topic_id": state.topic_id,
                    "model_call": model_call,
                    "attempt": attempt,
                    "model": model,
                    "status": "invalid",
                    "error": last_error,
                    "raw_output": output_text,
                },
            )

    raise VoleError(last_error)


def extract_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cached_tokens": getattr(input_details, "cached_tokens", None),
        "reasoning_tokens": getattr(output_details, "reasoning_tokens", None),
    }


def validate_decision(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise VoleError("Model output JSON must be an object.")

    latest_judgment = require_object(payload, "latest_judgment")
    docno = latest_judgment.get("docno")
    if docno is not None and not isinstance(docno, str):
        raise VoleError("latest_judgment.docno must be a string or null.")

    relevance = latest_judgment.get("relevance")
    if not isinstance(relevance, int) or isinstance(relevance, bool) or relevance not in {0, 1, 2, 3}:
        raise VoleError("latest_judgment.relevance must be an integer from 0 to 3.")

    require_string(latest_judgment, "reason", allow_empty=True)
    require_string(latest_judgment, "learned", allow_empty=True)

    action = require_object(payload, "action")
    op = require_string(action, "op")
    if op not in {"query", "next", "document"}:
        raise VoleError(f"Invalid action op: {op!r}")
    require_string(action, "reason", allow_empty=True)

    if op == "query":
        query = require_string(action, "query")
        action["query"] = query.lower()
    elif op == "next":
        require_string(action, "qid")
    elif op == "document":
        require_string(action, "docno")

    return payload


def require_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise VoleError(f"{key} must be an object.")
    return value


def require_string(payload: dict[str, Any], key: str, allow_empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise VoleError(f"{key} must be a string.")
    if not allow_empty and not value.strip():
        raise VoleError(f"{key} must be non-empty.")
    return value


def server_action(decision: dict[str, Any]) -> dict[str, str]:
    action = decision["action"]
    op = action["op"]
    if op == "query":
        return {"op": op, "query": action["query"]}
    if op == "next":
        return {"op": op, "qid": action["qid"]}
    return {"op": op, "docno": action["docno"]}


def connect_local(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, port))
    except OSError:
        sock.close()
        raise
    return sock


def request(stream: Any, query: dict[str, str]) -> dict[str, Any] | None:
    line = json.dumps(query, separators=(",", ":")) + "\n"
    stream.write(line.encode("utf-8"))
    stream.flush()
    response = stream.readline()
    if not response:
        return None
    return json.loads(response.decode("utf-8"))


def response_docno(record: dict[str, Any]) -> str | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    docno = response.get("docno")
    if not isinstance(docno, str) or not docno.strip():
        return None
    return normalize_docno(docno)


def response_qid(record: dict[str, Any]) -> str | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    qid = response.get("qid")
    return qid if isinstance(qid, str) and qid else None


def response_done(record: dict[str, Any]) -> bool:
    response = record.get("response")
    return isinstance(response, dict) and response.get("done") is True


def response_rank(record: dict[str, Any]) -> Any:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    return response.get("rank")


def add_skipped(record: dict[str, Any], skipped: list[dict[str, Any]]) -> None:
    if not skipped:
        return
    response = record.get("response")
    if isinstance(response, dict):
        response["skipped"] = skipped


def append_server_request(
    log_path: Path,
    meta_path: Path,
    state: RunState,
    action: dict[str, str],
) -> None:
    append_event(
        log_path,
        {
            "actor": "server",
            "event": "request",
            "topic_id": state.topic_id,
            "model_call": state.model_calls,
            "request": action,
            **state.as_dict(),
        },
    )
    append_event(
        meta_path,
        {
            "type": "server_request",
            "topic_id": state.topic_id,
            "step": state.model_calls,
            "model_call": state.model_calls,
            "request_op": action["op"],
            "query": action.get("query"),
            "qid": action.get("qid"),
            "docno": action.get("docno"),
        },
    )


def append_server_response(
    log_path: Path,
    meta_path: Path,
    state: RunState,
    action: dict[str, str],
    record: dict[str, Any],
) -> None:
    append_event(
        log_path,
        {
            "actor": "server",
            "event": "response",
            "topic_id": state.topic_id,
            "model_call": state.model_calls,
            "request": action,
            "record": record,
            **state.as_dict(),
        },
    )
    response = record.get("response") if isinstance(record, dict) else None
    append_event(
        meta_path,
        {
            "type": "server_response",
            "topic_id": state.topic_id,
            "step": state.model_calls,
            "model_call": state.model_calls,
            "request_op": action["op"],
            "response_op": response.get("op") if isinstance(response, dict) else None,
            "ok": response.get("ok") if isinstance(response, dict) else None,
            "done": response.get("done") if isinstance(response, dict) else None,
            "qid": response.get("qid") if isinstance(response, dict) else None,
            "rank": response.get("rank") if isinstance(response, dict) else None,
            "docno": response.get("docno") if isinstance(response, dict) else None,
            "skipped_count": len(response.get("skipped", [])) if isinstance(response, dict) else 0,
            "time": record.get("time") if isinstance(record, dict) else None,
        },
    )


def append_conversation_server_response(
    conversation_path: Path,
    action: dict[str, str],
    record: dict[str, Any],
) -> None:
    response = record.get("response")
    if isinstance(response, dict):
        response = {key: value for key, value in response.items() if key != "burrow"}
    payload: dict[str, Any] = {
        "request": action,
        "response": response,
    }
    append_event(conversation_path, payload)


def request_server(
    stream: Any,
    log_path: Path,
    meta_path: Path,
    state: RunState,
    action: dict[str, str],
) -> dict[str, Any]:
    skipped: list[dict[str, Any]] = []
    current_action = action

    for _skip_attempt in range(MAX_DUPLICATE_SKIPS + 1):
        append_server_request(log_path, meta_path, state, current_action)
        try:
            record = request(stream, current_action)
        except (OSError, json.JSONDecodeError) as exc:
            append_event(
                log_path,
                {
                    "actor": "runner",
                    "event": "error",
                    "topic_id": state.topic_id,
                    "model_call": state.model_calls,
                    "phase": "server_response",
                    "error": str(exc),
                    **state.as_dict(),
                },
            )
            raise VoleError(f"{state.topic_id}: bad server response: {exc}") from exc

        if record is None:
            append_event(
                log_path,
                {
                    "actor": "runner",
                    "event": "error",
                    "topic_id": state.topic_id,
                    "model_call": state.model_calls,
                    "phase": "server_response",
                    "error": "server connection closed",
                    **state.as_dict(),
                },
            )
            raise VoleError(f"{state.topic_id}: server connection closed")

        docno = response_docno(record)
        if (
            current_action["op"] in {"query", "next"}
            and docno is not None
            and docno in state.judged_relevance
        ):
            skipped_entry = {
                "docno": docno,
                "rank": response_rank(record),
                "relevance": state.judged_relevance[docno],
            }
            skipped.append(skipped_entry)
            add_skipped(record, [skipped_entry])
            append_server_response(log_path, meta_path, state, current_action, record)

            qid = response_qid(record)
            if qid is None or response_done(record):
                add_skipped(record, skipped)
                return record
            current_action = {"op": "next", "qid": qid}
            continue

        add_skipped(record, skipped)
        append_server_response(log_path, meta_path, state, current_action, record)
        return record

    append_event(
        log_path,
        {
            "actor": "runner",
            "event": "error",
            "topic_id": state.topic_id,
            "model_call": state.model_calls,
            "phase": "duplicate_skip",
            "error": f"exceeded duplicate skip limit ({MAX_DUPLICATE_SKIPS})",
            "skipped": skipped,
            **state.as_dict(),
        },
    )
    raise VoleError(f"{state.topic_id}: exceeded duplicate skip limit ({MAX_DUPLICATE_SKIPS})")


def run_topic(
    client: OpenAI,
    model: str,
    port: int,
    instructions_template: str,
    topic_id: str,
    information_need: str,
) -> None:
    log_path, meta_path, conversation_path = initialize_logs(topic_id)
    state = RunState(topic_id=topic_id)
    append_runner_event(
        log_path,
        "start",
        state,
        information_need=information_need,
    )

    try:
        sock = connect_local(port)
    except OSError as exc:
        raise VoleError(f"Cannot connect to {HOST}:{port}: {exc}") from exc

    with sock, sock.makefile("rwb") as stream:
        while True:
            reason = stop_reason(state)
            if reason is not None:
                append_runner_event(log_path, "stop", state, reason=reason)
                finalize_logs(topic_id, log_path, meta_path, conversation_path)
                print(f"{topic_id}: stopped ({reason})", flush=True)
                return

            append_conversation_runner_request(conversation_path, state)
            try:
                decision, usage = ask_model(
                    client=client,
                    model=model,
                    instructions_template=instructions_template,
                    information_need=information_need,
                    log_path=log_path,
                    meta_path=meta_path,
                    conversation_path=conversation_path,
                    state=state,
                )
            except TopicStop as exc:
                append_runner_event(log_path, "stop", state, reason=exc.reason)
                finalize_logs(topic_id, log_path, meta_path, conversation_path)
                print(f"{topic_id}: stopped ({exc.reason})", flush=True)
                return

            append_event(conversation_path, decision)

            judgment = decision["latest_judgment"]
            judged_docno = judgment["docno"]
            judged_relevance = judgment["relevance"]
            if judged_docno is not None:
                state.judged_relevance[normalize_docno(judged_docno)] = judged_relevance

            append_event(
                meta_path,
                {
                    "type": "model_decision",
                    "topic_id": topic_id,
                    "step": state.model_calls,
                    "model_call": state.model_calls,
                    "action_op": decision["action"]["op"],
                    "judged_docno": judged_docno,
                    "judged_relevance": judged_relevance,
                    "document_judgments": state.document_judgments,
                    "relevance_3": state.relevance_3,
                    "usage": usage,
                },
            )
            append_runner_event(log_path, "state", state)

            reason = stop_reason(state)
            if reason is not None:
                append_runner_event(log_path, "stop", state, reason=reason)
                finalize_logs(topic_id, log_path, meta_path, conversation_path)
                print(f"{topic_id}: stopped ({reason})", flush=True)
                return

            action = server_action(decision)
            record = request_server(stream, log_path, meta_path, state, action)
            append_conversation_server_response(conversation_path, action, record)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Vole SSR search driver.")
    parser.add_argument("port", type=int, help="Local SSR server port.")
    parser.add_argument("topics", help="TSV file whose first field is the topic id.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        instructions_template = load_instructions(INSTRUCTIONS_PATH)
        topics = load_topics(Path(args.topics))
        model = os.getenv(OPENAI_MODEL_ENV_VAR) or DEFAULT_MODEL
        client: OpenAI | None = None
        reset_working_logs()

        for topic_id, information_need in topics:
            if results_exist(topic_id):
                print(f"Skipping {topic_id}: results exist", flush=True)
                continue
            print(f"Searching {topic_id} ({information_need})", flush=True)
            if client is None:
                client = build_openai_client()
            run_topic(
                client=client,
                model=model,
                port=args.port,
                instructions_template=instructions_template,
                topic_id=topic_id,
                information_need=information_need,
            )
    except VoleError as exc:
        print(f"vole.py: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
