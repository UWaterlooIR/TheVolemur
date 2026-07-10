#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUN_TAG = "vole-run"


class VoleRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class Judgment:
    topic_id: str
    docno: str
    relevance: int
    position: int


def normalize_docno(docno: str) -> str:
    docno = docno.strip()
    if len(docno) >= 2 and docno[0] == docno[-1] == '"':
        return docno[1:-1]
    return docno


def infer_topic_id(path: Path, event: dict[str, Any]) -> str:
    topic_id = event.get("topic_id")
    if isinstance(topic_id, str) and topic_id.strip():
        return topic_id
    return path.stem


def is_conversation_request(event: dict[str, Any]) -> bool:
    return isinstance(event.get("request"), dict)


def extract_decision(event: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(event.get("latest_judgment"), dict) and isinstance(event.get("action"), dict):
        return event

    if is_conversation_request(event):
        return None

    if event.get("type") == "model_decision":
        decision = event.get("decision")
        return decision if isinstance(decision, dict) else None

    if event.get("actor") == "model" and event.get("event") == "response":
        decision = event.get("decision")
        return decision if isinstance(decision, dict) else None

    return None


def collect_judgments(path: Path, judgments: dict[str, list[Judgment]]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise VoleRunError(f"Log not found: {path}") from exc

    seen: set[tuple[str, str]] = {
        (judgment.topic_id, judgment.docno)
        for topic_judgments in judgments.values()
        for judgment in topic_judgments
    }

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VoleRunError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

        if not isinstance(event, dict):
            continue

        decision = extract_decision(event)
        if decision is None:
            continue

        judgment = decision.get("latest_judgment")
        if not isinstance(judgment, dict):
            raise VoleRunError(f"{path}:{line_number}: no latest_judgment object.")

        docno = judgment.get("docno")
        if docno is None:
            continue
        if not isinstance(docno, str) or not docno.strip():
            raise VoleRunError(f"{path}:{line_number}: latest_judgment.docno must be a string or null.")

        relevance = judgment.get("relevance")
        if not isinstance(relevance, int) or isinstance(relevance, bool):
            raise VoleRunError(f"{path}:{line_number}: latest_judgment.relevance must be an integer.")

        topic_id = infer_topic_id(path, event)
        docno = normalize_docno(docno)
        key = (topic_id, docno)
        if key in seen:
            continue

        seen.add(key)
        position = len(judgments.setdefault(topic_id, []))
        judgments[topic_id].append(Judgment(topic_id, docno, relevance, position))


def print_run(judgments: dict[str, list[Judgment]], run_tag: str) -> None:
    for topic_id, topic_judgments in judgments.items():
        ranked = sorted(topic_judgments, key=lambda judgment: (-judgment.relevance, judgment.position))
        count = len(ranked)

        for rank, judgment in enumerate(ranked, start=1):
            score = count - rank
            print(f"{topic_id} Q0 {judgment.docno} {rank} {score} {run_tag}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fake TREC run from Vole JSONL logs, sorted by relevance then log order."
    )
    parser.add_argument("logs", nargs="+", help="One or more Vole .log files.")
    parser.add_argument(
        "--run-tag",
        default=DEFAULT_RUN_TAG,
        help=f"TREC run tag to write in column 6. Default: {DEFAULT_RUN_TAG}.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        judgments: dict[str, list[Judgment]] = {}
        for log in args.logs:
            collect_judgments(Path(log), judgments)
    except VoleRunError as exc:
        print(f"vole_run.py: {exc}", file=sys.stderr)
        return 1

    print_run(judgments, args.run_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
