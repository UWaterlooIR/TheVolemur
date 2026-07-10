#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


class FinalizeQrelsError(RuntimeError):
    pass


def load_topic_ids(path: Path) -> set[str]:
    topic_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FinalizeQrelsError(f"Topic file not found: {path}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if not fields:
            raise FinalizeQrelsError(f"{path}:{line_number}: expected a topic id.")
        topic_ids.add(fields[0])

    if not topic_ids:
        raise FinalizeQrelsError(f"Topic file has no usable topics: {path}")
    return topic_ids


def parse_qrel_line(path: Path, line_number: int, line: str) -> tuple[str, str, str]:
    fields = line.split()
    if len(fields) < 4:
        raise FinalizeQrelsError(f"{path}:{line_number}: expected at least 4 qrel fields.")
    return fields[0], fields[2], fields[3]


def finalize_qrels(topic_path: Path, qrel_paths: list[Path]) -> None:
    topic_ids = load_topic_ids(topic_path)
    seen: dict[tuple[str, str], str] = {}

    for path in qrel_paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise FinalizeQrelsError(f"Qrels file not found: {path}") from exc

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            topic_id, docno, relevance = parse_qrel_line(path, line_number, line)
            if topic_id not in topic_ids:
                continue

            key = (topic_id, docno)
            previous = seen.get(key)
            if previous is not None:
                print(
                    f"duplicate {topic_id} {docno}: {previous} {relevance}",
                    file=sys.stderr,
                )
                continue

            seen[key] = relevance
            print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter qrels to a topic set and combine them in priority order."
    )
    parser.add_argument("topics", type=Path, help="Topic file; first field is the topic id.")
    parser.add_argument("qrels", nargs="+", type=Path, help="Qrels files in priority order.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        finalize_qrels(args.topics, args.qrels)
    except FinalizeQrelsError as exc:
        print(f"finalize_qrels.py: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
