# Vole Result Logs

This directory contains saved logs from `../vole.py` for the MS MARCO V2.1
deduped segment collection used by the TREC 2024 RAG track.

The logs cover the 86-topic track subset whose topics received full or partial
manual judgments. Each topic has three JSONL files with the same topic id:

- `.log`: the full run log, including runner events, model decisions, server
  requests, and server responses.
- `.conv`: the compact append-only conversation that was visible to the model
  while it searched.
- `.meta`: compact metadata for the run, including token usage, model-call
  summaries, and server timing records.
