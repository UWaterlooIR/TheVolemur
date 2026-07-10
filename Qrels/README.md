# Qrels

This directory contains the relevance judgments used for the Vole comparison on
the 86-topic TREC 2024 RAG Track subset.

Qrels are in standard four-column TREC format:

```text
topic_id iteration docno relevance
```

Files:

- `qrels.rag24.test-umbrela-all.txt`: the base RAG 2024 UMBRELA qrels.
- `umbrela.holes.qrels`: local UMBRELA hole-filling judgments for Vole-retrieved
  topic/document pairs.
- `combined.man86.qrels`: the final qrels used for the reported `man86`
  evaluation. It combines the base qrels with the local hole-filling qrels for
  the 86-topic subset, keeping the first judgment for each topic/document pair.
- `finalize_qrels.py`: helper script used to filter qrels to a topic set and
  combine qrels files in priority order.

`umbrela.holes.qrels` contains 2,686 judgments. Relative to the base qrels, it
adds 2,676 previously unjudged topic/document pairs, including 420 grade-2
judgments and 18 grade-3 judgments.

`combined.man86.qrels` contains 30,493 judgments and has no duplicate
topic/document pairs.
