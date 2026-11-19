# SmartChunk

A structure-aware chunking engine for RAG, written in Go.

It reads Markdown files and writes clean JSONL chunks that any vector database can ingest. It also ships an evaluation that runs it against a naive fixed-size splitter on real documentation and reports the difference.

## What it does

A fixed-size splitter counts tokens and cuts. It will cut a code block in half, separate a table from its header row, and leave a heading at the bottom of one chunk with its content at the top of the next.

SmartChunk parses the document first and cuts only where the document allows it:

- A code block, table or list is never cut. One that is too big on its own stays whole and is flagged `oversized`.
- A heading always stays with the content below it.
- Inside a long section, cuts go where the topic shifts, found by embedding neighbouring passages and looking for dips in similarity.
- Chunks that are too small are merged into the most closely related neighbour. Prose that is too long is split at sentence boundaries.
- Lines repeated across many chunks (footers, navigation) are stripped, and near-duplicate chunks are dropped.
- Every chunk records where it came from: file, line range, heading path.
