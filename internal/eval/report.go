package eval

import (
	"bytes"
	"fmt"
	"html/template"
	"runtime"
	"strings"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
)

// ResultsMarkdown renders the comparison table and its supporting detail.
func ResultsMarkdown(r *Report) string {
	var sb strings.Builder
	b, s := r.Baseline, r.Smart

	sb.WriteString("# SmartChunk evaluation\n\n")
	fmt.Fprintf(&sb, "- **Corpus:** `%s` (%d documents)\n", r.Config.CorpusDir, len(r.Docs))
	fmt.Fprintf(&sb, "- **Questions:** `%s` (%d questions)\n", r.Config.QuestionsPath, len(r.Questions))
	fmt.Fprintf(&sb, "- **Embedder:** %s\n", orDash(r.Config.EmbedderName))
	fmt.Fprintf(&sb, "- **Baseline:** %s\n", b.Settings)
	fmt.Fprintf(&sb, "- **SmartChunk:** %s\n", s.Settings)
	fmt.Fprintf(&sb, "- **Machine:** %s/%s, %d CPUs, %s, %d workers\n", runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.Version(), workersOf(r))
	fmt.Fprintf(&sb, "- **Generated:** %s\n\n", r.Generated.Format("2006-01-02 15:04 MST"))

	sb.WriteString("| Metric | Baseline | SmartChunk |\n|---|---:|---:|\n")
	row := func(name, bv, sv string) { fmt.Fprintf(&sb, "| %s | %s | %s |\n", name, bv, sv) }
	section := func(name string) { fmt.Fprintf(&sb, "| **%s** | | |\n", name) }

	row("Chunks", fmt.Sprint(len(b.Chunks)), fmt.Sprint(len(s.Chunks)))
	section("Structure")
	row("Code blocks cut", ratio(b.Structure.CodeCut, b.Structure.CodeBlocks), ratio(s.Structure.CodeCut, s.Structure.CodeBlocks))
	row("Tables cut", ratio(b.Structure.TableCut, b.Structure.Tables), ratio(s.Structure.TableCut, s.Structure.Tables))
	row("Lists cut", ratio(b.Structure.ListCut, b.Structure.Lists), ratio(s.Structure.ListCut, s.Structure.Lists))
	row("Structural violations (total)", ratio(b.Structure.Violations(), b.Structure.AtomicBlocks()), ratio(s.Structure.Violations(), s.Structure.AtomicBlocks()))
	row("Cut blocks with no intact copy", fmt.Sprint(b.Structure.NeverIntact), fmt.Sprint(s.Structure.NeverIntact))
	row("Orphan headings", ratio(b.Structure.OrphanHeadings, b.Structure.Headings), ratio(s.Structure.OrphanHeadings, s.Structure.Headings))
	row("Orphan headings never joined to their content", fmt.Sprint(b.Structure.OrphansNeverJoined), fmt.Sprint(s.Structure.OrphansNeverJoined))
	section("Duplication")
	row(fmt.Sprintf("Near-duplicate chunks (Jaccard >= %.2f)", r.Config.DuplicateThreshold), ratio(b.Duplicates, len(b.Chunks)), ratio(s.Duplicates, len(s.Chunks)))
	section("Retrieval")
	row("Hit rate @1", pct(b.Retrieval.Hit1), pct(s.Retrieval.Hit1))
	row("Hit rate @3", pct(b.Retrieval.Hit3), pct(s.Retrieval.Hit3))
	row("Hit rate @5", pct(b.Retrieval.Hit5), pct(s.Retrieval.Hit5))
	row("MRR", fmt.Sprintf("%.3f", b.Retrieval.MRR), fmt.Sprintf("%.3f", s.Retrieval.MRR))
	row("Whole answer in one top-5 chunk", pct(b.Retrieval.Complete5), pct(s.Retrieval.Complete5))
	section("Chunk size (tokens)")
	row("Mean", fmt.Sprintf("%.1f", b.Tokens.Mean), fmt.Sprintf("%.1f", s.Tokens.Mean))
	row("p95", fmt.Sprint(b.Tokens.P95), fmt.Sprint(s.Tokens.P95))
	row("Max", fmt.Sprint(b.Tokens.Max), fmt.Sprint(s.Tokens.Max))
	row("Chunks flagged oversized", fmt.Sprint(b.Oversized), fmt.Sprint(s.Oversized))
	section("Throughput")
	row("Docs/sec", rate(b.DocsPerSec), rate(s.DocsPerSec))
	row("Chunks/sec", rate(b.ChunksPerSec), rate(s.ChunksPerSec))
	sb.WriteString("\n")

	sb.WriteString("## Reading the numbers\n\n")
	sb.WriteString("- **Cut** means at least one chunk holds only part of the block. Both chunkers are judged by line range against the same parse of the source, so the count does not depend on how either one works internally.\n")
	sb.WriteString("- **No intact copy** counts cut blocks that no chunk holds whole. With overlapping windows a block can be cut by one chunk and complete in the next.\n")
	sb.WriteString("- **Orphan heading**: a heading stranded at the end of a chunk, which stops before the first line of the content the heading introduces. **Never joined** counts the orphans that no other chunk holds together with that content either.\n")
	sb.WriteString("- **Hit rate @k**: a relevant chunk (same file, overlapping the answer lines) is in the top k by cosine similarity. **MRR** is over the full ranking.\n")
	sb.WriteString("- **Whole answer in one top-5 chunk** is stricter: a single retrieved chunk must cover every answer line.\n")
	sb.WriteString("- **Tokens** are counted by the built-in approximate tokenizer for both chunkers.\n")
	if r.Config.NoHeadingContext {
		sb.WriteString("- Chunks were embedded as their bare text (`--no-heading-context`).\n")
	} else {
		sb.WriteString("- Chunks were embedded as heading breadcrumb plus text. The baseline has no heading path, so its chunks are embedded as plain text.\n")
	}
	sb.WriteString("\n")

	sb.WriteString("## Throughput detail\n\n")
	fmt.Fprintf(&sb, "- Baseline: %d documents in %s.\n", b.Docs, dur(b.Elapsed))
	fmt.Fprintf(&sb, "- SmartChunk, full pipeline: %d documents in %s.", s.Docs, dur(s.Elapsed))
	if r.Config.EmbedStats != nil {
		fmt.Fprintf(&sb, " The semantic stage looked up %d embeddings: %d served from the on-disk cache, %d computed.", s.CacheHits+s.CacheMisses, s.CacheHits, s.CacheMisses)
	}
	sb.WriteString("\n")
	if secs := r.StructureOnlyElapsed.Seconds(); secs > 0 {
		fmt.Fprintf(&sb, "- SmartChunk without the semantic stage (no embedding calls): %s, %s docs/sec.\n", dur(r.StructureOnlyElapsed), rate(float64(s.Docs)/secs))
	}
	sb.WriteString("- Throughput covers chunking only. Embedding the finished chunks for the retrieval test is not included for either chunker.\n\n")

	sb.WriteString("## Per-question ranks\n\n")
	sb.WriteString("Rank of the first relevant chunk (1 is best, `-` means no chunk overlapped the answer). A `*` marks questions where a top-5 chunk holds the whole answer.\n\n")
	sb.WriteString("| # | Question | Source | Baseline | SmartChunk |\n|---:|---|---|---:|---:|\n")
	for i, q := range r.Questions {
		fmt.Fprintf(&sb, "| %d | %s | `%s:%d-%d` | %s | %s |\n", i+1, escapeCell(q.Question), q.SourceFile, q.AnswerStartLine, q.AnswerEndLine,
			rankCell(b.PerQuestion, i), rankCell(s.PerQuestion, i))
	}
	sb.WriteString("\n")
	fmt.Fprintf(&sb, "The visual comparison in `compare.html` shows `%s` chunked both ways.\n", r.CompareDoc)
	return sb.String()
}

func workersOf(r *Report) int {
	if r.Config.Workers > 0 {
		return r.Config.Workers
	}
	return runtime.NumCPU()
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func pct(x float64) string { return fmt.Sprintf("%.1f%%", 100*x) }

func ratio(n, total int) string {
	if total == 0 {
		return fmt.Sprintf("%d / 0", n)
	}
	return fmt.Sprintf("%d / %d (%.1f%%)", n, total, 100*float64(n)/float64(total))
}

func rate(x float64) string {
	switch {
	case x >= 100:
		return fmt.Sprintf("%.0f", x)
	case x >= 10:
		return fmt.Sprintf("%.1f", x)
	default:
		return fmt.Sprintf("%.2f", x)
	}
}

func dur(d interface{ Seconds() float64 }) string {
	s := d.Seconds()
	switch {
	case s >= 10:
		return fmt.Sprintf("%.1f s", s)
	case s >= 1:
		return fmt.Sprintf("%.2f s", s)
	default:
		return fmt.Sprintf("%.0f ms", s*1000)
	}
}

func escapeCell(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "|", `\|`), "\n", " ")
}

func rankCell(results []QuestionResult, i int) string {
	if i >= len(results) {
		return "-"
	}
	r := results[i]
	cell := "-"
	if r.Rank > 0 {
		cell = fmt.Sprint(r.Rank)
	}
	if r.Complete5 {
		cell += " *"
	}
	return cell
}
