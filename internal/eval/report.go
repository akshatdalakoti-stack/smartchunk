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

// compareCard is one chunk as drawn in compare.html.
type compareCard struct {
	Number  int
	Type    string
	Lines   string
	Tokens  int
	Path    string
	Flags   []string
	Damage  []string
	Text    string
	Damaged bool
}

type compareColumn struct {
	Name     string
	Settings string
	Summary  string
	Clean    bool
	Cards    []compareCard
}

type compareData struct {
	Doc     string
	Columns []compareColumn
}

// CompareHTML renders one document chunked by both systems, side by side.
// Every chunk is drawn as a card whose colour shows its type; chunks that
// cut a block or strand a heading are marked.
func CompareHTML(r *Report, blocks []parse.Block) ([]byte, error) {
	data := compareData{Doc: r.CompareDoc}
	for _, sys := range []System{r.Baseline, r.Smart} {
		var chunks []chunk.Chunk
		for _, c := range sys.Chunks {
			if c.SourceFile == r.CompareDoc {
				chunks = append(chunks, c)
			}
		}
		st := MeasureStructure(blocks, chunks)
		col := compareColumn{
			Name:     sys.Name,
			Settings: sys.Settings,
			Summary:  fmt.Sprintf("%d chunks, %d blocks cut, %d orphan headings", len(chunks), st.Violations(), st.OrphanHeadings),
			Clean:    st.Violations() == 0 && st.OrphanHeadings == 0,
		}
		for i, c := range chunks {
			card := compareCard{
				Number: i + 1,
				Type:   string(c.ChunkType),
				Lines:  fmt.Sprintf("%d-%d", c.StartLine, c.EndLine),
				Tokens: c.TokenCount,
				Path:   strings.Join(c.HeadingPath, " > "),
				Flags:  c.Flags,
				Text:   c.Text,
			}
			if sys.Name == r.Baseline.Name {
				// The baseline calls everything prose. Colour its chunks
				// by what they really contain.
				card.Type = string(typeByOverlap(blocks, c))
			}
			for _, t := range CutBy(blocks, c) {
				card.Damage = append(card.Damage, "cuts a "+blockLabel(t))
			}
			if strandsHeading(blocks, c) {
				card.Damage = append(card.Damage, "ends on a heading")
			}
			card.Damaged = len(card.Damage) > 0
			col.Cards = append(col.Cards, card)
		}
		data.Columns = append(data.Columns, col)
	}
	var buf bytes.Buffer
	if err := compareTemplate.Execute(&buf, data); err != nil {
		return nil, fmt.Errorf("eval: rendering compare.html: %w", err)
	}
	return buf.Bytes(), nil
}

func blockLabel(t parse.BlockType) string {
	if t == parse.Code {
		return "code block"
	}
	return t.String()
}

// typeByOverlap classifies a chunk by the blocks its line range touches.
func typeByOverlap(blocks []parse.Block, c chunk.Chunk) chunk.Type {
	var code, table, prose int
	for _, b := range blocks {
		if c.StartLine > b.EndLine || c.EndLine < b.StartLine {
			continue
		}
		switch b.Type {
		case parse.Heading:
		case parse.Code:
			code++
		case parse.Table:
			table++
		default:
			prose++
		}
	}
	switch {
	case code > 0 && table == 0 && prose == 0:
		return chunk.Code
	case table > 0 && code == 0 && prose == 0:
		return chunk.Table
	case code == 0 && table == 0:
		return chunk.Prose
	default:
		return chunk.Mixed
	}
}

// strandsHeading reports whether the chunk's last block is a heading whose
// content starts after the chunk ends.
func strandsHeading(blocks []parse.Block, c chunk.Chunk) bool {
	for i, b := range blocks {
		if b.Type != parse.Heading || i+1 >= len(blocks) {
			continue
		}
		if b.StartLine >= c.StartLine && b.StartLine <= c.EndLine && blocks[i+1].StartLine > c.EndLine {
			return true
		}
	}
	return false
}

var compareTemplate = template.Must(template.New("compare").Parse(`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartChunk comparison: {{.Doc}}</title>
<style>
:root {
  --bg: #f5f4f0; --panel: #ffffff; --ink: #1d2126; --muted: #5f6b7a; --line: #dedbd2;
  --code: #2b6cd9; --table: #0b8f6c; --prose: #7b8492; --mixed: #a8479a; --bad: #c0362c;
  --bad-bg: #fdecea; --ok: #0b8f6c;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #131519; --panel: #1b1e24; --ink: #e5e7ea; --muted: #98a2ae; --line: #2b3038;
    --code: #73a5ff; --table: #3fcaa0; --prose: #8a93a0; --mixed: #dc8bcb; --bad: #ff8478;
    --bad-bg: #3a1d1b; --ok: #3fcaa0;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
header.page { padding: 24px 24px 8px; max-width: 1500px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
.doc { color: var(--muted); font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 14px; color: var(--muted); font-size: 13px; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; background: var(--c); }
main { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 16px 24px 48px; max-width: 1500px; margin: 0 auto; align-items: start; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; padding: 16px; } header.page { padding: 16px 16px 8px; } }
.col > h2 { position: sticky; top: 0; z-index: 1; background: var(--bg); margin: 0; padding: 10px 0; font-size: 16px; border-bottom: 2px solid var(--line); }
.col > h2 small { display: block; font-weight: 400; color: var(--muted); font-size: 12.5px; }
.col > h2 .sum { color: var(--bad); }
.col > h2 .sum.clean { color: var(--ok); }
.chunk { background: var(--panel); border: 1px solid var(--line); border-left: 5px solid var(--c); border-radius: 6px; margin-top: 12px; overflow: hidden; }
.chunk.damaged { border-color: var(--bad); border-left-color: var(--c); }
.meta { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; padding: 7px 12px; border-bottom: 1px solid var(--line); font-size: 12px; color: var(--muted); }
.meta b { color: var(--c); text-transform: uppercase; letter-spacing: .04em; }
.tag { border: 1px solid var(--line); border-radius: 10px; padding: 0 8px; }
.tag.bad { background: var(--bad-bg); border-color: var(--bad); color: var(--bad); font-weight: 600; }
.path { padding: 5px 12px 0; font-size: 12px; color: var(--muted); }
pre { margin: 0; padding: 10px 12px 12px; white-space: pre-wrap; overflow-wrap: anywhere; font: 12.5px/1.5 ui-monospace, "Cascadia Code", Consolas, monospace; }
.t-code { --c: var(--code); } .t-table { --c: var(--table); } .t-prose { --c: var(--prose); } .t-mixed { --c: var(--mixed); }
</style>
</head>
<body>
<header class="page">
  <h1>One document, two chunkers</h1>
  <div class="doc">{{.Doc}}</div>
  <div class="legend">
    <span class="t-code">code</span><span class="t-table">table</span><span class="t-prose">prose</span><span class="t-mixed">mixed</span>
    <span style="--c: var(--bad)">red outline: the chunk cuts a block or strands a heading</span>
  </div>
</header>
<main>
{{range .Columns}}
  <section class="col">
    <h2>{{.Name}} <small>{{.Settings}}</small><small class="sum{{if .Clean}} clean{{end}}">{{.Summary}}</small></h2>
    {{range .Cards}}
    <article class="chunk t-{{.Type}}{{if .Damaged}} damaged{{end}}">
      <div class="meta">
        <b>{{.Type}}</b><span>#{{.Number}}</span><span>lines {{.Lines}}</span><span>{{.Tokens}} tokens</span>
        {{range .Flags}}<span class="tag">{{.}}</span>{{end}}
        {{range .Damage}}<span class="tag bad">{{.}}</span>{{end}}
      </div>
      {{if .Path}}<div class="path">{{.Path}}</div>{{end}}
      <pre>{{.Text}}</pre>
    </article>
    {{end}}
  </section>
{{end}}
</main>
</body>
</html>
`))
