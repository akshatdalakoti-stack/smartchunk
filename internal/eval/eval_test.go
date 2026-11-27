package eval

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/akshatdalakoti-stack/smartchunk/internal/baseline"
	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/pipeline"
)

func ids(hits []Hit) []int {
	out := make([]int, len(hits))
	for i, h := range hits {
		out[i] = h.ID
	}
	return out
}

func TestIndexSearch(t *testing.T) {
	// Unit vectors at 0, 30, 60, 90 and 180 degrees, plus a zero vector.
	deg := func(d float64) []float32 {
		r := d * math.Pi / 180
		return []float32{float32(math.Cos(r)), float32(math.Sin(r))}
	}
	ix := NewIndex([][]float32{deg(90), deg(0), deg(180), deg(30), {0, 0}, deg(60)})

	tests := []struct {
		name  string
		query []float32
		k     int
		want  []int
	}{
		{"nearest first", deg(0), 3, []int{1, 3, 5}},
		{"top one", deg(85), 1, []int{0}},
		{"query length is irrelevant", []float32{50, 0}, 2, []int{1, 3}},
		{"k of zero returns everything", deg(0), 0, []int{1, 3, 5, 0, 4, 2}},
		{"k above size returns everything", deg(0), 99, []int{1, 3, 5, 0, 4, 2}},
		{"ties go to the lower id", deg(45), 2, []int{3, 5}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ix.Search(tt.query, tt.k)
			if !reflect.DeepEqual(ids(got), tt.want) {
				t.Errorf("Search = %v, want %v", ids(got), tt.want)
			}
			for i := 1; i < len(got); i++ {
				if got[i].Score > got[i-1].Score {
					t.Errorf("scores are not descending: %v", got)
				}
			}
		})
	}
	if ix.Len() != 6 {
		t.Errorf("Len = %d", ix.Len())
	}
	if got := NewIndex(nil).Search(deg(0), 5); len(got) != 0 {
		t.Errorf("empty index returned %v", got)
	}
}

// TestSearchMatchesFullSort checks the heap-based top-k against a full
// ranking on random data.
func TestSearchMatchesFullSort(t *testing.T) {
	vecs, _ := embed.Fake{Dim: 16}.Embed(context.Background(), manyTexts(300))
	ix := NewIndex(vecs)
	for q := 0; q < 20; q++ {
		full := ix.Search(vecs[q*7], 0)
		for _, k := range []int{1, 5, 17} {
			top := ix.Search(vecs[q*7], k)
			if !reflect.DeepEqual(top, full[:k]) {
				t.Fatalf("query %d: top-%d differs from the head of the full ranking", q, k)
			}
		}
	}
}

func manyTexts(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("alpha%d beta%d gamma%d shared words here", i%17, i%5, i)
	}
	return out
}

func TestRelevantAndComplete(t *testing.T) {
	q := Question{SourceFile: "a.md", AnswerStartLine: 10, AnswerEndLine: 14}
	tests := []struct {
		name         string
		c            chunk.Chunk
		wantRelevant bool
		wantComplete bool
	}{
		{"covers the answer", chunk.Chunk{SourceFile: "a.md", StartLine: 5, EndLine: 20}, true, true},
		{"exactly the answer", chunk.Chunk{SourceFile: "a.md", StartLine: 10, EndLine: 14}, true, true},
		{"first half", chunk.Chunk{SourceFile: "a.md", StartLine: 5, EndLine: 12}, true, false},
		{"touches the last line", chunk.Chunk{SourceFile: "a.md", StartLine: 14, EndLine: 30}, true, false},
		{"inside the answer", chunk.Chunk{SourceFile: "a.md", StartLine: 11, EndLine: 12}, true, false},
		{"just before", chunk.Chunk{SourceFile: "a.md", StartLine: 1, EndLine: 9}, false, false},
		{"just after", chunk.Chunk{SourceFile: "a.md", StartLine: 15, EndLine: 20}, false, false},
		{"other file", chunk.Chunk{SourceFile: "b.md", StartLine: 5, EndLine: 20}, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Relevant(tt.c, q); got != tt.wantRelevant {
				t.Errorf("Relevant = %v, want %v", got, tt.wantRelevant)
			}
			if got := Complete(tt.c, q); got != tt.wantComplete {
				t.Errorf("Complete = %v, want %v", got, tt.wantComplete)
			}
		})
	}
}

// tableEmbedder returns a fixed vector for each known text.
type tableEmbedder map[string][]float32

func (e tableEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, ok := e[t]
		if !ok {
			return nil, fmt.Errorf("no vector for %q", t)
		}
		out[i] = v
	}
	return out, nil
}

func TestMeasureRetrieval(t *testing.T) {
	// Four chunks along the unit circle; questions aimed at known angles.
	deg := func(d float64) []float32 {
		r := d * math.Pi / 180
		return []float32{float32(math.Cos(r)), float32(math.Sin(r))}
	}
	chunks := []chunk.Chunk{
		{Text: "c0", SourceFile: "a.md", StartLine: 1, EndLine: 10},
		{Text: "c1", SourceFile: "a.md", StartLine: 11, EndLine: 20},
		{Text: "c2", SourceFile: "a.md", StartLine: 21, EndLine: 30},
		{Text: "c3", SourceFile: "b.md", StartLine: 1, EndLine: 10},
		{Text: "c4", SourceFile: "b.md", StartLine: 11, EndLine: 20},
		{Text: "c5", SourceFile: "b.md", StartLine: 21, EndLine: 30},
	}
	emb := tableEmbedder{
		"c0": deg(0), "c1": deg(20), "c2": deg(40), "c3": deg(60), "c4": deg(80), "c5": deg(100),
		"q-first":   deg(1),  // nearest c0
		"q-second":  deg(1),  // nearest c0, answer in c1 -> rank 2
		"q-sixth":   deg(0),  // answer in c5 -> rank 6
		"q-missing": deg(0),  // answer lines no chunk covers
		"q-partial": deg(21), // nearest c1, answer spans c1 and c2
	}
	questions := []Question{
		{Question: "q-first", SourceFile: "a.md", AnswerStartLine: 2, AnswerEndLine: 4},
		{Question: "q-second", SourceFile: "a.md", AnswerStartLine: 12, AnswerEndLine: 13},
		{Question: "q-sixth", SourceFile: "b.md", AnswerStartLine: 25, AnswerEndLine: 26},
		{Question: "q-missing", SourceFile: "a.md", AnswerStartLine: 40, AnswerEndLine: 41},
		{Question: "q-partial", SourceFile: "a.md", AnswerStartLine: 18, AnswerEndLine: 23},
	}
	textOf := func(c chunk.Chunk) string { return c.Text }

	got, per, err := MeasureRetrieval(context.Background(), chunks, questions, emb, textOf)
	if err != nil {
		t.Fatalf("MeasureRetrieval: %v", err)
	}
	wantRanks := []int{1, 2, 6, 0, 1}
	wantComplete := []bool{true, true, false, false, false}
	for i, r := range per {
		if r.Rank != wantRanks[i] || r.Complete5 != wantComplete[i] {
			t.Errorf("question %q: rank %d complete %v, want rank %d complete %v",
				questions[i].Question, r.Rank, r.Complete5, wantRanks[i], wantComplete[i])
		}
	}
	want := Retrieval{
		Questions: 5,
		Hit1:      2.0 / 5,
		Hit3:      3.0 / 5,
		Hit5:      3.0 / 5,
		MRR:       (1 + 0.5 + 1.0/6 + 0 + 1) / 5,
		Complete5: 2.0 / 5,
	}
	if math.Abs(got.Hit1-want.Hit1) > 1e-9 || math.Abs(got.Hit3-want.Hit3) > 1e-9 || math.Abs(got.Hit5-want.Hit5) > 1e-9 ||
		math.Abs(got.MRR-want.MRR) > 1e-9 || math.Abs(got.Complete5-want.Complete5) > 1e-9 || got.Questions != want.Questions {
		t.Errorf("Retrieval\n got: %+v\nwant: %+v", got, want)
	}

	t.Run("errors", func(t *testing.T) {
		if _, _, err := MeasureRetrieval(context.Background(), chunks, questions, nil, textOf); err == nil {
			t.Error("nil embedder should be rejected")
		}
		unknown := []Question{{Question: "never seen", SourceFile: "a.md", AnswerStartLine: 1, AnswerEndLine: 1}}
		if _, _, err := MeasureRetrieval(context.Background(), chunks, unknown, emb, textOf); err == nil {
			t.Error("embedder failure should be reported")
		}
		r, per, err := MeasureRetrieval(context.Background(), chunks, nil, emb, textOf)
		if err != nil || r.Questions != 0 || per != nil {
			t.Errorf("no questions: %+v %v %v", r, per, err)
		}
	})
}

func TestMeasureTokens(t *testing.T) {
	mk := func(counts ...int) []chunk.Chunk {
		out := make([]chunk.Chunk, len(counts))
		for i, n := range counts {
			out[i] = chunk.Chunk{TokenCount: n}
		}
		return out
	}
	hundred := make([]int, 100)
	for i := range hundred {
		hundred[i] = i + 1
	}
	tests := []struct {
		name   string
		chunks []chunk.Chunk
		want   TokenStats
	}{
		{"none", nil, TokenStats{}},
		{"one", mk(42), TokenStats{Mean: 42, P95: 42, Max: 42}},
		{"unsorted input", mk(30, 10, 20), TokenStats{Mean: 20, P95: 30, Max: 30}},
		{"one to a hundred", mk(hundred...), TokenStats{Mean: 50.5, P95: 95, Max: 100}},
		{"outlier sets max, not p95", mk(append(make([]int, 99), 1000)...), TokenStats{Mean: 10, P95: 0, Max: 1000}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MeasureTokens(tt.chunks); got != tt.want {
				t.Errorf("MeasureTokens = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestMeasureDuplicates(t *testing.T) {
	long := func(seed string) string {
		var sb strings.Builder
		for i := 0; i < 60; i++ {
			fmt.Fprintf(&sb, "%s%d ", seed, i)
		}
		return sb.String()
	}
	tests := []struct {
		name      string
		texts     []string
		threshold float64
		wantCount int
		wantErr   bool
	}{
		{"none", nil, 0.85, 0, false},
		{"all distinct", []string{long("a"), long("b"), long("c")}, 0.85, 0, false},
		{"two copies of one", []string{long("a"), long("b"), long("a"), long("a")}, 0.85, 2, false},
		{"bad threshold", []string{long("a")}, 2, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks := make([]chunk.Chunk, len(tt.texts))
			for i, text := range tt.texts {
				chunks[i] = chunk.Chunk{Text: text}
			}
			count, rate, err := MeasureDuplicates(chunks, tt.threshold)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}
			if count != tt.wantCount {
				t.Errorf("count = %d, want %d", count, tt.wantCount)
			}
			if len(chunks) > 0 && math.Abs(rate-float64(tt.wantCount)/float64(len(chunks))) > 1e-9 {
				t.Errorf("rate = %v", rate)
			}
		})
	}
}

func TestLoadQuestions(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    int
		wantErr string
	}{
		{
			name:    "valid with blank lines",
			content: "{\"question\":\"q1\",\"source_file\":\"a.md\",\"answer_start_line\":1,\"answer_end_line\":3}\n\n{\"question\":\"q2\",\"source_file\":\"sub\\\\b.md\",\"answer_start_line\":5,\"answer_end_line\":5}\n",
			want:    2,
		},
		{"empty file", "", 0, "no questions"},
		{"bad json", "{nope}\n", 0, "line 1"},
		{"missing question", "{\"source_file\":\"a.md\",\"answer_start_line\":1,\"answer_end_line\":1}\n", 0, "question is empty"},
		{"missing file", "{\"question\":\"q\",\"answer_start_line\":1,\"answer_end_line\":1}\n", 0, "source_file is empty"},
		{"zero start", "{\"question\":\"q\",\"source_file\":\"a.md\",\"answer_start_line\":0,\"answer_end_line\":1}\n", 0, "not a valid range"},
		{"end before start", "{\"question\":\"q\",\"source_file\":\"a.md\",\"answer_start_line\":5,\"answer_end_line\":4}\n", 0, "not a valid range"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "q.jsonl")
			if err := os.WriteFile(path, []byte(tt.content), 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := LoadQuestions(path)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("err = %v, want it to mention %q", err, tt.wantErr)
				}
				return
			}
			if err != nil || len(got) != tt.want {
				t.Fatalf("got %d questions, err %v", len(got), err)
			}
			if got[1].SourceFile != "sub/b.md" {
				t.Errorf("backslashes were not normalised: %q", got[1].SourceFile)
			}
		})
	}
	if _, err := LoadQuestions(filepath.Join(t.TempDir(), "missing.jsonl")); err == nil {
		t.Error("missing file should be an error")
	}
}

// writeCorpus creates a small corpus where the two chunkers behave
// differently: long code blocks, a table and a list for the baseline to cut.
func writeCorpus(t *testing.T) (corpus, questions string) {
	t.Helper()
	root := t.TempDir()
	corpus = filepath.Join(root, "corpus")
	var a strings.Builder
	a.WriteString("# Widget guide\n\nWidgets are configured through a single file that lives next to the binary.\n\n")
	a.WriteString("## Configuration\n\nEvery widget reads the keys below when it starts and refuses to run when one is missing.\n\n")
	a.WriteString("```yaml\n")
	for i := 0; i < 30; i++ {
		fmt.Fprintf(&a, "widget_option_%d: value_%d\n", i, i)
	}
	a.WriteString("```\n\n## Limits\n\n| Limit | Value |\n|-------|-------|\n")
	for i := 0; i < 12; i++ {
		fmt.Fprintf(&a, "| quota_%d | %d units |\n", i, i*10)
	}
	a.WriteString("\n## Troubleshooting\n\nWhen a widget refuses to start, check these things in order:\n\n")
	for i := 0; i < 12; i++ {
		fmt.Fprintf(&a, "- symptom_%d usually means cause_%d and is fixed by remedy_%d\n", i, i, i)
	}
	b := "# Gadget notes\n\nGadgets are unrelated to widgets and ship separately.\n\n## Shipping\n\nGadgets ship by sea freight in sealed crates every second month.\n"
	files := map[string]string{"guide/widgets.md": a.String(), "gadgets.md": b}
	for name, content := range files {
		path := filepath.Join(corpus, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	questions = filepath.Join(root, "questions.jsonl")
	q := "{\"question\":\"How do gadgets ship by sea freight in crates?\",\"source_file\":\"gadgets.md\",\"answer_start_line\":7,\"answer_end_line\":7}\n" +
		"{\"question\":\"What does symptom_3 mean and which remedy_3 fixes it?\",\"source_file\":\"guide/widgets.md\",\"answer_start_line\":63,\"answer_end_line\":74}\n"
	if err := os.WriteFile(questions, []byte(q), 0o644); err != nil {
		t.Fatal(err)
	}
	return corpus, questions
}

func testConfig(t *testing.T) Config {
	corpus, questions := writeCorpus(t)
	return Config{
		CorpusDir:     corpus,
		QuestionsPath: questions,
		OutDir:        filepath.Join(t.TempDir(), "report"),
		Smart:         pipeline.Options{MaxTokens: 60, MinTokens: 15, Embedder: embed.Fake{}},
		Baseline:      baseline.Options{Size: 60, Overlap: 10},
		Embedder:      embed.Fake{},
		EmbedderName:  "fake",
		Workers:       2,
		Now:           func() time.Time { return time.Date(2026, 1, 2, 3, 4, 0, 0, time.UTC) },
	}
}

func TestRunEndToEnd(t *testing.T) {
	cfg := testConfig(t)
	rep, err := Run(context.Background(), cfg)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(rep.Docs) != 2 || len(rep.Questions) != 2 {
		t.Fatalf("docs %d questions %d", len(rep.Docs), len(rep.Questions))
	}
	b, s := rep.Baseline, rep.Smart
	if len(b.Chunks) == 0 || len(s.Chunks) == 0 {
		t.Fatal("a chunker produced nothing")
	}
	// The corpus has one code block, one table and one list, all larger
	// than the window: the baseline must cut all three and SmartChunk none.
	if b.Structure.Violations() != 3 {
		t.Errorf("baseline violations = %d, want 3 (%+v)", b.Structure.Violations(), b.Structure)
	}
	if s.Structure.Violations() != 0 || s.Structure.OrphanHeadings != 0 {
		t.Errorf("smartchunk damaged structure: %+v", s.Structure)
	}
	if s.Structure.AtomicBlocks() != 3 || b.Structure.AtomicBlocks() != 3 {
		t.Errorf("atomic block totals: %d and %d, want 3", s.Structure.AtomicBlocks(), b.Structure.AtomicBlocks())
	}
	if s.Oversized != 3 {
		t.Errorf("smartchunk flagged %d chunks oversized, want 3", s.Oversized)
	}
	if b.Tokens.Max > 60 {
		t.Errorf("baseline max tokens = %d, over its window", b.Tokens.Max)
	}
	if s.Retrieval.Questions != 2 || b.Retrieval.Questions != 2 {
		t.Errorf("retrieval question counts: %d, %d", s.Retrieval.Questions, b.Retrieval.Questions)
	}
	if s.Retrieval.Complete5 != 1 {
		t.Errorf("smartchunk should return both whole answers in the top 5, got %v", s.Retrieval.Complete5)
	}
	if b.DocsPerSec <= 0 || s.ChunksPerSec <= 0 {
		t.Errorf("throughput not measured: %v %v", b.DocsPerSec, s.ChunksPerSec)
	}
	if rep.CompareDoc != "guide/widgets.md" {
		t.Errorf("compare doc = %q, want the document the baseline damages", rep.CompareDoc)
	}

	results, err := os.ReadFile(filepath.Join(cfg.OutDir, "results.md"))
	if err != nil {
		t.Fatalf("results.md: %v", err)
	}
	for _, want := range []string{
		"| Metric | Baseline | SmartChunk |",
		"| Code blocks cut | 1 / 1 (100.0%) | 0 / 1 (0.0%) |",
		"| Tables cut | 1 / 1 (100.0%) | 0 / 1 (0.0%) |",
		"| Lists cut | 1 / 1 (100.0%) | 0 / 1 (0.0%) |",
		"| Orphan headings |", "| Hit rate @1 |", "| Hit rate @3 |", "| Hit rate @5 |", "| MRR |",
		"| Mean |", "| p95 |", "| Max |", "| Docs/sec |", "| Chunks/sec |",
		"Near-duplicate chunks (Jaccard >= 0.85)",
		"2026-01-02 03:04 UTC", "`guide/widgets.md:63-74`", "fixed 60 tokens, overlap 10", "max 60, min 15 tokens",
	} {
		if !strings.Contains(string(results), want) {
			t.Errorf("results.md is missing %q", want)
		}
	}

	html, err := os.ReadFile(filepath.Join(cfg.OutDir, "compare.html"))
	if err != nil {
		t.Fatalf("compare.html: %v", err)
	}
	for _, want := range []string{
		"<!doctype html>", "guide/widgets.md", "Baseline", "SmartChunk",
		"cuts a code block", "cuts a table", "cuts a list",
		"t-code", "t-table", "t-prose", "oversized", "widget_option_29",
	} {
		if !strings.Contains(string(html), want) {
			t.Errorf("compare.html is missing %q", want)
		}
	}
	if strings.Contains(string(html), "gadgets ship") {
		t.Error("compare.html shows a document other than the chosen one")
	}
}

func TestRunDeterministic(t *testing.T) {
	cfg := testConfig(t)
	a, err := Run(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Run(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Smart.Chunks, b.Smart.Chunks) || !reflect.DeepEqual(a.Baseline.Chunks, b.Baseline.Chunks) {
		t.Error("two runs produced different chunks")
	}
	if a.Smart.Retrieval != b.Smart.Retrieval || a.Baseline.Retrieval != b.Baseline.Retrieval {
		t.Error("two runs produced different retrieval scores")
	}
}

func TestRunOptionsAndErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("no heading context is reported", func(t *testing.T) {
		cfg := testConfig(t)
		cfg.NoHeadingContext = true
		cfg.CompareDoc = "gadgets.md"
		rep, err := Run(ctx, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if rep.CompareDoc != "gadgets.md" {
			t.Errorf("compare doc = %q", rep.CompareDoc)
		}
		results, _ := os.ReadFile(filepath.Join(cfg.OutDir, "results.md"))
		if !strings.Contains(string(results), "--no-heading-context") {
			t.Error("results.md does not mention the ablation")
		}
	})
	t.Run("cache statistics are reported", func(t *testing.T) {
		cfg := testConfig(t)
		cached, err := embed.NewCached(embed.Fake{}, "fake", filepath.Join(t.TempDir(), "c.bin"))
		if err != nil {
			t.Fatal(err)
		}
		defer cached.Close()
		cfg.Smart.Embedder, cfg.Embedder, cfg.EmbedStats = cached, cached, cached.Stats
		rep, err := Run(ctx, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if rep.Smart.CacheMisses == 0 || rep.Smart.CacheHits != 0 {
			t.Errorf("cold run: %d hits, %d misses", rep.Smart.CacheHits, rep.Smart.CacheMisses)
		}
		rep2, err := Run(ctx, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if rep2.Smart.CacheMisses != 0 || rep2.Smart.CacheHits != rep.Smart.CacheMisses {
			t.Errorf("warm run: %d hits, %d misses", rep2.Smart.CacheHits, rep2.Smart.CacheMisses)
		}
	})

	errorCases := []struct {
		name    string
		mutate  func(t *testing.T, c *Config)
		wantErr string
	}{
		{"no embedder", func(t *testing.T, c *Config) { c.Embedder = nil }, "embedder is required"},
		{"no output directory", func(t *testing.T, c *Config) { c.OutDir = "" }, "output directory"},
		{"missing corpus", func(t *testing.T, c *Config) { c.CorpusDir = filepath.Join(t.TempDir(), "nope") }, "nope"},
		{"empty corpus", func(t *testing.T, c *Config) { c.CorpusDir = t.TempDir() }, "no Markdown files"},
		{"missing questions", func(t *testing.T, c *Config) { c.QuestionsPath = filepath.Join(t.TempDir(), "q.jsonl") }, "opening questions"},
		{"unknown compare document", func(t *testing.T, c *Config) { c.CompareDoc = "nope.md" }, "not in the corpus"},
		{"invalid baseline", func(t *testing.T, c *Config) { c.Baseline.Size = 0 }, "baseline"},
		{"invalid sizing", func(t *testing.T, c *Config) { c.Smart.MinTokens = 500 }, "smartchunk"},
		{"question for a file outside the corpus", func(t *testing.T, c *Config) {
			os.WriteFile(c.QuestionsPath, []byte("{\"question\":\"q\",\"source_file\":\"other.md\",\"answer_start_line\":1,\"answer_end_line\":1}\n"), 0o644)
		}, "not in the corpus"},
		{"answer past the end of the file", func(t *testing.T, c *Config) {
			os.WriteFile(c.QuestionsPath, []byte("{\"question\":\"q\",\"source_file\":\"gadgets.md\",\"answer_start_line\":1,\"answer_end_line\":999}\n"), 0o644)
		}, "has 7 lines"},
		{"embedder failure", func(t *testing.T, c *Config) { c.Embedder = tableEmbedder{} }, "no vector"},
	}
	for _, tt := range errorCases {
		t.Run(tt.name, func(t *testing.T) {
			cfg := testConfig(t)
			tt.mutate(t, &cfg)
			_, err := Run(ctx, cfg)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("err = %v, want it to mention %q", err, tt.wantErr)
			}
		})
	}
	t.Run("cancelled context", func(t *testing.T) {
		cctx, cancel := context.WithCancel(ctx)
		cancel()
		if _, err := Run(cctx, testConfig(t)); !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})
}

func TestTypeByOverlapAndStrandedHeading(t *testing.T) {
	blocks, err := parse.Parse([]byte(structDoc))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name       string
		start, end int
		wantType   chunk.Type
		wantStrand bool
	}{
		{"prose only", 1, 3, chunk.Prose, false},
		{"inside the code block", 6, 7, chunk.Code, false},
		{"inside the table", 13, 14, chunk.Table, false},
		{"list is prose", 17, 19, chunk.Prose, false},
		{"prose and code", 3, 6, chunk.Mixed, false},
		{"ends on a heading", 5, 11, chunk.Code, true},
		{"heading with its content", 11, 15, chunk.Table, false},
		{"final heading has nothing to be cut from", 17, 21, chunk.Prose, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := chunk.Chunk{StartLine: tt.start, EndLine: tt.end}
			if got := typeByOverlap(blocks, c); got != tt.wantType {
				t.Errorf("typeByOverlap = %q, want %q", got, tt.wantType)
			}
			if got := strandsHeading(blocks, c); got != tt.wantStrand {
				t.Errorf("strandsHeading = %v, want %v", got, tt.wantStrand)
			}
		})
	}
}

func TestFormatHelpers(t *testing.T) {
	tests := []struct {
		got, want string
	}{
		{pct(0.625), "62.5%"},
		{pct(0), "0.0%"},
		{ratio(3, 12), "3 / 12 (25.0%)"},
		{ratio(0, 0), "0 / 0"},
		{rate(1234.5), "1234"},
		{rate(45.67), "45.7"},
		{rate(3.14159), "3.14"},
		{dur(1500 * time.Millisecond), "1.50 s"},
		{dur(42 * time.Second), "42.0 s"},
		{dur(250 * time.Millisecond), "250 ms"},
		{escapeCell("a|b\nc"), `a\|b c`},
		{orDash(""), "-"},
		{rankCell([]QuestionResult{{Rank: 2, Complete5: true}}, 0), "2 *"},
		{rankCell([]QuestionResult{{}}, 0), "-"},
		{rankCell(nil, 3), "-"},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("got %q, want %q", tt.got, tt.want)
		}
	}
}
