package eval

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/akshatdalakoti-stack/smartchunk/internal/baseline"
	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/dedup"
	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/export"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/pipeline"
)

// Config describes one evaluation run.
type Config struct {
	// CorpusDir is the directory of Markdown documents to chunk.
	CorpusDir string
	// QuestionsPath is the JSONL questions file.
	QuestionsPath string
	// OutDir receives results.md and compare.html.
	OutDir string

	// Smart configures the SmartChunk pipeline, including the embedder of
	// its semantic stage.
	Smart pipeline.Options
	// Baseline configures the fixed-size splitter.
	Baseline baseline.Options

	// Embedder embeds chunks and questions for the retrieval test.
	Embedder embed.Embedder
	// EmbedderName describes the embedder in the report.
	EmbedderName string
	// EmbedStats, when set, returns the embedding cache's running hit and
	// miss counts, so the report can say how warm the cache was while
	// throughput was measured.
	EmbedStats func() (hits, misses int)

	// DuplicateThreshold is the Jaccard similarity at which two chunks
	// count as near-duplicates when measuring the duplicate rate. Zero
	// means the dedup default.
	DuplicateThreshold float64
	// NoHeadingContext embeds the bare chunk text instead of the heading
	// breadcrumb plus text.
	NoHeadingContext bool
	// CompareDoc picks the document shown in compare.html. Empty means the
	// document where the baseline does the most structural damage.
	CompareDoc string
	// Workers is the number of documents processed at once.
	Workers int
	// Now stamps the report. Nil means time.Now.
	Now func() time.Time
}

// System holds every measurement for one chunker.
type System struct {
	Name        string
	Settings    string
	Docs        int
	Chunks      []chunk.Chunk
	Structure   Structural
	Duplicates  int
	DupRate     float64
	Retrieval   Retrieval
	PerQuestion []QuestionResult
	Tokens      TokenStats
	Oversized   int
	// Elapsed is the wall-clock time the chunker took over the corpus.
	Elapsed      time.Duration
	DocsPerSec   float64
	ChunksPerSec float64
	// CacheHits and CacheMisses count embedding cache lookups made while
	// chunking (only the semantic stage embeds during chunking).
	CacheHits, CacheMisses int
}

// Report is the outcome of an evaluation run.
type Report struct {
	Config    Config
	Generated time.Time
	Docs      []pipeline.Doc
	Questions []Question
	Baseline  System
	Smart     System
	// StructureOnly is SmartChunk timed again without the semantic stage,
	// to show the cost of chunking apart from the cost of embedding.
	StructureOnlyElapsed time.Duration
	// CompareDoc is the document rendered in compare.html.
	CompareDoc string
}

// Run chunks the corpus with both splitters, measures them, and writes
// results.md and compare.html into cfg.OutDir.
func Run(ctx context.Context, cfg Config) (*Report, error) {
	if cfg.Embedder == nil {
		return nil, errors.New("eval: an embedder is required for the retrieval test")
	}
	if cfg.OutDir == "" {
		return nil, errors.New("eval: output directory is empty")
	}
	if cfg.DuplicateThreshold == 0 {
		cfg.DuplicateThreshold = dedup.DefaultThreshold
	}
	now := time.Now
	if cfg.Now != nil {
		now = cfg.Now
	}

	docs, err := pipeline.Discover(cfg.CorpusDir)
	if err != nil {
		return nil, fmt.Errorf("eval: %w", err)
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("eval: no Markdown files found in %s (run scripts/fetch_corpus.sh?)", cfg.CorpusDir)
	}
	questions, err := LoadQuestions(cfg.QuestionsPath)
	if err != nil {
		return nil, err
	}

	// Read and parse every document once. The parse is the yardstick both
	// chunkers are judged against.
	blocks := make(map[string][]parse.Block, len(docs))
	lineCount := make(map[string]int, len(docs))
	for i := range docs {
		src, err := docs[i].Read()
		if err != nil {
			return nil, fmt.Errorf("eval: %w", err)
		}
		docs[i].Source = src
		b, err := parse.Parse(src)
		if err != nil {
			return nil, fmt.Errorf("eval: parsing %s: %w", docs[i].Name, err)
		}
		blocks[docs[i].Name] = b
		lineCount[docs[i].Name] = countLines(src)
	}
	for i, q := range questions {
		n, ok := lineCount[q.SourceFile]
		if !ok {
			return nil, fmt.Errorf("eval: question %d refers to %q, which is not in the corpus", i+1, q.SourceFile)
		}
		if q.AnswerEndLine > n {
			return nil, fmt.Errorf("eval: question %d: answer ends at line %d but %s has %d lines", i+1, q.AnswerEndLine, q.SourceFile, n)
		}
	}

	report := &Report{Config: cfg, Generated: now(), Docs: docs, Questions: questions}

	// Baseline.
	start := time.Now()
	baseChunks, err := pipeline.RunBaseline(ctx, docs, cfg.Baseline, cfg.Workers)
	if err != nil {
		return nil, fmt.Errorf("eval: baseline: %w", err)
	}
	report.Baseline = System{
		Name:     "Baseline",
		Settings: fmt.Sprintf("fixed %d tokens, overlap %d", cfg.Baseline.Size, cfg.Baseline.Overlap),
		Chunks:   baseChunks,
		Elapsed:  time.Since(start),
	}

	// SmartChunk.
	smartOpt := cfg.Smart
	smartOpt.Workers = cfg.Workers
	hits0, misses0 := 0, 0
	if cfg.EmbedStats != nil {
		hits0, misses0 = cfg.EmbedStats()
	}
	start = time.Now()
	smartChunks, _, err := pipeline.Run(ctx, docs, smartOpt)
	if err != nil {
		return nil, fmt.Errorf("eval: smartchunk: %w", err)
	}
	report.Smart = System{
		Name:     "SmartChunk",
		Settings: smartSettings(smartOpt),
		Chunks:   smartChunks,
		Elapsed:  time.Since(start),
	}
	if cfg.EmbedStats != nil {
		hits1, misses1 := cfg.EmbedStats()
		report.Smart.CacheHits, report.Smart.CacheMisses = hits1-hits0, misses1-misses0
	}

	// SmartChunk again with the semantic stage off, for timing only.
	structOpt := smartOpt
	structOpt.Embedder = nil
	start = time.Now()
	if _, _, err := pipeline.Run(ctx, docs, structOpt); err != nil {
		return nil, fmt.Errorf("eval: smartchunk without semantic stage: %w", err)
	}
	report.StructureOnlyElapsed = time.Since(start)

	textOf := export.EmbeddingText
	if cfg.NoHeadingContext {
		textOf = func(c chunk.Chunk) string { return c.Text }
	}
	for _, sys := range []*System{&report.Baseline, &report.Smart} {
		if err := measure(ctx, sys, docs, blocks, questions, cfg, textOf); err != nil {
			return nil, err
		}
	}

	report.CompareDoc, err = pickCompareDoc(cfg.CompareDoc, docs, blocks, lineCount, report.Baseline.Chunks)
	if err != nil {
		return nil, err
	}

	if err := os.MkdirAll(cfg.OutDir, 0o755); err != nil {
		return nil, fmt.Errorf("eval: creating %s: %w", cfg.OutDir, err)
	}
	if err := os.WriteFile(filepath.Join(cfg.OutDir, "results.md"), []byte(ResultsMarkdown(report)), 0o644); err != nil {
		return nil, fmt.Errorf("eval: writing results.md: %w", err)
	}
	html, err := CompareHTML(report, blocks[report.CompareDoc])
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(cfg.OutDir, "compare.html"), html, 0o644); err != nil {
		return nil, fmt.Errorf("eval: writing compare.html: %w", err)
	}
	return report, nil
}

func smartSettings(o pipeline.Options) string {
	max, min := o.MaxTokens, o.MinTokens
	if max == 0 {
		max = 256
	}
	if min == 0 {
		min = 64
	}
	if min < 0 {
		min = 0
	}
	s := fmt.Sprintf("max %d, min %d tokens", max, min)
	if o.Embedder == nil {
		s += ", no semantic stage"
	}
	return s
}

// measure fills in every metric of a system from its chunks.
func measure(ctx context.Context, sys *System, docs []pipeline.Doc, blocks map[string][]parse.Block, questions []Question, cfg Config, textOf func(chunk.Chunk) string) error {
	sys.Docs = len(docs)
	byFile := map[string][]chunk.Chunk{}
	for _, c := range sys.Chunks {
		byFile[c.SourceFile] = append(byFile[c.SourceFile], c)
		if c.HasFlag(chunk.FlagOversized) {
			sys.Oversized++
		}
	}
	for _, d := range docs {
		sys.Structure.Add(MeasureStructure(blocks[d.Name], byFile[d.Name]))
	}

	var err error
	sys.Duplicates, sys.DupRate, err = MeasureDuplicates(sys.Chunks, cfg.DuplicateThreshold)
	if err != nil {
		return err
	}
	sys.Tokens = MeasureTokens(sys.Chunks)

	// A tiny corpus can finish inside one tick of the system clock. Treat
	// that as one microsecond rather than report an infinite rate.
	if sys.Elapsed < time.Microsecond {
		sys.Elapsed = time.Microsecond
	}
	secs := sys.Elapsed.Seconds()
	sys.DocsPerSec = float64(len(docs)) / secs
	sys.ChunksPerSec = float64(len(sys.Chunks)) / secs

	sys.Retrieval, sys.PerQuestion, err = MeasureRetrieval(ctx, sys.Chunks, questions, cfg.Embedder, textOf)
	if err != nil {
		return fmt.Errorf("%s: %w", sys.Name, err)
	}
	return nil
}

func countLines(src []byte) int {
	if len(src) == 0 {
		return 0
	}
	n := 1
	for i, b := range src {
		if b == '\n' && i < len(src)-1 {
			n++
		}
	}
	return n
}

// maxCompareLines keeps the visual comparison to a document that can be
// read in one sitting.
const maxCompareLines = 400

// pickCompareDoc returns the requested document, or else the document of
// readable length in which the baseline cuts the most blocks.
func pickCompareDoc(requested string, docs []pipeline.Doc, blocks map[string][]parse.Block, lineCount map[string]int, baseChunks []chunk.Chunk) (string, error) {
	if requested != "" {
		requested = filepath.ToSlash(requested)
		if _, ok := blocks[requested]; !ok {
			return "", fmt.Errorf("eval: compare document %q is not in the corpus", requested)
		}
		return requested, nil
	}
	byFile := map[string][]chunk.Chunk{}
	for _, c := range baseChunks {
		byFile[c.SourceFile] = append(byFile[c.SourceFile], c)
	}
	type scored struct {
		name  string
		score int
		kinds int
		short bool
	}
	var all []scored
	for _, d := range docs {
		s := MeasureStructure(blocks[d.Name], byFile[d.Name])
		kinds := 0
		for _, n := range []int{s.CodeCut, s.TableCut, s.ListCut} {
			if n > 0 {
				kinds++
			}
		}
		all = append(all, scored{d.Name, s.Violations() + s.OrphanHeadings, kinds, lineCount[d.Name] <= maxCompareLines})
	}
	// Prefer readable documents, then variety of damage, then amount.
	sort.SliceStable(all, func(i, j int) bool {
		a, b := all[i], all[j]
		if a.short != b.short {
			return a.short
		}
		if a.kinds != b.kinds {
			return a.kinds > b.kinds
		}
		if a.score != b.score {
			return a.score > b.score
		}
		return a.name < b.name
	})
	return all[0].name, nil
}
