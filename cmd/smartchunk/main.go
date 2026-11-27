// Command smartchunk is a structure-aware chunking engine for RAG.
//
//	smartchunk run <path> --out chunks.jsonl
//	smartchunk baseline <path> --out baseline.jsonl
//	smartchunk eval --corpus <dir> --questions q.jsonl --out report/
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"time"

	"github.com/akshatdalakoti-stack/smartchunk/internal/baseline"
	"github.com/akshatdalakoti-stack/smartchunk/internal/dedup"
	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/eval"
	"github.com/akshatdalakoti-stack/smartchunk/internal/export"
	"github.com/akshatdalakoti-stack/smartchunk/internal/pipeline"
	"github.com/akshatdalakoti-stack/smartchunk/internal/sizing"
)

// Exit codes.
const (
	exitOK    = 0
	exitError = 1
	exitUsage = 2
)

const usage = `SmartChunk: structure-aware chunking for RAG.

Usage:
  smartchunk run <path> --out chunks.jsonl [flags]
      Chunk a Markdown file or directory with the full pipeline.
  smartchunk baseline <path> --out baseline.jsonl [flags]
      Chunk with the naive fixed-size splitter.
  smartchunk eval --corpus <dir> --questions q.jsonl --out report/ [flags]
      Run both over a corpus and write results.md and compare.html.

Run "smartchunk <command> --help" for the flags of a command.
`

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	os.Exit(run(ctx, os.Args[1:], os.Stdout, os.Stderr))
}

// run executes one command line and returns the process exit code. It is
// separate from main so tests can drive the whole CLI in-process.
func run(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, usage)
		return exitUsage
	}
	var err error
	switch args[0] {
	case "run":
		err = cmdRun(ctx, args[1:], stderr)
	case "baseline":
		err = cmdBaseline(ctx, args[1:], stderr)
	case "eval":
		err = cmdEval(ctx, args[1:], stdout, stderr)
	case "help", "-h", "--help":
		fmt.Fprint(stdout, usage)
		return exitOK
	default:
		fmt.Fprintf(stderr, "smartchunk: unknown command %q\n\n%s", args[0], usage)
		return exitUsage
	}
	var ue usageError
	switch {
	case err == nil:
		return exitOK
	case errors.Is(err, flag.ErrHelp):
		return exitOK
	case errors.As(err, &ue):
		fmt.Fprintf(stderr, "smartchunk: %v\n", err)
		return exitUsage
	default:
		fmt.Fprintf(stderr, "smartchunk: %v\n", err)
		return exitError
	}
}

// usageError marks a mistake in how the command was invoked.
type usageError struct{ msg string }

func (e usageError) Error() string { return e.msg }

func usagef(format string, args ...any) error {
	return usageError{fmt.Sprintf(format, args...)}
}

// parseInterleaved parses flags that may come before, after or between
// positional arguments ("run docs/ --out x.jsonl"), which the flag package
// does not do on its own. It returns the positional arguments.
func parseInterleaved(fs *flag.FlagSet, args []string) ([]string, error) {
	var positional []string
	for {
		if err := fs.Parse(args); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil, err
			}
			return nil, usageError{err.Error()}
		}
		rest := fs.Args()
		if len(rest) == 0 {
			return positional, nil
		}
		positional = append(positional, rest[0])
		args = rest[1:]
	}
}

func newFlagSet(name string, stderr io.Writer, synopsis string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintf(stderr, "Usage: smartchunk %s\n\nFlags:\n", synopsis)
		fs.PrintDefaults()
	}
	return fs
}

// sizeFlags are the chunking flags shared by run and eval.
type sizeFlags struct {
	maxTokens, minTokens int
	dedupThreshold       float64
	percentile           float64
	workers              int
}

func (s *sizeFlags) register(fs *flag.FlagSet) {
	d := sizing.DefaultOptions()
	fs.IntVar(&s.maxTokens, "max-tokens", d.MaxTokens, "largest chunk, in tokens")
	fs.IntVar(&s.minTokens, "min-tokens", d.MinTokens, "chunks smaller than this are merged into a neighbour (0 turns merging off)")
	fs.Float64Var(&s.dedupThreshold, "dedup-threshold", dedup.DefaultThreshold, "Jaccard similarity at which a chunk is dropped as a near-duplicate (0 keeps duplicates)")
	fs.Float64Var(&s.percentile, "percentile", 25, "cut sections where neighbour similarity is below this percentile of the document")
	fs.IntVar(&s.workers, "workers", runtime.NumCPU(), "documents processed at once")
}

func (s *sizeFlags) options() (pipeline.Options, error) {
	if s.maxTokens <= 0 {
		return pipeline.Options{}, usagef("--max-tokens must be positive, got %d", s.maxTokens)
	}
	if s.minTokens < 0 || s.minTokens >= s.maxTokens {
		return pipeline.Options{}, usagef("--min-tokens must be between 0 and --max-tokens (%d), got %d", s.maxTokens, s.minTokens)
	}
	if s.dedupThreshold < 0 || s.dedupThreshold > 1 {
		return pipeline.Options{}, usagef("--dedup-threshold must be between 0 and 1, got %v", s.dedupThreshold)
	}
	if s.percentile <= 0 || s.percentile > 100 {
		return pipeline.Options{}, usagef("--percentile must be in (0, 100], got %v", s.percentile)
	}
	if s.workers <= 0 {
		return pipeline.Options{}, usagef("--workers must be positive, got %d", s.workers)
	}
	opt := pipeline.Options{
		MaxTokens:      s.maxTokens,
		MinTokens:      s.minTokens,
		DedupThreshold: s.dedupThreshold,
		Percentile:     s.percentile,
		Workers:        s.workers,
	}
	// In pipeline.Options zero means "use the default"; an explicit zero on
	// the command line means "off".
	if s.minTokens == 0 {
		opt.MinTokens = -1
	}
	if s.dedupThreshold == 0 {
		opt.DedupThreshold = -1
	}
	return opt, nil
}

// embedFlags select and configure the embedder.
type embedFlags struct {
	kind     string
	url      string
	model    string
	cacheDir string
	noCache  bool
}

func (e *embedFlags) register(fs *flag.FlagSet, kinds string) {
	fs.StringVar(&e.kind, "embedder", "ollama", "embedder to use: "+kinds)
	fs.StringVar(&e.url, "ollama-url", embed.DefaultOllamaURL, "address of the Ollama server")
	fs.StringVar(&e.model, "model", embed.DefaultOllamaModel, "Ollama embedding model")
	fs.StringVar(&e.cacheDir, "cache-dir", "", "directory of the embedding cache (default: the user cache directory)")
	fs.BoolVar(&e.noCache, "no-cache", false, "do not read or write the embedding cache")
}

// embedder is a configured embedder plus what is needed to describe it and
// shut it down.
type embedder struct {
	embed.Embedder
	name  string
	stats func() (hits, misses int)
	close func() error
}

// build creates the embedder the flags describe. For Ollama it also sends
// one tiny request, so a server that is down is reported before any work
// starts instead of halfway through a corpus.
func (e *embedFlags) build(ctx context.Context) (embedder, error) {
	none := func() error { return nil }
	switch e.kind {
	case "none":
		return embedder{name: "none", close: none}, nil
	case "fake":
		return embedder{Embedder: embed.Fake{}, name: "fake (deterministic hashed bag of words)", close: none}, nil
	case "ollama":
		client := embed.NewOllama(e.url, e.model)
		if _, err := client.Embed(ctx, []string{"ping"}); err != nil {
			return embedder{}, fmt.Errorf("%w\nstart the server with `ollama serve` and `ollama pull %s`, or pass --embedder none to chunk without the semantic stage", err, e.model)
		}
		name := fmt.Sprintf("Ollama %s at %s", e.model, e.url)
		if e.noCache {
			return embedder{Embedder: client, name: name, close: none}, nil
		}
		dir := e.cacheDir
		if dir == "" {
			base, err := os.UserCacheDir()
			if err != nil {
				return embedder{}, fmt.Errorf("locating the cache directory (pass --cache-dir or --no-cache): %w", err)
			}
			dir = filepath.Join(base, "smartchunk")
		}
		cached, err := embed.NewCached(client, e.model, filepath.Join(dir, "embeddings.bin"))
		if err != nil {
			return embedder{}, err
		}
		return embedder{Embedder: cached, name: name, stats: cached.Stats, close: cached.Close}, nil
	default:
		return embedder{}, usagef("unknown --embedder %q", e.kind)
	}
}

func onePath(positional []string, what string) (string, error) {
	switch len(positional) {
	case 0:
		return "", usagef("missing <path>: give the Markdown file or directory to %s", what)
	case 1:
		return positional[0], nil
	default:
		return "", usagef("expected one <path>, got %d: %v", len(positional), positional)
	}
}

func discover(path string) ([]pipeline.Doc, error) {
	docs, err := pipeline.Discover(path)
	if err != nil {
		return nil, err
	}
	if len(docs) == 0 {
		return nil, fmt.Errorf("no Markdown files (.md, .markdown) found in %s", path)
	}
	return docs, nil
}

func cmdRun(ctx context.Context, args []string, stderr io.Writer) error {
	fs := newFlagSet("run", stderr, "run <path> --out chunks.jsonl [flags]")
	var size sizeFlags
	var emb embedFlags
	size.register(fs)
	emb.register(fs, "ollama, fake (deterministic, for tests) or none (skip the semantic stage)")
	out := fs.String("out", "chunks.jsonl", `output JSONL file ("-" for standard output)`)

	positional, err := parseInterleaved(fs, args)
	if err != nil {
		return err
	}
	path, err := onePath(positional, "chunk")
	if err != nil {
		return err
	}
	opt, err := size.options()
	if err != nil {
		return err
	}
	docs, err := discover(path)
	if err != nil {
		return err
	}
	e, err := emb.build(ctx)
	if err != nil {
		return err
	}
	defer e.close()
	opt.Embedder = e.Embedder

	start := time.Now()
	chunks, stats, err := pipeline.Run(ctx, docs, opt)
	if err != nil {
		return err
	}
	if err := export.WriteFile(*out, chunks); err != nil {
		return err
	}
	if err := e.close(); err != nil {
		return err
	}
	fmt.Fprintf(stderr, "smartchunk: %d documents -> %d chunks in %s (%d near-duplicates dropped, %d repeated lines stripped, %d oversized) -> %s\n",
		stats.Docs, stats.Chunks, time.Since(start).Round(time.Millisecond), stats.DuplicatesDrop, stats.LinesStripped, stats.Oversized, *out)
	return nil
}

func cmdBaseline(ctx context.Context, args []string, stderr io.Writer) error {
	fs := newFlagSet("baseline", stderr, "baseline <path> --out baseline.jsonl [flags]")
	d := baseline.DefaultOptions()
	size := fs.Int("size", d.Size, "window size, in tokens")
	overlap := fs.Int("overlap", d.Overlap, "tokens shared by consecutive windows")
	workers := fs.Int("workers", runtime.NumCPU(), "documents processed at once")
	out := fs.String("out", "baseline.jsonl", `output JSONL file ("-" for standard output)`)

	positional, err := parseInterleaved(fs, args)
	if err != nil {
		return err
	}
	path, err := onePath(positional, "split")
	if err != nil {
		return err
	}
	opt := baseline.Options{Size: *size, Overlap: *overlap}
	if err := opt.Validate(); err != nil {
		return usageError{err.Error()}
	}
	if *workers <= 0 {
		return usagef("--workers must be positive, got %d", *workers)
	}
	docs, err := discover(path)
	if err != nil {
		return err
	}

	start := time.Now()
	chunks, err := pipeline.RunBaseline(ctx, docs, opt, *workers)
	if err != nil {
		return err
	}
	if err := export.WriteFile(*out, chunks); err != nil {
		return err
	}
	fmt.Fprintf(stderr, "smartchunk: %d documents -> %d chunks in %s -> %s\n",
		len(docs), len(chunks), time.Since(start).Round(time.Millisecond), *out)
	return nil
}

func cmdEval(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := newFlagSet("eval", stderr, "eval --corpus <dir> --questions q.jsonl --out report/ [flags]")
	var size sizeFlags
	var emb embedFlags
	size.register(fs)
	emb.register(fs, "ollama or fake (deterministic, for tests)")
	d := baseline.DefaultOptions()
	corpus := fs.String("corpus", "", "directory of Markdown documents (required)")
	questions := fs.String("questions", "", "JSONL questions file (required)")
	out := fs.String("out", "report", "directory for results.md and compare.html")
	baseSize := fs.Int("size", d.Size, "baseline window size, in tokens")
	baseOverlap := fs.Int("overlap", d.Overlap, "baseline overlap, in tokens")
	compareDoc := fs.String("compare-doc", "", "document shown in compare.html (default: where the baseline does most damage)")
	noHeading := fs.Bool("no-heading-context", false, "embed bare chunk text, without the heading breadcrumb")
	noSemantic := fs.Bool("no-semantic", false, "run SmartChunk without its semantic stage")

	positional, err := parseInterleaved(fs, args)
	if err != nil {
		return err
	}
	if len(positional) > 0 {
		return usagef("eval takes no positional arguments, got %v", positional)
	}
	if *corpus == "" || *questions == "" {
		return usagef("eval needs --corpus and --questions")
	}
	if emb.kind == "none" {
		return usagef("eval needs an embedder for the retrieval test; use --no-semantic to switch off only the semantic stage")
	}
	opt, err := size.options()
	if err != nil {
		return err
	}
	baseOpt := baseline.Options{Size: *baseSize, Overlap: *baseOverlap}
	if err := baseOpt.Validate(); err != nil {
		return usageError{err.Error()}
	}

	e, err := emb.build(ctx)
	if err != nil {
		return err
	}
	defer e.close()
	if !*noSemantic {
		opt.Embedder = e.Embedder
	}

	dupThreshold := size.dedupThreshold
	if dupThreshold == 0 {
		dupThreshold = dedup.DefaultThreshold
	}
	report, err := eval.Run(ctx, eval.Config{
		CorpusDir:          *corpus,
		QuestionsPath:      *questions,
		OutDir:             *out,
		Smart:              opt,
		Baseline:           baseOpt,
		Embedder:           e.Embedder,
		EmbedderName:       e.name,
		EmbedStats:         e.stats,
		DuplicateThreshold: dupThreshold,
		NoHeadingContext:   *noHeading,
		CompareDoc:         *compareDoc,
		Workers:            size.workers,
	})
	if err != nil {
		return err
	}
	if err := e.close(); err != nil {
		return err
	}

	b, s := report.Baseline, report.Smart
	fmt.Fprintf(stdout, "%-34s %12s %12s\n", "", "baseline", "smartchunk")
	line := func(name, bv, sv string) { fmt.Fprintf(stdout, "%-34s %12s %12s\n", name, bv, sv) }
	line("chunks", fmt.Sprint(len(b.Chunks)), fmt.Sprint(len(s.Chunks)))
	line("structural violations", fmt.Sprint(b.Structure.Violations()), fmt.Sprint(s.Structure.Violations()))
	line("orphan headings", fmt.Sprint(b.Structure.OrphanHeadings), fmt.Sprint(s.Structure.OrphanHeadings))
	line("near-duplicate rate", pct(b.DupRate), pct(s.DupRate))
	line("hit rate @1", pct(b.Retrieval.Hit1), pct(s.Retrieval.Hit1))
	line("hit rate @3", pct(b.Retrieval.Hit3), pct(s.Retrieval.Hit3))
	line("hit rate @5", pct(b.Retrieval.Hit5), pct(s.Retrieval.Hit5))
	line("MRR", fmt.Sprintf("%.3f", b.Retrieval.MRR), fmt.Sprintf("%.3f", s.Retrieval.MRR))
	line("whole answer in a top-5 chunk", pct(b.Retrieval.Complete5), pct(s.Retrieval.Complete5))
	line("tokens mean / p95 / max",
		fmt.Sprintf("%.0f/%d/%d", b.Tokens.Mean, b.Tokens.P95, b.Tokens.Max),
		fmt.Sprintf("%.0f/%d/%d", s.Tokens.Mean, s.Tokens.P95, s.Tokens.Max))
	line("docs/sec", fmt.Sprintf("%.1f", b.DocsPerSec), fmt.Sprintf("%.1f", s.DocsPerSec))
	fmt.Fprintf(stderr, "smartchunk: wrote %s and %s\n", filepath.Join(*out, "results.md"), filepath.Join(*out, "compare.html"))
	return nil
}

func pct(x float64) string { return fmt.Sprintf("%.1f%%", 100*x) }
