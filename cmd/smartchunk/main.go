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
