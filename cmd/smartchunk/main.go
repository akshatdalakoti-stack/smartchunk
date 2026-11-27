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
