package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/export"
)

const (
	fixtureInput  = "../../internal/pipeline/testdata/input"
	fixtureGolden = "../../internal/pipeline/testdata/golden"
)

// invoke runs the CLI in-process and returns its exit code and output.
func invoke(t *testing.T, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errOut bytes.Buffer
	code = run(context.Background(), args, &out, &errOut)
	return code, out.String(), errOut.String()
}

func readNormalized(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return strings.ReplaceAll(string(data), "\r\n", "\n")
}

// TestRunMatchesGolden drives the real CLI over the pipeline fixtures and
// expects byte-for-byte the golden files the pipeline tests maintain.
func TestRunMatchesGolden(t *testing.T) {
	tests := []struct {
		name   string
		args   []string
		golden string
	}{
		{
			name:   "run with flags after the path",
			args:   []string{"run", fixtureInput, "--max-tokens", "120", "--min-tokens", "30", "--embedder", "fake", "--workers", "3"},
			golden: "smart.jsonl",
		},
		{
			name:   "run with flags before the path",
			args:   []string{"run", "--max-tokens=120", "--min-tokens=30", "--embedder=fake", fixtureInput},
			golden: "smart.jsonl",
		},
		{
			name:   "run without the semantic stage",
			args:   []string{"run", fixtureInput, "--max-tokens", "120", "--min-tokens", "30", "--embedder", "none"},
			golden: "structure_only.jsonl",
		},
		{
			name:   "baseline",
			args:   []string{"baseline", fixtureInput, "--size", "120", "--overlap", "20"},
			golden: "baseline.jsonl",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "nested", "out.jsonl")
			code, _, stderr := invoke(t, append(tt.args, "--out", out)...)
			if code != exitOK {
				t.Fatalf("exit code %d, stderr: %s", code, stderr)
			}
			if got, want := readNormalized(t, out), readNormalized(t, filepath.Join(fixtureGolden, tt.golden)); got != want {
				t.Errorf("output differs from %s", tt.golden)
			}
			if !strings.Contains(stderr, "documents ->") {
				t.Errorf("no summary on stderr: %q", stderr)
			}
		})
	}
}

func TestRunSingleFileAndOptions(t *testing.T) {
	file := filepath.Join(fixtureInput, "reference.md")
	tests := []struct {
		name  string
		args  []string
		check func(t *testing.T, lines int, stderr string)
	}{
		{
			name: "single file",
			args: []string{"run", file, "--embedder", "none"},
			check: func(t *testing.T, lines int, stderr string) {
				if lines == 0 {
					t.Error("no chunks written")
				}
			},
		},
		{
			name: "dedup off keeps every chunk",
			args: []string{"run", fixtureInput, "--embedder", "none", "--max-tokens", "120", "--min-tokens", "30", "--dedup-threshold", "0"},
			check: func(t *testing.T, lines int, stderr string) {
				if !strings.Contains(stderr, "(0 near-duplicates dropped") {
					t.Errorf("dedup was not disabled: %s", stderr)
				}
			},
		},
		{
			name: "merging off leaves small chunks alone",
			args: []string{"run", fixtureInput, "--embedder", "none", "--max-tokens", "120", "--min-tokens", "0"},
			check: func(t *testing.T, lines int, stderr string) {
				// The same input gives 17 chunks with --min-tokens 30.
				if lines <= 17 {
					t.Errorf("expected more than 17 chunks with merging off, got %d", lines)
				}
				if strings.Contains(stderr, "undersized") {
					t.Errorf("unexpected output: %s", stderr)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "out.jsonl")
			code, _, stderr := invoke(t, append(tt.args, "--out", out)...)
			if code != exitOK {
				t.Fatalf("exit code %d, stderr: %s", code, stderr)
			}
			f, err := os.Open(out)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			chunks, err := export.ReadJSONL(f)
			if err != nil {
				t.Fatalf("output is not valid JSONL: %v", err)
			}
			for _, c := range chunks {
				if c.SourceFile == "" || strings.Contains(c.SourceFile, `\`) {
					t.Errorf("bad source_file %q", c.SourceFile)
				}
			}
			tt.check(t, len(chunks), stderr)
		})
	}
}

func TestEvalCommand(t *testing.T) {
	questions := filepath.Join(t.TempDir(), "q.jsonl")
	q := `{"question":"Which port do the workers use?","source_file":"guide/install.md","answer_start_line":16,"answer_end_line":20}
{"question":"What does exit code 2 mean?","source_file":"reference.md","answer_start_line":57,"answer_end_line":62}
`
	if err := os.WriteFile(questions, []byte(q), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "report")
	code, stdout, stderr := invoke(t, "eval", "--corpus", fixtureInput, "--questions", questions, "--out", out,
		"--embedder", "fake", "--max-tokens", "120", "--min-tokens", "30", "--size", "120", "--overlap", "20")
	if code != exitOK {
		t.Fatalf("exit code %d, stderr: %s", code, stderr)
	}
	for _, want := range []string{"baseline", "smartchunk", "structural violations", "hit rate @1", "MRR"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("stdout is missing %q:\n%s", want, stdout)
		}
	}
	for _, name := range []string{"results.md", "compare.html"} {
		info, err := os.Stat(filepath.Join(out, name))
		if err != nil || info.Size() == 0 {
			t.Errorf("%s was not written: %v", name, err)
		}
	}
	results := readNormalized(t, filepath.Join(out, "results.md"))
	if !strings.Contains(results, "fake (deterministic hashed bag of words)") {
		t.Error("results.md does not name the embedder")
	}
}

func TestUsageErrors(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		wantCode int
		wantMsg  string
	}{
		{"no arguments", nil, exitUsage, "Usage:"},
		{"unknown command", []string{"frobnicate"}, exitUsage, "unknown command"},
		{"run without a path", []string{"run", "--embedder", "none"}, exitUsage, "missing <path>"},
		{"run with two paths", []string{"run", "a", "b", "--embedder", "none"}, exitUsage, "expected one <path>"},
		{"unknown flag", []string{"run", fixtureInput, "--bogus"}, exitUsage, "bogus"},
		{"bad flag value", []string{"run", fixtureInput, "--max-tokens", "lots"}, exitUsage, "max-tokens"},
		{"zero max tokens", []string{"run", fixtureInput, "--max-tokens", "0", "--embedder", "none"}, exitUsage, "--max-tokens"},
		{"min above max", []string{"run", fixtureInput, "--max-tokens", "50", "--min-tokens", "60", "--embedder", "none"}, exitUsage, "--min-tokens"},
		{"bad dedup threshold", []string{"run", fixtureInput, "--dedup-threshold", "1.5", "--embedder", "none"}, exitUsage, "--dedup-threshold"},
		{"bad percentile", []string{"run", fixtureInput, "--percentile", "0", "--embedder", "none"}, exitUsage, "--percentile"},
		{"bad workers", []string{"run", fixtureInput, "--workers", "0", "--embedder", "none"}, exitUsage, "--workers"},
		{"unknown embedder", []string{"run", fixtureInput, "--embedder", "magic"}, exitUsage, "unknown --embedder"},
		{"baseline overlap too large", []string{"baseline", fixtureInput, "--size", "10", "--overlap", "10"}, exitUsage, "overlap"},
		{"baseline without a path", []string{"baseline"}, exitUsage, "missing <path>"},
		{"eval without corpus", []string{"eval", "--questions", "q.jsonl"}, exitUsage, "--corpus and --questions"},
		{"eval with a positional argument", []string{"eval", "stray", "--corpus", "c", "--questions", "q"}, exitUsage, "no positional arguments"},
		{"eval without an embedder", []string{"eval", "--corpus", "c", "--questions", "q", "--embedder", "none"}, exitUsage, "needs an embedder"},
		{"missing input path", []string{"run", filepath.Join(t.TempDir(), "nope"), "--embedder", "none"}, exitError, "nope"},
		{"directory without markdown", []string{"run", t.TempDir(), "--embedder", "none"}, exitError, "no Markdown files"},
		{"eval with a missing corpus", []string{"eval", "--corpus", filepath.Join(t.TempDir(), "nope"), "--questions", "q", "--embedder", "fake"}, exitError, "nope"},
		{"ollama unreachable", []string{"run", fixtureInput, "--ollama-url", "http://127.0.0.1:1", "--no-cache"}, exitError, "ollama serve"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, stdout, stderr := invoke(t, tt.args...)
			if code != tt.wantCode {
				t.Errorf("exit code = %d, want %d (stderr: %s)", code, tt.wantCode, stderr)
			}
			if !strings.Contains(stderr+stdout, tt.wantMsg) {
				t.Errorf("output does not mention %q:\n%s", tt.wantMsg, stderr)
			}
		})
	}
}

func TestHelp(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{"top level", []string{"help"}, "smartchunk eval"},
		{"top level flag", []string{"--help"}, "smartchunk run"},
		{"run", []string{"run", "--help"}, "-max-tokens"},
		{"baseline", []string{"baseline", "-h"}, "-overlap"},
		{"eval", []string{"eval", "--help"}, "-questions"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, stdout, stderr := invoke(t, tt.args...)
			if code != exitOK {
				t.Errorf("exit code = %d, want 0", code)
			}
			if !strings.Contains(stdout+stderr, tt.want) {
				t.Errorf("help does not mention %q:\n%s%s", tt.want, stdout, stderr)
			}
		})
	}
}

func TestCacheDirFlag(t *testing.T) {
	// With the fake embedder no cache file should ever be created.
	cacheDir := filepath.Join(t.TempDir(), "cache")
	out := filepath.Join(t.TempDir(), "out.jsonl")
	code, _, stderr := invoke(t, "run", fixtureInput, "--embedder", "fake", "--cache-dir", cacheDir, "--out", out)
	if code != exitOK {
		t.Fatalf("exit code %d: %s", code, stderr)
	}
	if _, err := os.Stat(cacheDir); !os.IsNotExist(err) {
		t.Errorf("fake embedder created a cache directory (err=%v)", err)
	}
}
