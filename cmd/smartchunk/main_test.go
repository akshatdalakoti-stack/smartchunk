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
