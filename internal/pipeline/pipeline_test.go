package pipeline

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/baseline"
	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/export"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
)

var update = flag.Bool("update", false, "rewrite the golden files with the current output")

const inputDir = "testdata/input"

// goldenOptions are small on purpose: the test documents are short, and a
// tight limit makes them exercise sentence splitting, merging and the
// oversized flag.
func goldenOptions() Options {
	return Options{MaxTokens: 120, MinTokens: 30, Workers: 4, Embedder: embed.Fake{}}
}

// checkGolden compares chunks with a golden JSONL file, or rewrites the file
// when the test is run with -update.
func checkGolden(t *testing.T, name string, chunks []chunk.Chunk) {
	t.Helper()
	var buf bytes.Buffer
	if err := export.WriteJSONL(&buf, chunks); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	path := filepath.Join("testdata", "golden", name)
	if *update {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("updated %s (%d chunks)", path, len(chunks))
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading golden file (run `go test ./internal/pipeline -update` to create it): %v", err)
	}
	// Git may check golden files out with CRLF on Windows.
	want = bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n"))
	if bytes.Equal(buf.Bytes(), want) {
		return
	}
	gotLines := strings.Split(buf.String(), "\n")
	wantLines := strings.Split(string(want), "\n")
	for i := 0; i < len(gotLines) || i < len(wantLines); i++ {
		var g, w string
		if i < len(gotLines) {
			g = gotLines[i]
		}
		if i < len(wantLines) {
			w = wantLines[i]
		}
		if g != w {
			t.Fatalf("%s differs at line %d (%d lines got, %d want)\n got: %s\nwant: %s\nrun with -update if the change is intended",
				name, i+1, len(gotLines), len(wantLines), g, w)
		}
	}
}

func TestGoldenSmart(t *testing.T) {
	docs, err := Discover(inputDir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	chunks, stats, err := Run(context.Background(), docs, goldenOptions())
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	checkGolden(t, "smart.jsonl", chunks)

	// The documents are written to trigger every stage. If one of these
	// stops firing, the golden file is no longer testing that stage.
	if stats.LinesStripped == 0 {
		t.Error("no repeated lines were stripped")
	}
	if stats.DuplicatesDrop == 0 {
		t.Error("no near-duplicate chunk was dropped")
	}
	if stats.Oversized == 0 {
		t.Error("no chunk was flagged oversized")
	}
	seen := map[string]bool{}
	for _, c := range chunks {
		for _, f := range c.Flags {
			seen[f] = true
		}
		seen["type:"+string(c.ChunkType)] = true
		if strings.Contains(c.Text, "Back to the [documentation index]") {
			t.Errorf("chunk %s still contains the repeated footer", c.ID)
		}
		if strings.Contains(c.Text, "sidebar: guide") {
			t.Errorf("chunk %s contains front matter", c.ID)
		}
	}
	for _, want := range []string{
		chunk.FlagOversized, chunk.FlagSentenceSplit, chunk.FlagBoilerplateStripped,
		"type:prose", "type:code", "type:table", "type:mixed",
	} {
		if !seen[want] {
			t.Errorf("golden output never exercises %q", want)
		}
	}
}

func TestGoldenStructureOnly(t *testing.T) {
	docs, err := Discover(inputDir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	opt := goldenOptions()
	opt.Embedder = nil
	chunks, _, err := Run(context.Background(), docs, opt)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	checkGolden(t, "structure_only.jsonl", chunks)
}

func TestGoldenBaseline(t *testing.T) {
	docs, err := Discover(inputDir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	chunks, err := RunBaseline(context.Background(), docs, baseline.Options{Size: 120, Overlap: 20}, 4)
	if err != nil {
		t.Fatalf("RunBaseline: %v", err)
	}
	checkGolden(t, "baseline.jsonl", chunks)
}
