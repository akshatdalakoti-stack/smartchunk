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

// TestRunGuarantees checks the promises of the pipeline against the parsed
// source, independently of the golden files.
func TestRunGuarantees(t *testing.T) {
	docs, err := Discover(inputDir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	opt := goldenOptions()
	chunks, _, err := Run(context.Background(), docs, opt)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	byFile := map[string][]chunk.Chunk{}
	ids := map[string]bool{}
	for _, c := range chunks {
		byFile[c.SourceFile] = append(byFile[c.SourceFile], c)
		if ids[c.ID] {
			t.Errorf("duplicate chunk id %s", c.ID)
		}
		ids[c.ID] = true
		if c.TokenCount != (chunk.ApproxTokenizer{}).Count(c.Text) {
			t.Errorf("chunk %s: token_count %d does not match its text", c.ID, c.TokenCount)
		}
		if c.TokenCount > opt.MaxTokens && !c.HasFlag(chunk.FlagOversized) {
			t.Errorf("chunk %s has %d tokens and no oversized flag", c.ID, c.TokenCount)
		}
		if c.StartLine < 1 || c.EndLine < c.StartLine {
			t.Errorf("chunk %s has line range %d-%d", c.ID, c.StartLine, c.EndLine)
		}
	}
	for _, d := range docs {
		src, err := d.Read()
		if err != nil {
			t.Fatal(err)
		}
		blocks, err := parse.Parse(src)
		if err != nil {
			t.Fatal(err)
		}
		for _, b := range blocks {
			if !b.Type.Atomic() {
				continue
			}
			for _, c := range byFile[d.Name] {
				overlaps := c.StartLine <= b.EndLine && c.EndLine >= b.StartLine
				contains := c.StartLine <= b.StartLine && c.EndLine >= b.EndLine
				if overlaps && !contains {
					t.Errorf("%s: chunk %d-%d cuts the %s block at %d-%d", d.Name, c.StartLine, c.EndLine, b.Type, b.StartLine, b.EndLine)
				}
				if contains && !strings.Contains(c.Text, b.Text) {
					t.Errorf("%s: %s block at %d-%d is not verbatim in its chunk", d.Name, b.Type, b.StartLine, b.EndLine)
				}
			}
		}
	}
}

func TestRunDeterministicAcrossWorkers(t *testing.T) {
	// Enough documents that the workers genuinely interleave.
	var docs []Doc
	for i := 0; i < 40; i++ {
		docs = append(docs, Doc{Name: fmt.Sprintf("doc%02d.md", i), Source: syntheticDoc(i, 6)})
	}
	var want []chunk.Chunk
	for _, workers := range []int{1, 2, 8, 32} {
		opt := Options{MaxTokens: 80, MinTokens: 20, Workers: workers, Embedder: embed.Fake{}}
		got, _, err := Run(context.Background(), docs, opt)
		if err != nil {
			t.Fatalf("Run with %d workers: %v", workers, err)
		}
		if want == nil {
			want = got
			continue
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("output with %d workers differs from output with 1 worker", workers)
		}
	}
	for i := 1; i < len(want); i++ {
		if want[i].SourceFile < want[i-1].SourceFile {
			t.Fatalf("chunks are not in document order: %s after %s", want[i].SourceFile, want[i-1].SourceFile)
		}
	}
}

func TestRunOptions(t *testing.T) {
	doc := []Doc{
		{Name: "a.md", Source: []byte("# A\n\n" + strings.Repeat("Same sentence here every time. ", 12) + "\n")},
		{Name: "b.md", Source: []byte("# A\n\n" + strings.Repeat("Same sentence here every time. ", 12) + "\n")},
	}
	tests := []struct {
		name       string
		opt        Options
		wantChunks int
		wantErr    bool
	}{
		{"near-duplicate document is dropped", Options{MaxTokens: 200, MinTokens: 10}, 1, false},
		{"dedup off keeps both", Options{MaxTokens: 200, MinTokens: 10, DedupThreshold: -1}, 2, false},
		{"defaults apply", Options{}, 1, false},
		{"negative minimum turns merging off", Options{MaxTokens: 200, MinTokens: -1, DedupThreshold: -1}, 2, false},
		{"min above max is rejected", Options{MaxTokens: 10, MinTokens: 20}, 0, true},
		{"bad dedup threshold is rejected", Options{DedupThreshold: 1.5}, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, stats, err := Run(context.Background(), doc, tt.opt)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			if err == nil && (len(chunks) != tt.wantChunks || stats.Chunks != tt.wantChunks || stats.Docs != 2) {
				t.Errorf("got %d chunks (stats %+v), want %d", len(chunks), stats, tt.wantChunks)
			}
		})
	}
}

type failingEmbedder struct{ err error }

func (f failingEmbedder) Embed(context.Context, []string) ([][]float32, error) { return nil, f.err }

func TestRunErrors(t *testing.T) {
	boom := errors.New("embedding backend down")
	docs := []Doc{{Name: "ok.md", Source: syntheticDoc(1, 4)}}

	t.Run("embedder failure names the document", func(t *testing.T) {
		_, _, err := Run(context.Background(), docs, Options{Embedder: failingEmbedder{boom}})
		if !errors.Is(err, boom) || !strings.Contains(err.Error(), "ok.md") {
			t.Errorf("err = %v", err)
		}
	})
	t.Run("missing file", func(t *testing.T) {
		missing := []Doc{{Name: "gone.md", Path: filepath.Join(t.TempDir(), "gone.md")}}
		_, _, err := Run(context.Background(), missing, Options{})
		if err == nil || !strings.Contains(err.Error(), "gone.md") {
			t.Errorf("err = %v", err)
		}
		if _, err := RunBaseline(context.Background(), missing, baseline.DefaultOptions(), 2); err == nil {
			t.Error("baseline should fail on a missing file too")
		}
	})
	t.Run("cancelled context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, _, err := Run(ctx, docs, Options{Embedder: embed.Fake{}})
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})
	t.Run("invalid baseline options", func(t *testing.T) {
		if _, err := RunBaseline(context.Background(), docs, baseline.Options{Size: 0}, 1); err == nil {
			t.Error("expected an error")
		}
	})
}

func TestDiscover(t *testing.T) {
	root := t.TempDir()
	files := []string{
		"b.md", "a.md", "notes.txt", "sub/c.markdown", "sub/deep/d.MD",
		".git/hidden.md", "node_modules/pkg/readme.md", ".hidden/e.md",
	}
	for _, f := range files {
		path := filepath.Join(root, filepath.FromSlash(f))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("# x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	tests := []struct {
		name    string
		root    string
		want    []string
		wantErr bool
	}{
		{"directory is walked, sorted and filtered", root, []string{"a.md", "b.md", "sub/c.markdown", "sub/deep/d.MD"}, false},
		{"single file", filepath.Join(root, "sub", "c.markdown"), []string{"c.markdown"}, false},
		{"missing path", filepath.Join(root, "nope"), nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			docs, err := Discover(tt.root)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			var names []string
			for _, d := range docs {
				names = append(names, d.Name)
			}
			if !reflect.DeepEqual(names, tt.want) {
				t.Errorf("names = %v, want %v", names, tt.want)
			}
		})
	}
}

func TestForEach(t *testing.T) {
	ctx := context.Background()

	t.Run("visits every index once", func(t *testing.T) {
		for _, workers := range []int{0, 1, 3, 100} {
			seen := make([]int32, 50)
			err := ForEach(ctx, len(seen), workers, func(_ context.Context, i int) error {
				atomic.AddInt32(&seen[i], 1)
				return nil
			})
			if err != nil {
				t.Fatalf("workers=%d: %v", workers, err)
			}
			for i, n := range seen {
				if n != 1 {
					t.Errorf("workers=%d: index %d visited %d times", workers, i, n)
				}
			}
		}
	})
	t.Run("zero items", func(t *testing.T) {
		if err := ForEach(ctx, 0, 4, func(context.Context, int) error { return errors.New("never called") }); err != nil {
			t.Errorf("err = %v", err)
		}
	})
	t.Run("first error stops the rest", func(t *testing.T) {
		boom := errors.New("boom")
		var calls int32
		err := ForEach(ctx, 1000, 2, func(_ context.Context, i int) error {
			atomic.AddInt32(&calls, 1)
			if i == 3 {
				return boom
			}
			return nil
		})
		if !errors.Is(err, boom) {
			t.Errorf("err = %v, want boom", err)
		}
		if n := atomic.LoadInt32(&calls); n > 100 {
			t.Errorf("%d calls ran after the failure; the pool did not stop", n)
		}
	})
	t.Run("never exceeds the worker count", func(t *testing.T) {
		var mu sync.Mutex
		active, peak := 0, 0
		err := ForEach(ctx, 60, 4, func(context.Context, int) error {
			mu.Lock()
			active++
			if active > peak {
				peak = active
			}
			mu.Unlock()
			for i := 0; i < 1000; i++ {
				_ = fmt.Sprint(i)
			}
			mu.Lock()
			active--
			mu.Unlock()
			return nil
		})
		if err != nil || peak > 4 {
			t.Errorf("peak concurrency %d (limit 4), err %v", peak, err)
		}
	})
	t.Run("parent cancellation is reported", func(t *testing.T) {
		cctx, cancel := context.WithCancel(ctx)
		err := ForEach(cctx, 100, 2, func(_ context.Context, i int) error {
			if i == 0 {
				cancel()
			}
			return nil
		})
		if !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})
}
