package export

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
)

func sample() []chunk.Chunk {
	return []chunk.Chunk{
		{
			ID: "aaaa", Text: "# Title\n\nSome <b>text</b> & more.", ChunkType: chunk.Prose,
			HeadingPath: []string{"Title"}, SourceFile: "docs/a.md",
			StartLine: 1, EndLine: 3, TokenCount: 9, Flags: []string{},
		},
		{
			ID: "bbbb", Text: "```go\nx := \"quoted\"\n```", ChunkType: chunk.Code,
			HeadingPath: []string{"Title", "Usage"}, SourceFile: "docs/a.md",
			StartLine: 5, EndLine: 7, TokenCount: 11, Flags: []string{chunk.FlagOversized},
		},
	}
}

func TestWriteJSONLSchema(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteJSONL(&buf, sample()); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2", len(lines))
	}
	wantKeys := []string{"chunk_type", "end_line", "flags", "heading_path", "id", "source_file", "start_line", "text", "token_count"}
	for i, line := range lines {
		var obj map[string]any
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			t.Fatalf("line %d is not JSON: %v", i, err)
		}
		if len(obj) != len(wantKeys) {
			t.Errorf("line %d has %d fields, want %d", i, len(obj), len(wantKeys))
		}
		for _, k := range wantKeys {
			if _, ok := obj[k]; !ok {
				t.Errorf("line %d is missing field %q", i, k)
			}
		}
	}
	// encoding/json would by default write "<" as a backslash-u escape.
	if !strings.Contains(buf.String(), "Some <b>text</b> & more.") {
		t.Errorf("HTML characters were escaped: %s", buf.String())
	}
}

func TestWriteJSONLNilSlices(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteJSONL(&buf, []chunk.Chunk{{ID: "x", Text: "t"}}); err != nil {
		t.Fatalf("WriteJSONL: %v", err)
	}
	for _, want := range []string{`"heading_path":[]`, `"flags":[]`} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("output %s does not contain %s", buf.String(), want)
		}
	}
}

func TestRoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		chunks []chunk.Chunk
	}{
		{"no chunks", nil},
		{"sample", sample()},
		{"unicode and control characters", []chunk.Chunk{{
			ID: "u", Text: "tab\there \"quotes\" \\ backslash\nünïcode 日本語  ",
			ChunkType: chunk.Mixed, HeadingPath: []string{}, Flags: []string{},
		}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := WriteJSONL(&buf, tt.chunks); err != nil {
				t.Fatalf("WriteJSONL: %v", err)
			}
			if n := strings.Count(buf.String(), "\n"); n != len(tt.chunks) {
				t.Errorf("output has %d lines for %d chunks", n, len(tt.chunks))
			}
			got, err := ReadJSONL(&buf)
			if err != nil {
				t.Fatalf("ReadJSONL: %v", err)
			}
			if len(got) != len(tt.chunks) || (len(got) > 0 && !reflect.DeepEqual(got, tt.chunks)) {
				t.Errorf("round trip changed the chunks\n got: %+v\nwant: %+v", got, tt.chunks)
			}
		})
	}
}

func TestReadJSONLErrors(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    int
		wantErr string
	}{
		{"blank lines are skipped", "\n{\"id\":\"a\"}\n\n{\"id\":\"b\"}\n", 2, ""},
		{"bad json names the line", "{\"id\":\"a\"}\n{not json}\n", 0, "line 2"},
		{"wrong type", "{\"start_line\":\"one\"}\n", 0, "line 1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ReadJSONL(strings.NewReader(tt.input))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("err = %v, want it to mention %q", err, tt.wantErr)
				}
				return
			}
			if err != nil || len(got) != tt.want {
				t.Errorf("got %d chunks, err %v; want %d", len(got), err, tt.want)
			}
		})
	}
}

type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) { return 0, errors.New("disk full") }

func TestWriteErrors(t *testing.T) {
	if err := WriteJSONL(failingWriter{}, sample()); err == nil || !strings.Contains(err.Error(), "disk full") {
		t.Errorf("err = %v, want the writer's error", err)
	}
	if err := WriteFile("", sample()); err == nil {
		t.Error("empty path should be rejected")
	}
}

func TestWriteFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "dir", "out.jsonl")
	if err := WriteFile(path, sample()); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	got, err := ReadJSONL(f)
	if err != nil || !reflect.DeepEqual(got, sample()) {
		t.Errorf("file round trip failed: %v", err)
	}
}

func TestEmbeddingText(t *testing.T) {
	tests := []struct {
		name string
		c    chunk.Chunk
		want string
	}{
		{
			name: "no heading path",
			c:    chunk.Chunk{Text: "plain text"},
			want: "plain text",
		},
		{
			name: "mid-section chunk gets the full breadcrumb",
			c:    chunk.Chunk{Text: "continued text", HeadingPath: []string{"Guide", "Install"}},
			want: "Guide > Install\n\ncontinued text",
		},
		{
			name: "own heading is not repeated",
			c:    chunk.Chunk{Text: "## Install\n\nrun it", HeadingPath: []string{"Guide", "Install"}},
			want: "Guide\n\n## Install\n\nrun it",
		},
		{
			name: "top-level section needs no breadcrumb",
			c:    chunk.Chunk{Text: "# Guide\n\nintro", HeadingPath: []string{"Guide"}},
			want: "# Guide\n\nintro",
		},
		{
			name: "closing hashes and spacing",
			c:    chunk.Chunk{Text: "##  Install  ##\n\nrun it", HeadingPath: []string{"Guide", "Install"}},
			want: "Guide\n\n##  Install  ##\n\nrun it",
		},
		{
			name: "different heading keeps the full breadcrumb",
			c:    chunk.Chunk{Text: "### Linux\n\napt", HeadingPath: []string{"Guide", "Install"}},
			want: "Guide > Install\n\n### Linux\n\napt",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := EmbeddingText(tt.c); got != tt.want {
				t.Errorf("EmbeddingText = %q, want %q", got, tt.want)
			}
		})
	}
}
