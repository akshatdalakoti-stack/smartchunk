package baseline

import (
	"fmt"
	"strings"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
)

// words returns "w0 w1 ... w(n-1)".
func words(n int) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = fmt.Sprintf("w%d", i)
	}
	return strings.Join(parts, " ")
}

func TestSplitWindows(t *testing.T) {
	tests := []struct {
		name       string
		tokens     int
		size       int
		overlap    int
		wantCounts []int
	}{
		{"empty", 0, 10, 2, nil},
		{"shorter than a window", 5, 10, 2, []int{5}},
		{"exactly one window", 10, 10, 2, []int{10}},
		{"two windows", 15, 10, 2, []int{10, 7}},
		{"no overlap", 25, 10, 0, []int{10, 10, 5}},
		{"heavy overlap", 12, 10, 9, []int{10, 10, 10}},
		{"last window is not a pure repeat", 18, 10, 2, []int{10, 10}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := Split("doc.md", []byte(words(tt.tokens)), Options{Size: tt.size, Overlap: tt.overlap})
			if err != nil {
				t.Fatalf("Split: %v", err)
			}
			if len(chunks) != len(tt.wantCounts) {
				t.Fatalf("got %d chunks, want %d", len(chunks), len(tt.wantCounts))
			}
			tok := chunk.ApproxTokenizer{}
			for i, c := range chunks {
				if c.TokenCount != tt.wantCounts[i] {
					t.Errorf("chunk %d token count = %d, want %d", i, c.TokenCount, tt.wantCounts[i])
				}
				if got := tok.Count(c.Text); got != c.TokenCount {
					t.Errorf("chunk %d: text has %d tokens but token_count says %d", i, got, c.TokenCount)
				}
			}
		})
	}
}

func TestSplitOverlapContent(t *testing.T) {
	chunks, err := Split("doc.md", []byte(words(15)), Options{Size: 10, Overlap: 3})
	if err != nil {
		t.Fatalf("Split: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2", len(chunks))
	}
	if !strings.HasSuffix(chunks[0].Text, "w7 w8 w9") {
		t.Errorf("first chunk ends with %q", chunks[0].Text)
	}
	if !strings.HasPrefix(chunks[1].Text, "w7 w8 w9 w10") {
		t.Errorf("second chunk starts with %q", chunks[1].Text)
	}
}

func TestSplitLineNumbers(t *testing.T) {
	src := "one two\nthree four\n\nfive six\nseven eight\n"
	tests := []struct {
		name      string
		size      int
		overlap   int
		wantLines [][2]int
	}{
		{"whole file", 100, 0, [][2]int{{1, 5}}},
		{"two tokens per chunk", 2, 0, [][2]int{{1, 1}, {2, 2}, {4, 4}, {5, 5}}},
		{"window straddles lines", 3, 0, [][2]int{{1, 2}, {2, 4}, {5, 5}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := Split("doc.md", []byte(src), Options{Size: tt.size, Overlap: tt.overlap})
			if err != nil {
				t.Fatalf("Split: %v", err)
			}
			if len(chunks) != len(tt.wantLines) {
				t.Fatalf("got %d chunks, want %d", len(chunks), len(tt.wantLines))
			}
			for i, c := range chunks {
				if c.StartLine != tt.wantLines[i][0] || c.EndLine != tt.wantLines[i][1] {
					t.Errorf("chunk %d lines = %d-%d, want %d-%d", i, c.StartLine, c.EndLine, tt.wantLines[i][0], tt.wantLines[i][1])
				}
			}
		})
	}
}

func TestSplitSchema(t *testing.T) {
	chunks, err := Split("dir/doc.md", []byte("# Title\r\n\r\nbody text\r\n"), DefaultOptions())
	if err != nil {
		t.Fatalf("Split: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	c := chunks[0]
	if c.ID == "" || c.SourceFile != "dir/doc.md" || c.ChunkType != chunk.Prose {
		t.Errorf("unexpected chunk metadata: %+v", c)
	}
	if c.HeadingPath == nil || c.Flags == nil {
		t.Errorf("heading_path and flags must be empty slices, not nil: %+v", c)
	}
	if strings.Contains(c.Text, "\r") {
		t.Errorf("text still contains carriage returns: %q", c.Text)
	}
}

func TestSplitInvalidOptions(t *testing.T) {
	tests := []struct {
		name string
		file string
		opt  Options
	}{
		{"zero size", "a.md", Options{Size: 0}},
		{"negative size", "a.md", Options{Size: -1}},
		{"negative overlap", "a.md", Options{Size: 10, Overlap: -1}},
		{"overlap equals size", "a.md", Options{Size: 10, Overlap: 10}},
		{"overlap above size", "a.md", Options{Size: 10, Overlap: 11}},
		{"empty file name", "", Options{Size: 10}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := Split(tt.file, []byte("some text"), tt.opt); err == nil {
				t.Error("expected an error, got nil")
			}
		})
	}
}
