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
