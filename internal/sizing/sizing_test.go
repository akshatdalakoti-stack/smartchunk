package sizing

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/split"
)

var tok = chunk.ApproxTokenizer{}

func structure(t *testing.T, src string) []split.Candidate {
	t.Helper()
	blocks, err := parse.Parse([]byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return split.Structure(blocks)
}

// words returns n distinct words with the given prefix, as one line.
func words(prefix string, n int) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = fmt.Sprintf("%s%d", prefix, i)
	}
	return strings.Join(parts, " ")
}

// sentences returns n sentences of size words each.
func sentences(n, size int) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = fmt.Sprintf("S%d %s.", i, words("w", size-2))
	}
	return strings.Join(parts, " ")
}

func tokenCounts(cands []split.Candidate) []int {
	out := make([]int, len(cands))
	for i, c := range cands {
		out[i] = tok.Count(c.Text())
	}
	return out
}

func TestSplitSentences(t *testing.T) {
	tests := []struct {
		name string
		text string
		want []string
	}{
		{"empty", "", nil},
		{"single", "Just one sentence.", []string{"Just one sentence."}},
		{"three terminators", "First one. Second one! Third one?", []string{"First one.", "Second one!", "Third one?"}},
		{"across lines", "First line\nstill first. Second\nsentence.", []string{"First line\nstill first.", "Second\nsentence."}},
		{"abbreviation", "Use a tool, e.g. Make. Then run it.", []string{"Use a tool, e.g. Make.", "Then run it."}},
		{"abbreviation before capital", "Compare i.e. Go and Rust. Done.", []string{"Compare i.e. Go and Rust.", "Done."}},
		{"lowercase continuation", "It ends with etc. and goes on. Next.", []string{"It ends with etc. and goes on.", "Next."}},
		{"initial", "Written by J. Smith in 2020. It works.", []string{"Written by J. Smith in 2020.", "It works."}},
		{"decimal and version", "Requires Go 1.22 or later. Install it.", []string{"Requires Go 1.22 or later.", "Install it."}},
		{"inline code", "Call `a.B. C` here. Then stop.", []string{"Call `a.B. C` here.", "Then stop."}},
		{"file names", "Edit config.yaml then save. Restart.", []string{"Edit config.yaml then save.", "Restart."}},
		{"closing quote", `He said "stop." Then left.`, []string{`He said "stop."`, "Then left."}},
		{"bold", "**Important.** Read this.", []string{"**Important.**", "Read this."}},
		{"next starts with code", "Run it. `go test` passes.", []string{"Run it.", "`go test` passes."}},
		{"list style number", "1. First thing. Second thing.", []string{"1. First thing.", "Second thing."}},
		{"no terminator", "no punctuation at all", []string{"no punctuation at all"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SplitSentences(tt.text)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("SplitSentences(%q)\n got: %q\nwant: %q", tt.text, got, tt.want)
			}
		})
	}
}

// TestSentenceSpansPartition checks that the spans tile the text exactly.
func TestSentenceSpansPartition(t *testing.T) {
	texts := []string{
		"",
		"One. Two. Three.",
		"  leading and trailing  ",
		"A `code. Span` and e.g. more.\nNew line! End?",
		"Unicode café. Ünïcode next. 日本語。",
	}
	for _, text := range texts {
		var sb strings.Builder
		prev := 0
		for _, sp := range sentenceSpans(text) {
			if sp[0] != prev || sp[1] <= sp[0] {
				t.Errorf("%q: span %v does not follow offset %d", text, sp, prev)
			}
			sb.WriteString(text[sp[0]:sp[1]])
			prev = sp[1]
		}
		if sb.String() != text {
			t.Errorf("spans of %q rebuild %q", text, sb.String())
		}
	}
}
