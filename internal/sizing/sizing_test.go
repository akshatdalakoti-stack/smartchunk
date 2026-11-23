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

func TestApplySplit(t *testing.T) {
	tests := []struct {
		name string
		src  string
		opt  Options
		// wantFirst is the first line of each resulting chunk.
		wantFirst []string
		wantFlags [][]string
	}{
		{
			name:      "fits: untouched",
			src:       "# H\n\n" + words("a", 10) + "\n",
			opt:       Options{MaxTokens: 50, MinTokens: 5},
			wantFirst: []string{"# H"},
			wantFlags: [][]string{nil},
		},
		{
			name: "cut between paragraphs, balanced",
			// 2 + 4x15 = 62 tokens, limit 40: two parts of about 31.
			src:       "# H\n\n" + words("a", 15) + "\n\n" + words("b", 15) + "\n\n" + words("c", 15) + "\n\n" + words("d", 15) + "\n",
			opt:       Options{MaxTokens: 40, MinTokens: 5},
			wantFirst: []string{"# H", "c0 c1"},
			wantFlags: [][]string{nil, nil},
		},
		{
			name: "code stays with the paragraph that introduces it",
			src: "# H\n\n" + words("a", 20) + "\n\n" + words("b", 12) + "\n\n```\n" + words("code", 12) + "\n```\n\n" +
				words("c", 20) + "\n",
			opt:       Options{MaxTokens: 40, MinTokens: 5},
			wantFirst: []string{"# H", "b0 b1", "c0 c1"},
			wantFlags: [][]string{nil, nil, nil},
		},
		{
			name:      "oversized code block stays whole",
			src:       "# H\n\nintro text here\n\n```\n" + words("code", 60) + "\n```\n\nafter text here\n",
			opt:       Options{MaxTokens: 30, MinTokens: 2},
			wantFirst: []string{"# H", "```", "after text"},
			wantFlags: [][]string{nil, {chunk.FlagOversized}, nil},
		},
		{
			name:      "oversized table and list stay whole",
			src:       "| a | b |\n|---|---|\n| " + words("x", 30) + " | y |\n\n- " + words("i", 40) + "\n- second\n",
			opt:       Options{MaxTokens: 20, MinTokens: 2},
			wantFirst: []string{"| a | b |", "- i0 i1"},
			wantFlags: [][]string{{chunk.FlagOversized}, {chunk.FlagOversized}},
		},
		{
			name:      "heading stays with an oversized code block",
			src:       "# H\n\n```\n" + words("code", 60) + "\n```\n",
			opt:       Options{MaxTokens: 30, MinTokens: 2},
			wantFirst: []string{"# H"},
			wantFlags: [][]string{{chunk.FlagOversized}},
		},
		{
			name: "long paragraph is cut at sentence boundaries",
			// 6 sentences of 10 tokens, limit 25: three parts of two sentences.
			src:       "# H\n\n" + sentences(6, 10) + "\n",
			opt:       Options{MaxTokens: 25, MinTokens: 2},
			wantFirst: []string{"# H", "S2 w0", "S4 w0"},
			wantFlags: [][]string{{chunk.FlagSentenceSplit}, {chunk.FlagSentenceSplit}, {chunk.FlagSentenceSplit}},
		},
		{
			name:      "paragraph without sentences is cut between tokens",
			src:       words("x", 50) + "\n",
			opt:       Options{MaxTokens: 20, MinTokens: 2},
			wantFirst: []string{"x0 x1", "x20 x21", "x40 x41"},
			wantFlags: [][]string{{chunk.FlagSentenceSplit}, {chunk.FlagSentenceSplit}, {chunk.FlagSentenceSplit}},
		},
		{
			name:      "blockquote holding a fence is not cut",
			src:       "> " + words("q", 20) + "\n>\n> ```\n> " + words("c", 20) + "\n> ```\n",
			opt:       Options{MaxTokens: 20, MinTokens: 2},
			wantFirst: []string{"> q0 q1"},
			wantFlags: [][]string{{chunk.FlagOversized}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Apply(structure(t, tt.src), tt.opt)
			if err != nil {
				t.Fatalf("Apply: %v", err)
			}
			var first []string
			var flags [][]string
			for _, c := range got {
				first = append(first, strings.SplitN(c.Text(), "\n", 2)[0])
				flags = append(flags, c.Flags)
			}
			// wantFirst holds a prefix of each chunk's first line.
			match := len(first) == len(tt.wantFirst)
			for i := 0; match && i < len(first); i++ {
				match = strings.HasPrefix(first[i], tt.wantFirst[i])
			}
			if !match {
				t.Errorf("chunk starts\n got: %q\nwant prefixes: %q\ntokens: %v", first, tt.wantFirst, tokenCounts(got))
			}
			if !reflect.DeepEqual(flags, tt.wantFlags) {
				t.Errorf("flags\n got: %v\nwant: %v", flags, tt.wantFlags)
			}
			for i, c := range got {
				n := tok.Count(c.Text())
				oversized := false
				for _, f := range c.Flags {
					oversized = oversized || f == chunk.FlagOversized
				}
				if n > tt.opt.MaxTokens && !oversized {
					t.Errorf("chunk %d has %d tokens (limit %d) and no oversized flag", i, n, tt.opt.MaxTokens)
				}
			}
		})
	}
}

func TestApplyMerge(t *testing.T) {
	tests := []struct {
		name      string
		src       string
		opt       Options
		wantFirst []string
		wantFlags [][]string
	}{
		{
			name:      "small sections merge up to the limit",
			src:       "# A\n\n" + words("a", 5) + "\n\n# B\n\n" + words("b", 5) + "\n\n# C\n\n" + words("c", 5) + "\n",
			opt:       Options{MaxTokens: 100, MinTokens: 10},
			wantFirst: []string{"# A"},
			wantFlags: [][]string{nil},
		},
		{
			name: "child section merges into its parent, not the unrelated section before",
			src: "# One\n\n" + words("a", 20) + "\n\n# Two\n\n" + words("b", 4) + "\n\n## Two child\n\n" +
				words("c", 20) + "\n",
			opt:       Options{MaxTokens: 40, MinTokens: 10},
			wantFirst: []string{"# One", "# Two"},
			wantFlags: [][]string{nil, nil},
		},
		{
			name: "tail merges back into its own section",
			src: "# One\n\n" + words("a", 20) + "\n\n## Sub\n\n" + words("b", 3) + "\n\n# Two\n\n" +
				words("c", 20) + "\n",
			opt:       Options{MaxTokens: 40, MinTokens: 10},
			wantFirst: []string{"# One", "# Two"},
			wantFlags: [][]string{nil, nil},
		},
		{
			name:      "no room in either neighbour: flagged",
			src:       "# One\n\n" + words("a", 28) + "\n\n# Tiny\n\nfew words\n\n# Two\n\n" + words("c", 28) + "\n",
			opt:       Options{MaxTokens: 32, MinTokens: 10},
			wantFirst: []string{"# One", "# Tiny", "# Two"},
			wantFlags: [][]string{nil, {chunk.FlagUndersized}, nil},
		},
		{
			name:      "short lead-in rides with an oversized block, trailing text does not",
			src:       "# H\n\nintro text here\n\n```\n" + words("code", 60) + "\n```\n\nafter text here\n",
			opt:       Options{MaxTokens: 30, MinTokens: 10},
			wantFirst: []string{"# H", "after text here"},
			wantFlags: [][]string{{chunk.FlagOversized}, {chunk.FlagUndersized}},
		},
		{
			name:      "single small document is kept and flagged",
			src:       "tiny\n",
			opt:       Options{MaxTokens: 32, MinTokens: 10},
			wantFirst: []string{"tiny"},
			wantFlags: [][]string{{chunk.FlagUndersized}},
		},
		{
			name:      "chunks at or above the minimum are left alone",
			src:       "# A\n\n" + words("a", 10) + "\n\n# B\n\n" + words("b", 10) + "\n",
			opt:       Options{MaxTokens: 100, MinTokens: 10},
			wantFirst: []string{"# A", "# B"},
			wantFlags: [][]string{nil, nil},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Apply(structure(t, tt.src), tt.opt)
			if err != nil {
				t.Fatalf("Apply: %v", err)
			}
			var first []string
			var flags [][]string
			for _, c := range got {
				first = append(first, strings.SplitN(c.Text(), "\n", 2)[0])
				flags = append(flags, c.Flags)
			}
			if !reflect.DeepEqual(first, tt.wantFirst) {
				t.Errorf("chunk starts\n got: %q\nwant: %q\ntokens: %v", first, tt.wantFirst, tokenCounts(got))
			}
			if !reflect.DeepEqual(flags, tt.wantFlags) {
				t.Errorf("flags\n got: %v\nwant: %v", flags, tt.wantFlags)
			}
		})
	}
}

func TestMergePrefersSimilarNeighbour(t *testing.T) {
	mk := func(text string, soft bool, sim float64) split.Candidate {
		return split.Candidate{
			Blocks:  []parse.Block{{Type: parse.Paragraph, Text: text, StartLine: 1, EndLine: 1, HeadingPath: []string{"H"}}},
			Soft:    soft,
			SimPrev: sim,
			HasSim:  true,
		}
	}
	tests := []struct {
		name             string
		simPrev, simNext float64
		want             []string
	}{
		{"closer to previous", 0.9, 0.2, []string{words("a", 12) + "\n\ntiny one", words("c", 12)}},
		{"closer to next", 0.2, 0.9, []string{words("a", 12), "tiny one\n\n" + words("c", 12)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cands := []split.Candidate{
				mk(words("a", 12), false, 0),
				mk("tiny one", true, tt.simPrev),
				mk(words("c", 12), true, tt.simNext),
			}
			got, err := Apply(cands, Options{MaxTokens: 20, MinTokens: 5})
			if err != nil {
				t.Fatalf("Apply: %v", err)
			}
			var texts []string
			for _, c := range got {
				texts = append(texts, c.Text())
			}
			if !reflect.DeepEqual(texts, tt.want) {
				t.Errorf("merged\n got: %q\nwant: %q", texts, tt.want)
			}
		})
	}
}
