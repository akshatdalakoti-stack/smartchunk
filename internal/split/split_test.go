package split

import (
	"reflect"
	"strings"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
)

func mustParse(t *testing.T, src string) []parse.Block {
	t.Helper()
	blocks, err := parse.Parse([]byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return blocks
}

// firstLines returns the first line of every block, which is enough to tell
// candidates apart in the table tests.
func firstLines(blocks []parse.Block) []string {
	out := make([]string, len(blocks))
	for i, b := range blocks {
		out[i] = strings.SplitN(b.Text, "\n", 2)[0]
	}
	return out
}

func TestStructure(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want [][]string
	}{
		{
			name: "empty",
			src:  "",
			want: nil,
		},
		{
			name: "no headings is one candidate",
			src:  "one\n\ntwo\n\n```\ncode\n```\n",
			want: [][]string{{"one", "two", "```"}},
		},
		{
			name: "one candidate per section",
			src:  "# A\n\na text\n\n## B\n\nb text\n\nb more\n\n## C\n\nc text\n",
			want: [][]string{{"# A", "a text"}, {"## B", "b text", "b more"}, {"## C", "c text"}},
		},
		{
			name: "preamble before first heading",
			src:  "preamble\n\n# A\n\ntext\n",
			want: [][]string{{"preamble"}, {"# A", "text"}},
		},
		{
			name: "heading run attaches to content below",
			src:  "# Title\n\n## Sub\n\n### Deep\n\ntext\n\n## Next\n\nmore\n",
			want: [][]string{{"# Title", "## Sub", "### Deep", "text"}, {"## Next", "more"}},
		},
		{
			name: "trailing heading joins previous section",
			src:  "# A\n\ntext\n\n## Empty tail\n",
			want: [][]string{{"# A", "text", "## Empty tail"}},
		},
		{
			name: "document of only headings",
			src:  "# A\n\n## B\n",
			want: [][]string{{"# A", "## B"}},
		},
		{
			name: "atomic blocks stay in their section",
			src:  "# A\n\n- x\n- y\n\n| a |\n|---|\n| 1 |\n\n# B\n\n```\nc\n```\n",
			want: [][]string{{"# A", "- x", "| a |"}, {"# B", "```"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cands := Structure(mustParse(t, tt.src))
			var got [][]string
			for _, c := range cands {
				got = append(got, firstLines(c.Blocks))
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("candidates\n got: %q\nwant: %q", got, tt.want)
			}
		})
	}
}

// TestStructureInvariants checks the two promises of the package on a
// document that exercises every block type.
func TestStructureInvariants(t *testing.T) {
	src := "intro\n\n# A\n\n## B\n\ntext\n\n```\ncode\n```\n\n## C\n\n- l1\n- l2\n\n| t |\n|---|\n| 1 |\n\n> quote\n\n# D\n"
	blocks := mustParse(t, src)
	cands := Structure(blocks)

	total := 0
	for i, c := range cands {
		total += len(c.Blocks)
		if c.OnlyHeadings() {
			t.Errorf("candidate %d is a bare heading: %q", i, firstLines(c.Blocks))
		}
	}
	if total != len(blocks) {
		t.Errorf("candidates hold %d blocks, document has %d", total, len(blocks))
	}
	// Flattened candidates must reproduce the block order exactly.
	var flat []parse.Block
	for _, c := range cands {
		flat = append(flat, c.Blocks...)
	}
	if !reflect.DeepEqual(flat, blocks) {
		t.Error("candidates reordered or altered blocks")
	}
}

func TestUnits(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want [][]string
	}{
		{
			name: "prose blocks are separate units",
			src:  "one\n\ntwo\n\nthree\n",
			want: [][]string{{"one"}, {"two"}, {"three"}},
		},
		{
			name: "code follows its explanation",
			src:  "explain\n\n```\ncode\n```\n\nnext\n",
			want: [][]string{{"explain", "```"}, {"next"}},
		},
		{
			name: "heading leads the first unit",
			src:  "# H\n\nfirst\n\nsecond\n",
			want: [][]string{{"# H", "first"}, {"second"}},
		},
		{
			name: "section starting with code",
			src:  "# H\n\n```\ncode\n```\n\ntext\n",
			want: [][]string{{"# H", "```"}, {"text"}},
		},
		{
			name: "several atomic blocks share a unit",
			src:  "options:\n\n- a\n- b\n\n| t |\n|---|\n| 1 |\n\n> note\n",
			want: [][]string{{"options:", "- a", "| t |"}, {"> note"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got [][]string
			for _, u := range Units(mustParse(t, tt.src)) {
				got = append(got, firstLines(u))
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("units\n got: %q\nwant: %q", got, tt.want)
			}
		})
	}
}

func TestCandidateType(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want chunk.Type
	}{
		{"paragraphs", "one\n\ntwo\n", chunk.Prose},
		{"heading only", "# H\n", chunk.Prose},
		{"list and quote are prose", "- a\n\n> q\n", chunk.Prose},
		{"code only", "```\nx\n```\n", chunk.Code},
		{"heading plus code", "# H\n\n```\nx\n```\n", chunk.Code},
		{"table only", "| a |\n|---|\n| 1 |\n", chunk.Table},
		{"prose plus code", "text\n\n```\nx\n```\n", chunk.Mixed},
		{"code plus table", "```\nx\n```\n\n| a |\n|---|\n| 1 |\n", chunk.Mixed},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := Candidate{Blocks: mustParse(t, tt.src)}
			if got := c.Type(); got != tt.want {
				t.Errorf("Type() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCandidateHeadingPath(t *testing.T) {
	src := "# Guide\n\nintro\n\n## Install\n\ninstall text\n\n## Usage\n\nusage text\n"
	blocks := mustParse(t, src)
	tests := []struct {
		name   string
		blocks []parse.Block
		want   []string
	}{
		{"single section", blocks[2:4], []string{"Guide", "Install"}},
		{"heading only", blocks[0:1], []string{"Guide"}},
		{"sibling sections share the parent", blocks[2:6], []string{"Guide"}},
		{"parent merged with child", blocks[0:4], []string{"Guide"}},
		{"heading run then content", []parse.Block{blocks[0], blocks[2], blocks[3]}, []string{"Guide", "Install"}},
		{"no blocks", nil, []string{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := Candidate{Blocks: tt.blocks}
			if got := c.HeadingPath(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("HeadingPath() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCandidateChunk(t *testing.T) {
	src := "# T\n\nSome words here.\n\n```\ncode\n```\n"
	c := Candidate{Blocks: mustParse(t, src), Flags: []string{chunk.FlagOversized}}
	got := c.Chunk("doc.md", chunk.ApproxTokenizer{})

	if got.Text != "# T\n\nSome words here.\n\n```\ncode\n```" {
		t.Errorf("text = %q", got.Text)
	}
	if got.StartLine != 1 || got.EndLine != 7 {
		t.Errorf("lines = %d-%d, want 1-7", got.StartLine, got.EndLine)
	}
	if got.ChunkType != chunk.Mixed || got.SourceFile != "doc.md" {
		t.Errorf("metadata = %+v", got)
	}
	if got.TokenCount != chunk.ApproxTokenizer.Count(chunk.ApproxTokenizer{}, got.Text) {
		t.Errorf("token count %d does not match text", got.TokenCount)
	}
	if !reflect.DeepEqual(got.Flags, []string{chunk.FlagOversized}) {
		t.Errorf("flags = %v", got.Flags)
	}
	if got.ID == "" {
		t.Error("id is empty")
	}
}

func TestAddFlag(t *testing.T) {
	var c Candidate
	c.AddFlag("a")
	c.AddFlag("b")
	c.AddFlag("a")
	if !reflect.DeepEqual(c.Flags, []string{"a", "b"}) {
		t.Errorf("flags = %v, want [a b]", c.Flags)
	}
}
