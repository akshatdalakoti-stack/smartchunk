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
