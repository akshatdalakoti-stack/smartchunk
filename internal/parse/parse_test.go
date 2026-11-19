package parse

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// brief is the part of a Block the table tests compare.
type brief struct {
	Type       BlockType
	Start, End int
}

func briefs(blocks []Block) []brief {
	out := make([]brief, len(blocks))
	for i, b := range blocks {
		out[i] = brief{b.Type, b.StartLine, b.EndLine}
	}
	return out
}

func TestParseBlocks(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want []brief
	}{
		{
			name: "empty document",
			src:  "",
			want: []brief{},
		},
		{
			name: "heading and paragraph",
			src:  "# Title\n\nSome text\nover two lines.\n",
			want: []brief{{Heading, 1, 1}, {Paragraph, 3, 4}},
		},
		{
			name: "fenced code keeps both fences",
			src:  "Intro\n\n```go\nfmt.Println(1)\n\nfmt.Println(2)\n```\n\nAfter\n",
			want: []brief{{Paragraph, 1, 1}, {Code, 3, 7}, {Paragraph, 9, 9}},
		},
		{
			name: "empty fenced code",
			src:  "```\n```\n\ntext\n",
			want: []brief{{Code, 1, 2}, {Paragraph, 4, 4}},
		},
		{
			name: "unclosed fence runs to end of file",
			src:  "para\n\n```sh\necho hi\necho bye\n",
			want: []brief{{Paragraph, 1, 1}, {Code, 3, 5}},
		},
		{
			name: "tilde fence",
			src:  "~~~\ncode\n~~~\n",
			want: []brief{{Code, 1, 3}},
		},
		{
			name: "indented code block",
			src:  "para\n\n    indented one\n    indented two\n\nnext\n",
			want: []brief{{Paragraph, 1, 1}, {Code, 3, 4}, {Paragraph, 6, 6}},
		},
		{
			name: "table with body",
			src:  "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\ntext\n",
			want: []brief{{Table, 1, 4}, {Paragraph, 6, 6}},
		},
		{
			name: "table with header only",
			src:  "| a | b |\n|---|---|\n\ntext\n",
			want: []brief{{Table, 1, 2}, {Paragraph, 4, 4}},
		},
		{
			name: "table directly under a paragraph line",
			src:  "intro\n\n**Caption**\n| a | b |\n|---|---|\n| 1 | 2 |\n\nafter\n",
			want: []brief{{Paragraph, 1, 1}, {Paragraph, 3, 3}, {Table, 4, 6}, {Paragraph, 8, 8}},
		},
		{
			name: "table under a two-line paragraph",
			src:  "line one\nline two\n| a | b |\n|---|---|\n| 1 | 2 |\n",
			want: []brief{{Paragraph, 1, 2}, {Table, 3, 5}},
		},
		{
			name: "nested list is one block",
			src:  "- one\n- two\n  - nested\n  - nested two\n- three\n\ntext\n",
			want: []brief{{List, 1, 5}, {Paragraph, 7, 7}},
		},
		{
			name: "loose list with code inside",
			src:  "1. first\n\n   ```sh\n   run\n   ```\n\n2. second\n\ndone\n",
			want: []brief{{List, 1, 7}, {Paragraph, 9, 9}},
		},
		{
			name: "list ending in a fenced block",
			src:  "- item\n\n  ```\n  code\n  ```\n\nafter\n",
			want: []brief{{List, 1, 5}, {Paragraph, 7, 7}},
		},
		{
			name: "blockquote",
			src:  "> quoted\n> more\n\ntext\n",
			want: []brief{{Blockquote, 1, 2}, {Paragraph, 4, 4}},
		},
		{
			name: "setext heading includes underline",
			src:  "Title\n=====\n\nbody\n",
			want: []brief{{Heading, 1, 2}, {Paragraph, 4, 4}},
		},
		{
			name: "thematic break is skipped",
			src:  "one\n\n---\n\ntwo\n",
			want: []brief{{Paragraph, 1, 1}, {Paragraph, 5, 5}},
		},
		{
			name: "html block is prose",
			src:  "<div>\nhello\n</div>\n\ntext\n",
			want: []brief{{Paragraph, 1, 3}, {Paragraph, 5, 5}},
		},
		{
			name: "front matter is ignored",
			src:  "---\ntitle: Doc\nid: x\n---\n\n# Real\n\nbody\n",
			want: []brief{{Heading, 6, 6}, {Paragraph, 8, 8}},
		},
		{
			name: "crlf line endings",
			src:  "# T\r\n\r\npara\r\n\r\n```\r\nx\r\n```\r\n",
			want: []brief{{Heading, 1, 1}, {Paragraph, 3, 3}, {Code, 5, 7}},
		},
		{
			name: "link reference definitions are dropped",
			src:  "See [docs].\n\n[docs]: https://example.com\n",
			want: []brief{{Paragraph, 1, 1}},
		},
		{
			name: "no trailing newline",
			src:  "# T\n\nlast line",
			want: []brief{{Heading, 1, 1}, {Paragraph, 3, 3}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blocks, err := Parse([]byte(tt.src))
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			got := briefs(blocks)
			if len(got) == 0 && len(tt.want) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("blocks\n got: %v\nwant: %v", got, tt.want)
			}
		})
	}
}

func TestParseRawText(t *testing.T) {
	src := "# Title\n\nIntro text.\n\n```go\nx := 1\n```\n\n- a\n- b\n"
	blocks, err := Parse([]byte(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	want := []string{"# Title", "Intro text.", "```go\nx := 1\n```", "- a\n- b"}
	if len(blocks) != len(want) {
		t.Fatalf("got %d blocks, want %d", len(blocks), len(want))
	}
	for i, b := range blocks {
		if b.Text != want[i] {
			t.Errorf("block %d text = %q, want %q", i, b.Text, want[i])
		}
	}
}

// TestParseTextMatchesLines checks the invariant every later stage relies
// on: a block's Text is exactly the source lines StartLine..EndLine.
func TestParseTextMatchesLines(t *testing.T) {
	src := "# A\n\npara one\nstill one\n\n## B\n\n| x | y |\n|---|---|\n| 1 | 2 |\n\n> quote\n\n1. a\n2. b\n\n```\ncode\n```\n"
	lines := strings.Split(src, "\n")
	blocks, err := Parse([]byte(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	prevEnd := 0
	for i, b := range blocks {
		if b.StartLine <= prevEnd {
			t.Errorf("block %d starts at %d, overlapping previous end %d", i, b.StartLine, prevEnd)
		}
		want := strings.Join(lines[b.StartLine-1:b.EndLine], "\n")
		if b.Text != want {
			t.Errorf("block %d text = %q, want lines %d-%d = %q", i, b.Text, b.StartLine, b.EndLine, want)
		}
		prevEnd = b.EndLine
	}
}

func TestParseHeadingPath(t *testing.T) {
	src := strings.Join([]string{
		"intro before any heading",
		"",
		"# Guide",
		"",
		"guide text",
		"",
		"## Install",
		"",
		"install text",
		"",
		"### Linux `apt`",
		"",
		"linux text",
		"",
		"## Usage",
		"",
		"usage text",
		"",
		"# Reference",
		"",
		"ref text",
	}, "\n")
	blocks, err := Parse([]byte(src))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	tests := []struct {
		text  string
		path  []string
		level int
	}{
		{"intro before any heading", []string{}, 0},
		{"# Guide", []string{"Guide"}, 1},
		{"guide text", []string{"Guide"}, 0},
		{"## Install", []string{"Guide", "Install"}, 2},
		{"install text", []string{"Guide", "Install"}, 0},
		{"### Linux `apt`", []string{"Guide", "Install", "Linux apt"}, 3},
		{"linux text", []string{"Guide", "Install", "Linux apt"}, 0},
		{"## Usage", []string{"Guide", "Usage"}, 2},
		{"usage text", []string{"Guide", "Usage"}, 0},
		{"# Reference", []string{"Reference"}, 1},
		{"ref text", []string{"Reference"}, 0},
	}
	if len(blocks) != len(tests) {
		t.Fatalf("got %d blocks, want %d", len(blocks), len(tests))
	}
	for i, tt := range tests {
		b := blocks[i]
		if b.Text != tt.text {
			t.Errorf("block %d text = %q, want %q", i, b.Text, tt.text)
		}
		if !reflect.DeepEqual(b.HeadingPath, tt.path) {
			t.Errorf("block %d (%q) path = %v, want %v", i, tt.text, b.HeadingPath, tt.path)
		}
		if b.Level != tt.level {
			t.Errorf("block %d (%q) level = %d, want %d", i, tt.text, b.Level, tt.level)
		}
	}
}

func TestBlockTypeString(t *testing.T) {
	tests := []struct {
		typ    BlockType
		want   string
		atomic bool
	}{
		{Heading, "heading", false},
		{Paragraph, "paragraph", false},
		{Code, "code", true},
		{Table, "table", true},
		{List, "list", true},
		{Blockquote, "blockquote", false},
		{BlockType(42), "blocktype(42)", false},
	}
	for _, tt := range tests {
		if got := tt.typ.String(); got != tt.want {
			t.Errorf("String(%d) = %q, want %q", int(tt.typ), got, tt.want)
		}
		if got := tt.typ.Atomic(); got != tt.atomic {
			t.Errorf("Atomic(%s) = %v, want %v", tt.want, got, tt.atomic)
		}
	}
}
