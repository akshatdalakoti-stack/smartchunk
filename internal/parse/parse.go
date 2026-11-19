// Package parse turns a Markdown document into an ordered list of blocks.
//
// It uses goldmark only to build the AST. Everything a later stage needs
// (block type, raw source text, line range, heading path) is derived here so
// that no other package has to know about goldmark.
package parse

import (
	"bytes"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/extension"
	extast "github.com/yuin/goldmark/extension/ast"
	"github.com/yuin/goldmark/text"
)

// BlockType is the structural kind of a top-level Markdown block.
type BlockType int

// Block types, in the order the spec lists them.
const (
	Heading BlockType = iota
	Paragraph
	Code
	Table
	List
	Blockquote
)

var blockTypeNames = [...]string{"heading", "paragraph", "code", "table", "list", "blockquote"}

// String returns the lowercase name of the block type.
func (t BlockType) String() string {
	if t < 0 || int(t) >= len(blockTypeNames) {
		return fmt.Sprintf("blocktype(%d)", int(t))
	}
	return blockTypeNames[t]
}

// Atomic reports whether a block of this type must never be cut.
func (t BlockType) Atomic() bool {
	return t == Code || t == Table || t == List
}

// Block is one top-level block of a Markdown document.
type Block struct {
	Type BlockType
	// Text is the raw source text of the block, exactly as written
	// (fences, list markers and table pipes included).
	Text string
	// StartLine and EndLine are 1-based and inclusive.
	StartLine int
	EndLine   int
	// HeadingPath is the chain of headings the block sits under. For a
	// heading block the path includes the heading itself.
	HeadingPath []string
	// Level is the heading level (1-6) for heading blocks and 0 otherwise.
	Level int
}

var (
	setextUnderline = regexp.MustCompile(`^ {0,3}(=+|-+)\s*$`)
	tableDelimRow   = regexp.MustCompile(`^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$`)
)

// md is safe for concurrent use: goldmark parsers hold no per-document state.
var md = goldmark.New(goldmark.WithExtensions(extension.Table))

// Parse converts Markdown source into blocks in document order.
//
// Line endings are normalised to "\n" and YAML front matter is ignored, in
// both cases without shifting line numbers.
func Parse(source []byte) (blocks []Block, err error) {
	// goldmark is third-party code running over arbitrary input. A panic in
	// it must not take the caller down with it.
	defer func() {
		if r := recover(); r != nil {
			blocks, err = nil, fmt.Errorf("parse: markdown parser panicked: %v", r)
		}
	}()

	src := normalize(source)
	li := newLineIndex(src)
	doc := md.Parser().Parse(text.NewReader(src))

	var nodes []ast.Node
	for n := doc.FirstChild(); n != nil; n = n.NextSibling() {
		nodes = append(nodes, n)
	}

	var stack []headingEntry
	prevEnd := 0
	for i, n := range nodes {
		typ, ok := classify(n)
		if !ok {
			continue
		}
		start := prevEnd + 1
		if p := startOffset(n); p >= 0 {
			if line := li.lineOf(p); line > start {
				start = line
			}
		}
		start = li.skipBlank(start)
		if start > li.count() {
			continue
		}

		end := li.lineOf(maxOffset(n, src))
		end = extendForClosers(n, src, li, end)
		// A block can never run into the next top-level block.
		if i+1 < len(nodes) {
			if np := startOffset(nodes[i+1]); np >= 0 {
				if limit := li.lineOf(np) - 1; end > limit {
					end = limit
				}
			}
		}
		if end < start {
			end = start
		}
		for end > start && li.isBlank(end) {
			end--
		}
		prevEnd = end

		b := Block{
			Type:      typ,
			Text:      li.text(start, end),
			StartLine: start,
			EndLine:   end,
		}
		if h, isHeading := n.(*ast.Heading); isHeading {
			b.Level = h.Level
			title := strings.TrimSpace(inlineText(h, src))
			for len(stack) > 0 && stack[len(stack)-1].level >= h.Level {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, headingEntry{level: h.Level, title: title})
		}
		b.HeadingPath = pathOf(stack)
		blocks = append(blocks, b)
	}
	return blocks, nil
}

type headingEntry struct {
	level int
	title string
}

func pathOf(stack []headingEntry) []string {
	path := make([]string, len(stack))
	for i, h := range stack {
		path[i] = h.title
	}
	return path
}

// classify maps a top-level AST node to a block type. Nodes that carry no
// content of their own (thematic breaks, link reference definitions) are
// skipped.
func classify(n ast.Node) (BlockType, bool) {
	switch n.Kind() {
	case ast.KindHeading:
		return Heading, true
	case ast.KindParagraph, ast.KindTextBlock, ast.KindHTMLBlock:
		return Paragraph, true
	case ast.KindFencedCodeBlock, ast.KindCodeBlock:
		return Code, true
	case extast.KindTable:
		return Table, true
	case ast.KindList:
		return List, true
	case ast.KindBlockquote:
		return Blockquote, true
	case ast.KindThematicBreak, ast.KindLinkReferenceDefinition:
		return 0, false
	default:
		// Unknown block kinds (from future goldmark versions) are kept as
		// prose rather than dropped.
		return Paragraph, true
	}
}

// normalize converts CRLF/CR line endings to LF and blanks out YAML front
// matter. Both operations keep the number of lines unchanged.
func normalize(source []byte) []byte {
	src := bytes.ReplaceAll(source, []byte("\r\n"), []byte("\n"))
	src = bytes.ReplaceAll(src, []byte("\r"), []byte("\n"))
	src = bytes.TrimPrefix(src, []byte("\xef\xbb\xbf"))
	return blankFrontMatter(src)
}

// blankFrontMatter replaces a leading "---" ... "---" YAML block with empty
// lines. Left in place, goldmark reads "title: x\n---" as a setext heading.
func blankFrontMatter(src []byte) []byte {
	if !bytes.HasPrefix(src, []byte("---\n")) {
		return src
	}
	lines := bytes.Split(src, []byte("\n"))
	closing := -1
	for i := 1; i < len(lines); i++ {
		l := string(bytes.TrimRight(lines[i], " \t"))
		if l == "---" || l == "..." {
			closing = i
			break
		}
	}
	if closing < 0 {
		return src
	}
	out := make([]byte, 0, len(src))
	for i, l := range lines {
		if i > closing {
			out = append(out, l...)
		}
		if i < len(lines)-1 {
			out = append(out, '\n')
		}
	}
	return out
}

// startOffset returns the offset where a top-level node begins, or -1 when
// goldmark recorded none. A table is the exception to trusting Pos: when a
// table directly follows paragraph text, goldmark carves it out of that
// paragraph and leaves the table's Pos at the paragraph's first line. Its
// real start is its header row.
func startOffset(n ast.Node) int {
	if n.Kind() != extast.KindTable {
		return n.Pos()
	}
	start := -1
	_ = ast.Walk(n, func(c ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering || c == n {
			return ast.WalkContinue, nil
		}
		note := func(p int) {
			if p >= 0 && (start < 0 || p < start) {
				start = p
			}
		}
		note(c.Pos())
		if t, ok := c.(*ast.Text); ok {
			note(t.Segment.Start)
		}
		return ast.WalkContinue, nil
	})
	if start < 0 {
		return n.Pos()
	}
	return start
}

// maxOffset returns the offset of the last source byte that belongs to n or
// any of its descendants.
func maxOffset(n ast.Node, src []byte) int {
	end := 0
	if p := n.Pos(); p > end {
		end = p
	}
	note := func(stop int) {
		// Segment stops are exclusive and usually sit just past a newline.
		if stop-1 > end {
			end = stop - 1
		}
	}
	_ = ast.Walk(n, func(c ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		if p := c.Pos(); p > end {
			end = p
		}
		switch v := c.(type) {
		case *ast.Text:
			note(v.Segment.Stop)
		case *ast.RawHTML:
			if v.Segments != nil {
				for i := 0; i < v.Segments.Len(); i++ {
					note(v.Segments.At(i).Stop)
				}
			}
		case *ast.FencedCodeBlock:
			if v.Info != nil {
				note(v.Info.Segment.Stop)
			}
		case *ast.HTMLBlock:
			if v.HasClosure() {
				note(v.ClosureLine.Stop)
			}
		}
		if c.Type() == ast.TypeBlock {
			if lines := c.Lines(); lines != nil {
				for i := 0; i < lines.Len(); i++ {
					note(lines.At(i).Stop)
				}
			}
		}
		return ast.WalkContinue, nil
	})
	if end >= len(src) {
		end = len(src) - 1
	}
	if end < 0 {
		end = 0
	}
	return end
}

// extendForClosers widens end to cover syntax that closes a block but that
// goldmark does not attribute to any node: the closing fence of a code block,
// the underline of a setext heading and a table's delimiter row.
func extendForClosers(n ast.Node, src []byte, li *lineIndex, end int) int {
	_ = ast.Walk(n, func(c ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		if _, ok := c.(*ast.FencedCodeBlock); ok {
			last := li.lineOf(maxOffset(c, src))
			if isFenceLine(li.line(last + 1)) {
				if last+1 > end {
					end = last + 1
				}
			}
		}
		return ast.WalkContinue, nil
	})
	switch v := n.(type) {
	case *ast.Heading:
		first := strings.TrimLeft(li.line(li.lineOf(maxInt(v.Pos(), 0))), " ")
		if !strings.HasPrefix(first, "#") && setextUnderline.MatchString(li.line(end+1)) {
			end++
		}
	case *extast.Table:
		if tableDelimRow.MatchString(li.line(end+1)) && strings.Contains(li.line(end+1), "-") {
			end++
		}
	}
	return end
}

// isFenceLine reports whether a line is a code fence, ignoring any list
// indentation or blockquote markers in front of it.
func isFenceLine(line string) bool {
	s := strings.TrimLeft(line, " \t>")
	return strings.HasPrefix(s, "```") || strings.HasPrefix(s, "~~~")
}

// inlineText collects the plain text of a node's inline children.
func inlineText(n ast.Node, src []byte) string {
	var sb strings.Builder
	_ = ast.Walk(n, func(c ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}
		switch v := c.(type) {
		case *ast.Text:
			sb.Write(v.Segment.Value(src))
			if v.SoftLineBreak() || v.HardLineBreak() {
				sb.WriteByte(' ')
			}
		case *ast.String:
			sb.Write(v.Value)
		}
		return ast.WalkContinue, nil
	})
	return sb.String()
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
