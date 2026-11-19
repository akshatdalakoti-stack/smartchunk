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
