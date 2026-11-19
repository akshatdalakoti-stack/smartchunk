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
