// Package eval measures a chunker's output: structural damage, duplicate
// content, retrieval quality, chunk sizes and throughput. It also renders
// the comparison reports.
package eval

import (
	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
)

// Structural counts how a set of chunks treats the structure of the
// documents it was cut from. All counts are judged against the parsed source
// document by line range, so they apply to any chunker's output.
type Structural struct {
	// CodeBlocks, Tables and Lists are how many of each the corpus has.
	CodeBlocks, Tables, Lists int
	// CodeCut, TableCut and ListCut count the blocks that at least one chunk
	// holds only part of.
	CodeCut, TableCut, ListCut int
	// NeverIntact counts cut blocks that no chunk holds in full. With
	// overlapping windows a block can be cut by one chunk and whole in the
	// next; those are cut but not counted here.
	NeverIntact int
	// Headings counts headings that have content after them.
	Headings int
	// OrphanHeadings counts headings stranded at the end of at least one
	// chunk: the chunk holds the heading but stops before the first line
	// of the content below it.
	OrphanHeadings int
	// OrphansNeverJoined counts orphan headings that no chunk holds
	// together with their content. With overlapping windows a heading can
	// be stranded in one chunk and reunited with its content in the next;
	// those are orphans but not counted here.
	OrphansNeverJoined int
}

// Violations is the total number of cut code blocks, tables and lists.
func (s Structural) Violations() int { return s.CodeCut + s.TableCut + s.ListCut }

// AtomicBlocks is the total number of code blocks, tables and lists.
func (s Structural) AtomicBlocks() int { return s.CodeBlocks + s.Tables + s.Lists }

// Add accumulates another document's counts.
func (s *Structural) Add(o Structural) {
	s.CodeBlocks += o.CodeBlocks
	s.Tables += o.Tables
	s.Lists += o.Lists
	s.CodeCut += o.CodeCut
	s.TableCut += o.TableCut
	s.ListCut += o.ListCut
	s.NeverIntact += o.NeverIntact
	s.Headings += o.Headings
	s.OrphanHeadings += o.OrphanHeadings
	s.OrphansNeverJoined += o.OrphansNeverJoined
}

// MeasureStructure compares one document's chunks with its parsed blocks.
func MeasureStructure(blocks []parse.Block, chunks []chunk.Chunk) Structural {
	var s Structural
	for i, b := range blocks {
		switch {
		case b.Type.Atomic():
			cut, intact := cutStatus(b, chunks)
			switch b.Type {
			case parse.Code:
				s.CodeBlocks++
				if cut {
					s.CodeCut++
				}
			case parse.Table:
				s.Tables++
				if cut {
					s.TableCut++
				}
			case parse.List:
				s.Lists++
				if cut {
					s.ListCut++
				}
			}
			if cut && !intact {
				s.NeverIntact++
			}
		case b.Type == parse.Heading && i+1 < len(blocks):
			s.Headings++
			stranded, joined := headingStatus(b.StartLine, blocks[i+1].StartLine, chunks)
			if stranded {
				s.OrphanHeadings++
				if !joined {
					s.OrphansNeverJoined++
				}
			}
		}
	}
	return s
}

// cutStatus reports whether any chunk holds only part of the block, and
// whether any chunk holds all of it.
func cutStatus(b parse.Block, chunks []chunk.Chunk) (cut, intact bool) {
	for _, c := range chunks {
		overlaps := c.StartLine <= b.EndLine && c.EndLine >= b.StartLine
		if !overlaps {
			continue
		}
		if c.StartLine <= b.StartLine && c.EndLine >= b.EndLine {
			intact = true
		} else {
			cut = true
		}
	}
	return cut, intact
}

// CutBy returns the types of the atomic blocks that the chunk holds only
// part of, without repeats, for labelling chunks in the visual comparison.
func CutBy(blocks []parse.Block, c chunk.Chunk) []parse.BlockType {
	var out []parse.BlockType
	seen := map[parse.BlockType]bool{}
	for _, b := range blocks {
		if !b.Type.Atomic() || seen[b.Type] {
			continue
		}
		overlaps := c.StartLine <= b.EndLine && c.EndLine >= b.StartLine
		contains := c.StartLine <= b.StartLine && c.EndLine >= b.EndLine
		if overlaps && !contains {
			seen[b.Type] = true
			out = append(out, b.Type)
		}
	}
	return out
}

// headingStatus reports whether some chunk holds the heading without the
// first line of its content (stranded), and whether some chunk holds both
// (joined).
func headingStatus(headingLine, nextLine int, chunks []chunk.Chunk) (stranded, joined bool) {
	for _, c := range chunks {
		if c.StartLine > headingLine || c.EndLine < headingLine {
			continue
		}
		if c.EndLine >= nextLine {
			joined = true
		} else {
			stranded = true
		}
	}
	return stranded, joined
}
