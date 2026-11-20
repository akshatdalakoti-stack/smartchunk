// Package split groups parsed blocks into candidate chunks along the
// document's own structure.
//
// Two rules hold for everything this package produces, and every later stage
// preserves them:
//
//   - a code block, table or list is never cut: blocks are the smallest unit
//     a candidate is made of;
//   - a heading always stays with the content below it.
package split

import (
	"strings"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
)

// Candidate is a run of consecutive blocks that will become one chunk unless
// a later stage splits or merges it.
type Candidate struct {
	Blocks []parse.Block
	// Soft is true when the boundary in front of this candidate is a
	// semantic cut inside a section, and false when it is structural (a
	// heading, or the start of the document).
	Soft bool
	// SimPrev is the cosine similarity between this candidate and the one
	// before it. It is only meaningful when HasSim is true.
	SimPrev float64
	HasSim  bool
	// Flags are carried through to the exported chunk.
	Flags []string
}

// Structure groups blocks into one candidate per section: a heading plus
// everything up to the next heading. Consecutive headings with nothing
// between them stay together and attach to the content that follows, and a
// heading at the very end of the document attaches to the section before it,
// so no candidate is ever a bare heading when there is content to join.
func Structure(blocks []parse.Block) []Candidate {
	var out []Candidate
	var cur []parse.Block
	flush := func() {
		if len(cur) > 0 {
			out = append(out, Candidate{Blocks: cur})
			cur = nil
		}
	}
	for _, b := range blocks {
		if b.Type == parse.Heading && len(cur) > 0 && !onlyHeadings(cur) {
			flush()
		}
		cur = append(cur, b)
	}
	flush()

	if n := len(out); n >= 2 && onlyHeadings(out[n-1].Blocks) {
		out[n-2].Blocks = append(out[n-2].Blocks, out[n-1].Blocks...)
		out = out[:n-1]
	}
	return out
}
