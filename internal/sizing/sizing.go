// Package sizing brings candidate chunks into a token range: it splits the
// ones that are too large and merges the ones that are too small.
//
// Splitting follows a strict order of preference. A candidate is cut between
// units first (a paragraph and the code that follows it stay together),
// then between blocks, and only then inside a prose block, at sentence
// boundaries. Code blocks, tables and lists are never cut: one that is over
// the limit on its own stays whole and is flagged "oversized".
package sizing

import (
	"fmt"
	"math"
	"strings"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/split"
)

// Options sets the token range.
type Options struct {
	// MaxTokens is the upper limit for a chunk.
	MaxTokens int
	// MinTokens is the size below which a chunk is merged into a neighbour.
	MinTokens int
	// Tokenizer defaults to chunk.ApproxTokenizer.
	Tokenizer chunk.Tokenizer
}

// DefaultOptions returns the range used when the CLI is given no flags.
func DefaultOptions() Options {
	return Options{MaxTokens: 256, MinTokens: 64, Tokenizer: chunk.ApproxTokenizer{}}
}

// Validate reports whether the range is usable.
func (o Options) Validate() error {
	if o.MaxTokens <= 0 {
		return fmt.Errorf("sizing: max tokens must be positive, got %d", o.MaxTokens)
	}
	if o.MinTokens < 0 {
		return fmt.Errorf("sizing: min tokens must not be negative, got %d", o.MinTokens)
	}
	if o.MinTokens >= o.MaxTokens {
		return fmt.Errorf("sizing: min tokens (%d) must be smaller than max tokens (%d)", o.MinTokens, o.MaxTokens)
	}
	return nil
}

// Apply splits oversized candidates and then merges undersized ones.
func Apply(cands []split.Candidate, opt Options) ([]split.Candidate, error) {
	if err := opt.Validate(); err != nil {
		return nil, err
	}
	if opt.Tokenizer == nil {
		opt.Tokenizer = chunk.ApproxTokenizer{}
	}
	s := sizer{opt: opt}

	var sized []split.Candidate
	for _, c := range cands {
		sized = append(sized, s.splitOversized(c)...)
	}
	return s.mergeUndersized(sized), nil
}

type sizer struct {
	opt Options
}

func (s sizer) count(text string) int { return s.opt.Tokenizer.Count(text) }

func (s sizer) countBlocks(blocks []parse.Block) int {
	n := 0
	for _, b := range blocks {
		n += s.count(b.Text)
	}
	return n
}

// item is a run of blocks that packing treats as indivisible.
type item struct {
	blocks []parse.Block
	tokens int
	// origin is non-zero for pieces cut out of one prose block; pieces with
	// the same origin are never packed back together.
	origin int
}

// splitOversized returns the candidate unchanged when it fits, and otherwise
// repacks it into several candidates of roughly equal size.
func (s sizer) splitOversized(c split.Candidate) []split.Candidate {
	max := s.opt.MaxTokens
	total := s.countBlocks(c.Blocks)
	if total <= max {
		return []split.Candidate{c}
	}

	var items []item
	origin := 0
	for _, unit := range split.Units(c.Blocks) {
		if t := s.countBlocks(unit); t <= max {
			items = append(items, item{blocks: unit, tokens: t})
			continue
		}
		// The unit is too big to keep together: fall back to its blocks,
		// with headings glued to the block that follows them.
		var headings []parse.Block
		headingTokens := 0
		for _, b := range unit {
			bt := s.count(b.Text)
			if b.Type == parse.Heading {
				headings = append(headings, b)
				headingTokens += bt
				continue
			}
			if splittable(b) && headingTokens+bt > max {
				budget := max - headingTokens
				if budget < max/2 {
					budget = max / 2
				}
				origin++
				for k, piece := range s.splitProse(b, budget) {
					blocks := []parse.Block{piece}
					tokens := s.count(piece.Text)
					if k == 0 {
						blocks = append(headings, piece)
						tokens += headingTokens
					}
					items = append(items, item{blocks: blocks, tokens: tokens, origin: origin})
				}
			} else {
				items = append(items, item{blocks: append(headings, b), tokens: headingTokens + bt})
			}
			headings, headingTokens = nil, 0
		}
		if len(headings) > 0 {
			items = append(items, item{blocks: headings, tokens: headingTokens})
		}
	}

	// Balanced greedy packing: aim for equal parts rather than filling each
	// chunk to the limit and leaving a scrap at the end.
	parts := int(math.Ceil(float64(total) / float64(max)))
	target := float64(total) / float64(parts)

	var out []split.Candidate
	var cur split.Candidate
	curTokens, curOrigin := 0, 0
	flush := func() {
		if len(cur.Blocks) == 0 {
			return
		}
		if curTokens > max {
			cur.AddFlag(chunk.FlagOversized)
		}
		if len(out) == 0 {
			cur.Soft, cur.SimPrev, cur.HasSim = c.Soft, c.SimPrev, c.HasSim
		} else {
			cur.Soft = true // a cut inside what was one candidate
		}
		for _, f := range c.Flags {
			cur.AddFlag(f)
		}
		out = append(out, cur)
		cur = split.Candidate{}
		curTokens, curOrigin = 0, 0
	}
	for _, it := range items {
		samePiece := it.origin != 0 && it.origin == curOrigin
		if len(cur.Blocks) > 0 && (curTokens+it.tokens > max || samePiece) {
			flush()
		}
		cur.Blocks = append(cur.Blocks, it.blocks...)
		curTokens += it.tokens
		if it.origin != 0 {
			curOrigin = it.origin
			cur.AddFlag(chunk.FlagSentenceSplit)
		}
		if float64(curTokens) >= target {
			flush()
		}
	}
	flush()

	// A trailing piece that is nothing but headings joins the piece before.
	if n := len(out); n >= 2 && out[n-1].OnlyHeadings() {
		out[n-2].Blocks = append(out[n-2].Blocks, out[n-1].Blocks...)
		out = out[:n-1]
	}
	return out
}

// splittable reports whether a block may be cut at sentence boundaries.
// Blockquotes that contain a code fence are treated as atomic, since a cut
// could land inside the fence.
func splittable(b parse.Block) bool {
	switch b.Type {
	case parse.Paragraph:
		return true
	case parse.Blockquote:
		return !strings.Contains(b.Text, "```") && !strings.Contains(b.Text, "~~~")
	default:
		return false
	}
}

// splitProse cuts a prose block into pieces of at most budget tokens,
// preferring sentence boundaries, then line breaks, then (for a single line
// with no usable boundary at all) a hard cut between tokens.
func (s sizer) splitProse(b parse.Block, budget int) []parse.Block {
	text := b.Text
	var segs [][2]int
	for _, sp := range sentenceSpans(text) {
		if s.count(text[sp[0]:sp[1]]) <= budget {
			segs = append(segs, sp)
			continue
		}
		for _, ln := range lineSpans(text, sp) {
			if s.count(text[ln[0]:ln[1]]) <= budget {
				segs = append(segs, ln)
				continue
			}
			segs = append(segs, s.hardSpans(text, ln, budget)...)
		}
	}

	total := s.count(text)
	parts := int(math.Ceil(float64(total) / float64(budget)))
	target := float64(total) / float64(parts)

	var pieces []parse.Block
	start, end, tokens := -1, -1, 0
	flush := func() {
		if start < 0 {
			return
		}
		if p, ok := subBlock(b, start, end); ok {
			pieces = append(pieces, p)
		}
		start, end, tokens = -1, -1, 0
	}
	for _, sg := range segs {
		t := s.count(text[sg[0]:sg[1]])
		if start >= 0 && tokens+t > budget {
			flush()
		}
		if start < 0 {
			start = sg[0]
		}
		end = sg[1]
		tokens += t
		if float64(tokens) >= target {
			flush()
		}
	}
	flush()
	if len(pieces) == 0 {
		return []parse.Block{b}
	}
	return pieces
}

// lineSpans splits a span of text at newlines.
func lineSpans(text string, sp [2]int) [][2]int {
	var out [][2]int
	start := sp[0]
	for i := sp[0]; i < sp[1]; i++ {
		if text[i] == '\n' {
			out = append(out, [2]int{start, i + 1})
			start = i + 1
		}
	}
	if start < sp[1] {
		out = append(out, [2]int{start, sp[1]})
	}
	return out
}

// hardSpans cuts a span every budget tokens. It is the last resort for text
// with no sentence or line boundary to use.
func (s sizer) hardSpans(text string, sp [2]int, budget int) [][2]int {
	toks := s.opt.Tokenizer.Spans(text[sp[0]:sp[1]])
	if len(toks) == 0 {
		return [][2]int{sp}
	}
	var out [][2]int
	start := sp[0]
	for i := budget; i < len(toks); i += budget {
		cut := sp[0] + toks[i].Start
		out = append(out, [2]int{start, cut})
		start = cut
	}
	return append(out, [2]int{start, sp[1]})
}

// subBlock builds a block from the byte range [from, to) of b's text,
// trimming whitespace and working out the piece's own line range.
func subBlock(b parse.Block, from, to int) (parse.Block, bool) {
	raw := b.Text[from:to]
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return parse.Block{}, false
	}
	lead := from + strings.Index(raw, trimmed)
	startLine := b.StartLine + strings.Count(b.Text[:lead], "\n")
	endLine := startLine + strings.Count(trimmed, "\n")
	return parse.Block{
		Type:        b.Type,
		Text:        trimmed,
		StartLine:   startLine,
		EndLine:     endLine,
		HeadingPath: b.HeadingPath,
		Level:       b.Level,
	}, true
}
