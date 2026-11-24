package dedup

import (
	"math"
	"strings"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/split"
)

// LineOptions decides which lines count as repeated boilerplate.
type LineOptions struct {
	// MinRepeats is the smallest number of chunks a line must appear in.
	// Zero means 5.
	MinRepeats int
	// MinRatio is the smallest share of all chunks a line must appear in.
	// Zero means 0.05. Both conditions have to hold, so a phrase that
	// recurs a handful of times in a large corpus is left alone.
	MinRatio float64
	// MinLength ignores lines shorter than this many characters. Zero
	// means 8.
	MinLength int
}

func (o LineOptions) withDefaults() LineOptions {
	if o.MinRepeats <= 0 {
		o.MinRepeats = 5
	}
	if o.MinRatio <= 0 {
		o.MinRatio = 0.05
	}
	if o.MinLength <= 0 {
		o.MinLength = 8
	}
	return o
}

// normalizeLine makes lines comparable: trimmed, lowercased, inner
// whitespace collapsed.
func normalizeLine(line string) string {
	return strings.Join(strings.Fields(strings.ToLower(line)), " ")
}

// RepeatedLines returns the normalised lines that appear in enough of the
// candidates to be considered boilerplate. Only prose is examined: a line
// such as "```" or "| --- |" repeats everywhere and means nothing.
func RepeatedLines(cands []split.Candidate, opt LineOptions) map[string]bool {
	opt = opt.withDefaults()
	freq := map[string]int{}
	for _, c := range cands {
		seen := map[string]bool{}
		for _, b := range c.Blocks {
			if !split.IsProse(b.Type) {
				continue
			}
			for _, line := range strings.Split(b.Text, "\n") {
				n := normalizeLine(line)
				if len(n) < opt.MinLength || seen[n] {
					continue
				}
				seen[n] = true
				freq[n]++
			}
		}
	}
	need := int(math.Ceil(opt.MinRatio * float64(len(cands))))
	if need < opt.MinRepeats {
		need = opt.MinRepeats
	}
	out := map[string]bool{}
	for line, n := range freq {
		if n >= need {
			out[line] = true
		}
	}
	return out
}

// StripLines removes the given normalised lines from the prose blocks of
// every candidate. Blocks left empty are dropped, candidates left with no
// content are dropped, and candidates that lost lines are flagged. It
// returns the surviving candidates and the number of lines removed.
func StripLines(cands []split.Candidate, lines map[string]bool) ([]split.Candidate, int) {
	if len(lines) == 0 {
		return cands, 0
	}
	removed := 0
	out := make([]split.Candidate, 0, len(cands))
	for _, c := range cands {
		changed := false
		blocks := make([]parse.Block, 0, len(c.Blocks))
		for _, b := range c.Blocks {
			if !split.IsProse(b.Type) {
				blocks = append(blocks, b)
				continue
			}
			kept, n := stripBlock(b, lines)
			if n == 0 {
				blocks = append(blocks, b)
				continue
			}
			removed += n
			changed = true
			if kept != nil {
				blocks = append(blocks, *kept)
			}
		}
		if !changed {
			out = append(out, c)
			continue
		}
		c.Blocks = blocks
		if len(blocks) == 0 || c.OnlyHeadings() {
			continue
		}
		c.AddFlag(chunk.FlagBoilerplateStripped)
		out = append(out, c)
	}
	return out, removed
}

// stripBlock removes boilerplate lines from one block. It returns nil when
// nothing is left, and the number of lines removed. The block's line range
// shrinks to the lines that remain.
func stripBlock(b parse.Block, lines map[string]bool) (*parse.Block, int) {
	src := strings.Split(b.Text, "\n")
	var kept []string
	first, last, removed := -1, -1, 0
	for i, line := range src {
		if lines[normalizeLine(line)] {
			removed++
			continue
		}
		kept = append(kept, line)
		if strings.TrimSpace(line) != "" {
			if first < 0 {
				first = i
			}
			last = i
		}
	}
	if removed == 0 {
		return &b, 0
	}
	if first < 0 {
		return nil, removed
	}
	out := b
	out.Text = strings.TrimSpace(strings.Join(kept, "\n"))
	out.StartLine = b.StartLine + first
	out.EndLine = b.StartLine + last
	return &out, removed
}
