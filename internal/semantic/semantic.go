// Package semantic refines structural candidates by cutting sections where
// the topic shifts.
//
// Every unit of a document (a prose block plus the code, tables and lists
// attached to it) is embedded. The cosine similarity of each neighbouring
// pair is computed, and a section is cut wherever that similarity falls
// below a percentile of the document's own similarity distribution. Using a
// per-document percentile instead of a fixed number means a tightly focused
// reference page and a rambling tutorial each get cuts at their own weakest
// joins.
package semantic

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/split"
)

// cutEpsilon is how far below the threshold a similarity must fall to count
// as a boundary. Without it, a document whose joins are all equally strong
// would be cut on float32 rounding noise.
const cutEpsilon = 1e-5

// Options tunes boundary detection.
type Options struct {
	// Percentile (0-100) of the document's neighbour similarities below
	// which a boundary is cut. Zero means 25.
	Percentile float64
	// MinPairs is the number of neighbouring pairs a document needs before
	// its distribution is trusted. Shorter documents are left as they are.
	// Zero means 4.
	MinPairs int
}

func (o Options) withDefaults() Options {
	if o.Percentile <= 0 {
		o.Percentile = 25
	}
	if o.Percentile > 100 {
		o.Percentile = 100
	}
	if o.MinPairs <= 0 {
		o.MinPairs = 4
	}
	return o
}

// Refine splits candidates at semantic boundaries and records, on every
// resulting candidate, how similar it is to the one before it. With a nil
// embedder it returns the candidates unchanged.
//
// Units are never divided, so the structural guarantees of package split
// (atomic blocks stay whole, headings stay with their content) still hold.
func Refine(ctx context.Context, cands []split.Candidate, emb embed.Embedder, opt Options) ([]split.Candidate, error) {
	if emb == nil || len(cands) == 0 {
		return cands, nil
	}
	opt = opt.withDefaults()

	type unit struct {
		blocks []parse.Block
		cand   int // index of the candidate the unit came from
	}
	var units []unit
	for i, c := range cands {
		for _, u := range split.Units(c.Blocks) {
			units = append(units, unit{blocks: u, cand: i})
		}
	}
	if len(units) < 2 {
		return cands, nil
	}

	texts := make([]string, len(units))
	for i, u := range units {
		texts[i] = split.JoinText(u.blocks)
	}
	vecs, err := emb.Embed(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("semantic: embedding %d units: %w", len(texts), err)
	}
	if len(vecs) != len(units) {
		return nil, fmt.Errorf("semantic: embedder returned %d vectors for %d units", len(vecs), len(units))
	}

	// sims[i] is the similarity between unit i and unit i+1.
	sims := make([]float64, len(units)-1)
	for i := range sims {
		sims[i] = embed.Cosine(vecs[i], vecs[i+1])
	}
	threshold := math.Inf(-1)
	if len(sims) >= opt.MinPairs {
		threshold = Percentile(sims, opt.Percentile)
	}

	var out []split.Candidate
	for i, u := range units {
		newSection := i == 0 || units[i-1].cand != u.cand
		cut := !newSection && sims[i-1] < threshold-cutEpsilon
		if newSection || cut {
			c := split.Candidate{
				Soft:  cut,
				Flags: append([]string(nil), cands[u.cand].Flags...),
			}
			if i > 0 {
				c.SimPrev, c.HasSim = sims[i-1], true
			}
			out = append(out, c)
		}
		last := &out[len(out)-1]
		last.Blocks = append(last.Blocks, u.blocks...)
	}
	return out, nil
}

// Percentile returns the p-th percentile (0-100) of values using linear
// interpolation between the closest ranks. It returns 0 for no values.
func Percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[len(sorted)-1]
	}
	rank := p / 100 * float64(len(sorted)-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	frac := rank - float64(lo)
	return sorted[lo] + (sorted[hi]-sorted[lo])*frac
}
