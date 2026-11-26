package eval

import (
	"fmt"
	"math"
	"sort"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/dedup"
)

// TokenStats summarises chunk sizes in tokens.
type TokenStats struct {
	Mean float64
	P95  int
	Max  int
}

// MeasureTokens computes size statistics over the chunks' token counts. The
// 95th percentile uses the nearest-rank method, so it is always a size that
// some chunk actually has.
func MeasureTokens(chunks []chunk.Chunk) TokenStats {
	if len(chunks) == 0 {
		return TokenStats{}
	}
	counts := make([]int, len(chunks))
	sum := 0
	for i, c := range chunks {
		counts[i] = c.TokenCount
		sum += c.TokenCount
	}
	sort.Ints(counts)
	rank := int(math.Ceil(0.95*float64(len(counts)))) - 1
	if rank < 0 {
		rank = 0
	}
	return TokenStats{
		Mean: float64(sum) / float64(len(counts)),
		P95:  counts[rank],
		Max:  counts[len(counts)-1],
	}
}

// MeasureDuplicates returns how many chunks are near-duplicates of an
// earlier chunk at the given Jaccard threshold, and that number as a share
// of all chunks.
func MeasureDuplicates(chunks []chunk.Chunk, threshold float64) (count int, rate float64, err error) {
	if len(chunks) == 0 {
		return 0, 0, nil
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	dupOf, err := dedup.FindDuplicates(texts, dedup.Options{Threshold: threshold})
	if err != nil {
		return 0, 0, fmt.Errorf("eval: measuring duplicates: %w", err)
	}
	for _, d := range dupOf {
		if d >= 0 {
			count++
		}
	}
	return count, float64(count) / float64(len(chunks)), nil
}
