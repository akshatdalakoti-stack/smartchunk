package eval

import (
	"container/heap"
	"sort"

	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
)

// Hit is one search result: the position of a vector in the index and its
// cosine similarity to the query.
type Hit struct {
	ID    int
	Score float64
}

// Index is an in-memory, brute-force cosine similarity index. Vectors are
// normalised once when the index is built, so a search is one dot product
// per stored vector.
type Index struct {
	vecs [][]float32
}

// NewIndex builds an index over the vectors. The id of a vector is its
// position in the slice.
func NewIndex(vecs [][]float32) *Index {
	ix := &Index{vecs: make([][]float32, len(vecs))}
	for i, v := range vecs {
		ix.vecs[i] = embed.Normalize(v)
	}
	return ix
}

// Len returns the number of vectors in the index.
func (ix *Index) Len() int { return len(ix.vecs) }

// better reports whether a ranks ahead of b: higher score first, lower id
// on ties, so rankings are deterministic.
func better(a, b Hit) bool {
	if a.Score != b.Score {
		return a.Score > b.Score
	}
	return a.ID < b.ID
}

// worstFirst is a heap that keeps the worst of the current top-k at the
// root, ready to be replaced.
type worstFirst []Hit

func (h worstFirst) Len() int           { return len(h) }
func (h worstFirst) Less(i, j int) bool { return better(h[j], h[i]) }
func (h worstFirst) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *worstFirst) Push(x any)        { *h = append(*h, x.(Hit)) }
func (h *worstFirst) Pop() any {
	old := *h
	last := old[len(old)-1]
	*h = old[:len(old)-1]
	return last
}

// Search returns the k most similar vectors, best first. A k of zero or
// less, or larger than the index, returns the full ranking.
func (ix *Index) Search(query []float32, k int) []Hit {
	q := embed.Normalize(query)
	n := len(ix.vecs)
	if k <= 0 || k >= n {
		hits := make([]Hit, n)
		for i, v := range ix.vecs {
			hits[i] = Hit{ID: i, Score: embed.Dot(q, v)}
		}
		sort.Slice(hits, func(i, j int) bool { return better(hits[i], hits[j]) })
		return hits
	}

	top := make(worstFirst, 0, k)
	for i, v := range ix.vecs {
		h := Hit{ID: i, Score: embed.Dot(q, v)}
		if len(top) < k {
			heap.Push(&top, h)
		} else if better(h, top[0]) {
			top[0] = h
			heap.Fix(&top, 0)
		}
	}
	hits := []Hit(top)
	sort.Slice(hits, func(i, j int) bool { return better(hits[i], hits[j]) })
	return hits
}
