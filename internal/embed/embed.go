// Package embed turns text into vectors. It defines the Embedder interface,
// an Ollama HTTP client, an on-disk cache that wraps any embedder, a
// deterministic fake for tests, and the vector maths the rest of SmartChunk
// needs.
package embed

import (
	"context"
	"math"
)

// Embedder converts texts into vectors, one per text, in the same order.
// Implementations must be safe for concurrent use and must stop early when
// the context is cancelled.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Cosine returns the cosine similarity of two vectors in [-1, 1]. Vectors of
// different lengths, empty vectors and zero vectors have similarity 0.
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// Normalize returns a unit-length copy of v. A zero vector is returned
// unchanged (as a copy).
func Normalize(v []float32) []float32 {
	out := make([]float32, len(v))
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		copy(out, v)
		return out
	}
	inv := 1 / math.Sqrt(norm)
	for i, x := range v {
		out[i] = float32(float64(x) * inv)
	}
	return out
}

// Dot returns the dot product of two vectors of equal length, and 0 when
// the lengths differ.
func Dot(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}
