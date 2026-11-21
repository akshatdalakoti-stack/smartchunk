package embed

import (
	"context"
	"hash/fnv"
	"strings"
	"unicode"
)

// Fake is a deterministic embedder for tests: a hashed bag of words. Texts
// that share words get similar vectors, and the same text always gets the
// same vector on every machine, with no model and no network.
type Fake struct {
	// Dim is the vector length. Zero means 64.
	Dim int
}

// Embed implements Embedder.
func (f Fake) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	dim := f.Dim
	if dim <= 0 {
		dim = 64
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		v := make([]float32, dim)
		for _, w := range strings.FieldsFunc(strings.ToLower(t), notWord) {
			h := fnv.New64a()
			h.Write([]byte(w))
			sum := h.Sum64()
			sign := float32(1)
			if sum>>63 == 1 {
				sign = -1
			}
			v[sum%uint64(dim)] += sign
		}
		out[i] = Normalize(v)
	}
	return out, nil
}

func notWord(r rune) bool {
	return !unicode.IsLetter(r) && !unicode.IsDigit(r)
}
