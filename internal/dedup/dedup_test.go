package dedup

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/split"
)

// vocabText returns n pseudo-random words drawn with a fixed seed.
func vocabText(seed int64, n int) []string {
	rng := rand.New(rand.NewSource(seed))
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("word%d", rng.Intn(5000))
	}
	return out
}

func TestWords(t *testing.T) {
	tests := []struct {
		text string
		want []string
	}{
		{"", nil},
		{"Hello, World!", []string{"hello", "world"}},
		{"  spaced\tout\nlines ", []string{"spaced", "out", "lines"}},
		{"snake_case and kebab-case", []string{"snake", "case", "and", "kebab", "case"}},
		{"v1.22 Ünïcode", []string{"v1", "22", "ünïcode"}},
		{"```go\nx := 1\n```", []string{"go", "x", "1"}},
	}
	for _, tt := range tests {
		if got := Words(tt.text); !reflect.DeepEqual(got, tt.want) && !(len(got) == 0 && len(tt.want) == 0) {
			t.Errorf("Words(%q) = %q, want %q", tt.text, got, tt.want)
		}
	}
}

func TestSignatureBasics(t *testing.T) {
	h := NewHasher(0, 0)
	tests := []struct {
		name    string
		a, b    string
		wantSim float64
	}{
		{"identical", "the quick brown fox jumps", "the quick brown fox jumps", 1},
		{"case and punctuation ignored", "The quick, brown fox JUMPS!", "the quick brown fox jumps", 1},
		{"whitespace ignored", "the quick\n\nbrown   fox jumps", "the quick brown fox jumps", 1},
		{"disjoint", "alpha beta gamma delta epsilon", "one two three four five six", 0},
		{"shorter than a shingle", "hi there", "hi there", 1},
		{"empty against text", "", "some words here", 0},
		{"both empty", "", "", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Similarity(h.Signature(tt.a), h.Signature(tt.b))
			if got != tt.wantSim {
				t.Errorf("Similarity = %v, want %v", got, tt.wantSim)
			}
		})
	}
	if sig := h.Signature("a b c d"); len(sig) != DefaultNumHashes {
		t.Errorf("signature length = %d, want %d", len(sig), DefaultNumHashes)
	}
	if h.Signature("...!!!") != nil {
		t.Error("text without words should have a nil signature")
	}
	if Similarity(Signature{1, 2}, Signature{1, 2, 3}) != 0 {
		t.Error("signatures of different lengths should have similarity 0")
	}
	a, b := NewHasher(64, 3).Signature("same text here now"), NewHasher(64, 3).Signature("same text here now")
	if !reflect.DeepEqual(a, b) {
		t.Error("two hashers with the same settings disagree: seeds are not fixed")
	}
}

// TestMinHashEstimatesJaccard checks the estimate against the exact value
// across the whole similarity range.
func TestMinHashEstimatesJaccard(t *testing.T) {
	h := NewHasher(256, 3)
	base := vocabText(1, 400)
	for _, changed := range []int{0, 10, 40, 100, 200, 400} {
		t.Run(fmt.Sprintf("changed%d", changed), func(t *testing.T) {
			other := append([]string{}, base...)
			repl := vocabText(int64(changed)+99, changed)
			for i := 0; i < changed; i++ {
				other[i*len(base)/changed] = "x" + repl[i]
			}
			a, b := strings.Join(base, " "), strings.Join(other, " ")
			exact := h.Jaccard(a, b)
			est := Similarity(h.Signature(a), h.Signature(b))
			// Standard error at 256 hashes is at most 0.5/16 = 0.031.
			if math.Abs(est-exact) > 0.1 {
				t.Errorf("estimate %.3f is too far from exact Jaccard %.3f", est, exact)
			}
		})
	}
}
