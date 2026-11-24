// Package dedup removes noise from a set of chunks: lines repeated across
// many chunks (headers, footers, navigation) and chunks that are
// near-duplicates of an earlier chunk.
//
// Near-duplicate detection is MinHash over word shingles with LSH banding,
// written from scratch: a chunk's signature estimates its Jaccard similarity
// to any other chunk, and banding finds likely matches without comparing
// every pair.
package dedup

import (
	"fmt"
	"math"
	"strings"
	"unicode"
)

// Defaults for near-duplicate detection.
const (
	DefaultThreshold   = 0.85
	DefaultNumHashes   = 128
	DefaultShingleSize = 3
)

// Signature is a MinHash signature. Two signatures made by the same Hasher
// agree in a fraction of positions that estimates the Jaccard similarity of
// the texts they came from. A nil Signature means the text had no words.
type Signature []uint64

// Hasher computes MinHash signatures. It is immutable and safe for
// concurrent use.
type Hasher struct {
	seeds   []uint64
	shingle int
}

// NewHasher returns a Hasher producing signatures of numHashes values over
// shingles of shingleSize consecutive words. Non-positive arguments fall
// back to the defaults. The hash seeds are fixed, so signatures are
// reproducible across runs and machines.
func NewHasher(numHashes, shingleSize int) *Hasher {
	if numHashes <= 0 {
		numHashes = DefaultNumHashes
	}
	if shingleSize <= 0 {
		shingleSize = DefaultShingleSize
	}
	seeds := make([]uint64, numHashes)
	state := uint64(0x9E3779B97F4A7C15)
	for i := range seeds {
		state += 0x9E3779B97F4A7C15
		seeds[i] = mix64(state)
	}
	return &Hasher{seeds: seeds, shingle: shingleSize}
}

// mix64 is the SplitMix64 finaliser: a fast, well-distributed 64-bit
// permutation.
func mix64(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xBF58476D1CE4E5B9
	x ^= x >> 27
	x *= 0x94D049BB133111EB
	x ^= x >> 31
	return x
}

// Words lowercases text and returns its runs of letters and digits.
// Punctuation and whitespace are ignored, so reflowed or re-indented copies
// of the same content produce the same words.
func Words(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// shingles returns the hashes of every run of h.shingle consecutive words.
// A text shorter than one shingle is treated as a single shingle.
func (h *Hasher) shingles(text string) []uint64 {
	words := Words(text)
	if len(words) == 0 {
		return nil
	}
	n := h.shingle
	if len(words) < n {
		n = len(words)
	}
	out := make([]uint64, 0, len(words)-n+1)
	for i := 0; i+n <= len(words); i++ {
		// FNV-1a over the words of the shingle, with a separator so that
		// ("ab","c") and ("a","bc") hash differently.
		sum := uint64(14695981039346656037)
		for _, w := range words[i : i+n] {
			for j := 0; j < len(w); j++ {
				sum ^= uint64(w[j])
				sum *= 1099511628211
			}
			sum ^= 0xFF
			sum *= 1099511628211
		}
		out = append(out, sum)
	}
	return out
}

// Signature computes the MinHash signature of text.
func (h *Hasher) Signature(text string) Signature {
	shingles := h.shingles(text)
	if len(shingles) == 0 {
		return nil
	}
	sig := make(Signature, len(h.seeds))
	for i, seed := range h.seeds {
		min := uint64(math.MaxUint64)
		for _, s := range shingles {
			if v := mix64(s ^ seed); v < min {
				min = v
			}
		}
		sig[i] = min
	}
	return sig
}

// Similarity estimates the Jaccard similarity of the texts behind two
// signatures: the fraction of positions in which they agree. Signatures of
// different lengths, or nil signatures, have similarity 0.
func Similarity(a, b Signature) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	same := 0
	for i := range a {
		if a[i] == b[i] {
			same++
		}
	}
	return float64(same) / float64(len(a))
}

// Jaccard is the exact Jaccard similarity of the shingle sets of two texts.
// It is quadratic in memory compared with signatures and exists to verify
// the MinHash estimate in tests.
func (h *Hasher) Jaccard(a, b string) float64 {
	sa, sb := map[uint64]bool{}, map[uint64]bool{}
	for _, s := range h.shingles(a) {
		sa[s] = true
	}
	for _, s := range h.shingles(b) {
		sb[s] = true
	}
	if len(sa) == 0 || len(sb) == 0 {
		return 0
	}
	inter := 0
	for s := range sa {
		if sb[s] {
			inter++
		}
	}
	return float64(inter) / float64(len(sa)+len(sb)-inter)
}

// Index is an LSH index over signatures. A signature is cut into bands of a
// few rows each; two signatures that agree on every row of any one band land
// in the same bucket and become candidates for a full comparison.
type Index struct {
	threshold float64
	bands     int
	rows      int
	buckets   []map[uint64][]int
	sigs      []Signature
}

// NewIndex returns an index for signatures of numHashes values that reports
// matches with estimated similarity of at least threshold.
func NewIndex(numHashes int, threshold float64) (*Index, error) {
	if numHashes <= 0 {
		return nil, fmt.Errorf("dedup: number of hashes must be positive, got %d", numHashes)
	}
	if threshold <= 0 || threshold > 1 || math.IsNaN(threshold) {
		return nil, fmt.Errorf("dedup: threshold must be in (0, 1], got %v", threshold)
	}
	bands, rows := chooseBands(numHashes, threshold)
	ix := &Index{threshold: threshold, bands: bands, rows: rows, buckets: make([]map[uint64][]int, bands)}
	for i := range ix.buckets {
		ix.buckets[i] = map[uint64][]int{}
	}
	return ix, nil
}

// chooseBands picks the banding for a threshold. With b bands of r rows, a
// pair of similarity s becomes a candidate with probability 1-(1-s^r)^b, an
// S-curve whose steep part sits near (1/b)^(1/r). The widest rows that keep
// that point a safe margin below the threshold give the fewest false
// candidates while still catching almost every true match.
func chooseBands(numHashes int, threshold float64) (bands, rows int) {
	const margin = 0.1
	bands, rows = numHashes, 1
	for r := 2; r <= numHashes; r++ {
		if numHashes%r != 0 {
			continue
		}
		b := numHashes / r
		if math.Pow(1/float64(b), 1/float64(r)) <= threshold-margin {
			bands, rows = b, r
		}
	}
	return bands, rows
}

func (ix *Index) bandKey(sig Signature, band int) uint64 {
	sum := uint64(band) + 0x9E3779B97F4A7C15
	for _, v := range sig[band*ix.rows : (band+1)*ix.rows] {
		sum = mix64(sum ^ v)
	}
	return sum
}

// Add stores a signature and returns its id. Nil signatures get an id but
// never match anything.
func (ix *Index) Add(sig Signature) int {
	id := len(ix.sigs)
	ix.sigs = append(ix.sigs, sig)
	if len(sig) < ix.bands*ix.rows {
		return id
	}
	for b := 0; b < ix.bands; b++ {
		k := ix.bandKey(sig, b)
		ix.buckets[b][k] = append(ix.buckets[b][k], id)
	}
	return id
}

// Query returns the id of the stored signature most similar to sig, and
// that similarity, if it reaches the threshold. Otherwise it returns -1.
// Ties go to the lowest id, so results do not depend on map order.
func (ix *Index) Query(sig Signature) (id int, similarity float64) {
	if len(sig) < ix.bands*ix.rows {
		return -1, 0
	}
	best, bestSim := -1, 0.0
	seen := map[int]bool{}
	for b := 0; b < ix.bands; b++ {
		for _, cand := range ix.buckets[b][ix.bandKey(sig, b)] {
			if seen[cand] {
				continue
			}
			seen[cand] = true
			sim := Similarity(sig, ix.sigs[cand])
			if sim < ix.threshold {
				continue
			}
			if sim > bestSim || (sim == bestSim && cand < best) {
				best, bestSim = cand, sim
			}
		}
	}
	return best, bestSim
}

// Options configures near-duplicate detection.
type Options struct {
	// Threshold is the estimated Jaccard similarity at or above which a
	// text counts as a duplicate. Zero means DefaultThreshold.
	Threshold float64
	// NumHashes is the signature length. Zero means DefaultNumHashes.
	NumHashes int
	// ShingleSize is the number of words per shingle. Zero means
	// DefaultShingleSize.
	ShingleSize int
}

func (o Options) withDefaults() Options {
	if o.Threshold == 0 {
		o.Threshold = DefaultThreshold
	}
	if o.NumHashes <= 0 {
		o.NumHashes = DefaultNumHashes
	}
	if o.ShingleSize <= 0 {
		o.ShingleSize = DefaultShingleSize
	}
	return o
}

// FindDuplicates returns, for every text, the index of the earlier text it
// is a near-duplicate of, or -1 if it is the first of its kind. Duplicates
// always point at a text that is itself kept, so dropping every text with a
// non-negative entry leaves one representative of each group.
func FindDuplicates(texts []string, opt Options) ([]int, error) {
	opt = opt.withDefaults()
	ix, err := NewIndex(opt.NumHashes, opt.Threshold)
	if err != nil {
		return nil, err
	}
	hasher := NewHasher(opt.NumHashes, opt.ShingleSize)

	dupOf := make([]int, len(texts))
	kept := make([]int, 0, len(texts)) // index id -> position in texts
	for i, t := range texts {
		sig := hasher.Signature(t)
		if id, _ := ix.Query(sig); id >= 0 {
			dupOf[i] = kept[id]
			continue
		}
		dupOf[i] = -1
		ix.Add(sig)
		kept = append(kept, i)
	}
	return dupOf, nil
}
