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

func TestChooseBands(t *testing.T) {
	tests := []struct {
		hashes    int
		threshold float64
		wantRows  int
	}{
		{128, 0.85, 8},
		{128, 0.95, 8},
		{128, 1.0, 16},
		{128, 0.5, 2},
		{128, 0.2, 1},
		{64, 0.85, 4},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%d@%.2f", tt.hashes, tt.threshold), func(t *testing.T) {
			bands, rows := chooseBands(tt.hashes, tt.threshold)
			if rows != tt.wantRows || bands*rows != tt.hashes {
				t.Errorf("chooseBands = %d bands x %d rows, want %d rows", bands, rows, tt.wantRows)
			}
			// A pair right at the threshold must almost always be a candidate.
			p := 1 - math.Pow(1-math.Pow(tt.threshold, float64(rows)), float64(bands))
			if p < 0.95 {
				t.Errorf("recall at the threshold is only %.3f", p)
			}
		})
	}
}

func TestNewIndexValidation(t *testing.T) {
	tests := []struct {
		name      string
		hashes    int
		threshold float64
		wantErr   bool
	}{
		{"ok", 128, 0.85, false},
		{"threshold one", 128, 1, false},
		{"zero hashes", 0, 0.85, true},
		{"zero threshold", 128, 0, true},
		{"negative threshold", 128, -0.1, true},
		{"threshold above one", 128, 1.01, true},
		{"nan threshold", 128, math.NaN(), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewIndex(tt.hashes, tt.threshold)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewIndex error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestIndexQuery(t *testing.T) {
	h := NewHasher(128, 3)
	ix, err := NewIndex(128, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	base := strings.Join(vocabText(7, 300), " ")
	other := strings.Join(vocabText(8, 300), " ")
	ix.Add(h.Signature(other))
	ix.Add(h.Signature(base))
	ix.Add(nil)

	tests := []struct {
		name   string
		text   string
		wantID int
	}{
		{"exact copy", base, 1},
		{"copy with a few extra words", base + " and a short tail", 1},
		{"unrelated", strings.Join(vocabText(9, 300), " "), -1},
		{"no words", "!!!", -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			id, sim := ix.Query(h.Signature(tt.text))
			if id != tt.wantID {
				t.Errorf("Query = id %d (sim %.2f), want id %d", id, sim, tt.wantID)
			}
			if id >= 0 && sim < 0.8 {
				t.Errorf("returned similarity %.2f is below the threshold", sim)
			}
		})
	}
}

func TestFindDuplicates(t *testing.T) {
	a := strings.Join(vocabText(1, 200), " ")
	b := strings.Join(vocabText(2, 200), " ")
	c := strings.Join(vocabText(3, 200), " ")
	aEdited := strings.Replace(a, "word", "term", 3)    // 3 of 200 words differ
	aHalf := strings.Join(strings.Fields(a)[:100], " ") // Jaccard about 0.5

	tests := []struct {
		name  string
		texts []string
		opt   Options
		want  []int
	}{
		{"empty", nil, Options{}, []int{}},
		{"all different", []string{a, b, c}, Options{}, []int{-1, -1, -1}},
		{"exact duplicate", []string{a, b, a}, Options{}, []int{-1, -1, 0}},
		{"near duplicate", []string{a, aEdited}, Options{}, []int{-1, 0}},
		{"reformatted duplicate", []string{a, strings.ReplaceAll(strings.ToUpper(a), " ", "\n")}, Options{}, []int{-1, 0}},
		{"half overlap is kept at 0.85", []string{a, aHalf}, Options{}, []int{-1, -1}},
		{"half overlap is dropped at 0.4", []string{a, aHalf}, Options{Threshold: 0.4}, []int{-1, 0}},
		{"chain points at the kept original", []string{a, a, a}, Options{}, []int{-1, 0, 0}},
		{"empty texts are never duplicates", []string{"", "", "..."}, Options{}, []int{-1, -1, -1}},
		{"short identical texts", []string{"ok", "ok"}, Options{}, []int{-1, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := FindDuplicates(tt.texts, tt.opt)
			if err != nil {
				t.Fatalf("FindDuplicates: %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("FindDuplicates = %v, want %v", got, tt.want)
			}
		})
	}

	if _, err := FindDuplicates([]string{a}, Options{Threshold: 1.5}); err == nil {
		t.Error("threshold above 1 should be rejected")
	}
}

func para(text string, start int, typ parse.BlockType) parse.Block {
	return parse.Block{Type: typ, Text: text, StartLine: start, EndLine: start + strings.Count(text, "\n")}
}

func TestRepeatedLines(t *testing.T) {
	footer := "Back to the documentation index"
	var cands []split.Candidate
	for i := 0; i < 10; i++ {
		body := fmt.Sprintf("Unique body text number %d.\n%s", i, footer)
		cands = append(cands, split.Candidate{Blocks: []parse.Block{
			para(body, 1, parse.Paragraph),
			para("```\nshared code line here\n```", 4, parse.Code),
		}})
	}
	// A line that appears in only two chunks.
	cands[0].Blocks = append(cands[0].Blocks, para("See the related section.", 8, parse.Paragraph))
	cands[1].Blocks = append(cands[1].Blocks, para("See the related section.", 8, parse.Paragraph))

	tests := []struct {
		name string
		opt  LineOptions
		want []string
	}{
		{"defaults catch the footer only", LineOptions{}, []string{"back to the documentation index"}},
		{"low bar catches the rarer line too", LineOptions{MinRepeats: 2, MinRatio: 0.1}, []string{"back to the documentation index", "see the related section."}},
		{"ratio can veto", LineOptions{MinRepeats: 2, MinRatio: 1.1}, nil},
		{"length filter", LineOptions{MinLength: 100}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := RepeatedLines(cands, tt.opt)
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for _, w := range tt.want {
				if !got[w] {
					t.Errorf("missing %q in %v", w, got)
				}
			}
		})
	}
}

func TestStripLines(t *testing.T) {
	lines := map[string]bool{"edit this page": true, "copyright acme": true}
	tests := []struct {
		name       string
		blocks     []parse.Block
		wantTexts  []string
		wantLines  [][2]int
		wantFlag   bool
		wantRemove int
		wantGone   bool
	}{
		{
			name:      "untouched",
			blocks:    []parse.Block{para("Plain text.\nMore text.", 5, parse.Paragraph)},
			wantTexts: []string{"Plain text.\nMore text."},
			wantLines: [][2]int{{5, 6}},
		},
		{
			name:       "trailing line removed shrinks the range",
			blocks:     []parse.Block{para("Real content.\nEdit this page", 5, parse.Paragraph)},
			wantTexts:  []string{"Real content."},
			wantLines:  [][2]int{{5, 5}},
			wantFlag:   true,
			wantRemove: 1,
		},
		{
			name:       "leading line removed shifts the start",
			blocks:     []parse.Block{para("  Copyright   ACME\nReal content.\nMore.", 5, parse.Paragraph)},
			wantTexts:  []string{"Real content.\nMore."},
			wantLines:  [][2]int{{6, 7}},
			wantFlag:   true,
			wantRemove: 1,
		},
		{
			name: "emptied block is dropped, rest stays",
			blocks: []parse.Block{
				para("# H", 1, parse.Heading),
				para("Keep me.", 3, parse.Paragraph),
				para("Edit this page", 5, parse.Paragraph),
			},
			wantTexts:  []string{"# H", "Keep me."},
			wantLines:  [][2]int{{1, 1}, {3, 3}},
			wantFlag:   true,
			wantRemove: 1,
		},
		{
			name: "code is never stripped",
			blocks: []parse.Block{
				para("Text.", 1, parse.Paragraph),
				para("```\nEdit this page\n```", 3, parse.Code),
			},
			wantTexts: []string{"Text.", "```\nEdit this page\n```"},
			wantLines: [][2]int{{1, 1}, {3, 5}},
		},
		{
			name: "chunk left with only a heading is dropped",
			blocks: []parse.Block{
				para("# Footer", 1, parse.Heading),
				para("Edit this page\nCopyright ACME", 3, parse.Paragraph),
			},
			wantRemove: 2,
			wantGone:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, removed := StripLines([]split.Candidate{{Blocks: tt.blocks}}, lines)
			if removed != tt.wantRemove {
				t.Errorf("removed %d lines, want %d", removed, tt.wantRemove)
			}
			if tt.wantGone {
				if len(got) != 0 {
					t.Errorf("candidate should be dropped, got %+v", got)
				}
				return
			}
			if len(got) != 1 {
				t.Fatalf("got %d candidates, want 1", len(got))
			}
			var texts []string
			var ranges [][2]int
			for _, b := range got[0].Blocks {
				texts = append(texts, b.Text)
				ranges = append(ranges, [2]int{b.StartLine, b.EndLine})
			}
			if !reflect.DeepEqual(texts, tt.wantTexts) {
				t.Errorf("texts = %q, want %q", texts, tt.wantTexts)
			}
			if !reflect.DeepEqual(ranges, tt.wantLines) {
				t.Errorf("line ranges = %v, want %v", ranges, tt.wantLines)
			}
			hasFlag := len(got[0].Flags) == 1 && got[0].Flags[0] == chunk.FlagBoilerplateStripped
			if hasFlag != tt.wantFlag {
				t.Errorf("flags = %v, want flag %v", got[0].Flags, tt.wantFlag)
			}
		})
	}

	t.Run("no lines is a no-op", func(t *testing.T) {
		in := []split.Candidate{{Blocks: []parse.Block{para("x", 1, parse.Paragraph)}}}
		got, removed := StripLines(in, nil)
		if removed != 0 || !reflect.DeepEqual(got, in) {
			t.Errorf("got %+v, removed %d", got, removed)
		}
	})
}

func BenchmarkSignature(b *testing.B) {
	h := NewHasher(DefaultNumHashes, DefaultShingleSize)
	text := strings.Join(vocabText(1, 300), " ")
	b.SetBytes(int64(len(text)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Signature(text)
	}
}

func BenchmarkFindDuplicates(b *testing.B) {
	// 2000 chunks of 200 words, a tenth of them near-duplicates.
	texts := make([]string, 2000)
	for i := range texts {
		seed := int64(i)
		if i%10 == 9 {
			seed = int64(i - 1)
		}
		texts[i] = strings.Join(vocabText(seed, 200), " ")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dup, err := FindDuplicates(texts, Options{})
		if err != nil {
			b.Fatal(err)
		}
		if i == 0 {
			n := 0
			for _, d := range dup {
				if d >= 0 {
					n++
				}
			}
			if n != 200 {
				b.Fatalf("found %d duplicates, want 200", n)
			}
		}
	}
}
