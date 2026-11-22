package semantic

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/split"
)

// angleEmbedder embeds a text as a unit vector at the angle (in degrees)
// written after "@" in the text, so tests can dictate every similarity:
// the cosine of two texts is the cosine of the difference of their angles.
type angleEmbedder struct {
	calls int
	err   error
	short bool
}

func (a *angleEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	a.calls++
	if a.err != nil {
		return nil, a.err
	}
	out := make([][]float32, 0, len(texts))
	for _, t := range texts {
		deg := 0.0
		if i := strings.LastIndex(t, "@"); i >= 0 {
			for _, c := range t[i+1:] {
				if c < '0' || c > '9' {
					break
				}
				deg = deg*10 + float64(c-'0')
			}
		}
		rad := deg * math.Pi / 180
		out = append(out, []float32{float32(math.Cos(rad)), float32(math.Sin(rad))})
	}
	if a.short && len(out) > 0 {
		out = out[:len(out)-1]
	}
	return out, nil
}

func structure(t *testing.T, src string) []split.Candidate {
	t.Helper()
	blocks, err := parse.Parse([]byte(src))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return split.Structure(blocks)
}

// shape describes a candidate by the first line of each block and the kind
// of boundary in front of it.
type shape struct {
	Lines []string
	Soft  bool
}

func shapes(cands []split.Candidate) []shape {
	out := make([]shape, len(cands))
	for i, c := range cands {
		for _, b := range c.Blocks {
			out[i].Lines = append(out[i].Lines, strings.SplitN(b.Text, "\n", 2)[0])
		}
		out[i].Soft = c.Soft
	}
	return out
}

func TestRefine(t *testing.T) {
	tests := []struct {
		name string
		src  string
		opt  Options
		want []shape
	}{
		{
			name: "topic shift inside a section is cut",
			// Angles 0,2,4 | 80,82,84: one large drop among small ones.
			src: "# H\n\na @0\n\nb @2\n\nc @4\n\nd @80\n\ne @82\n\nf @84\n",
			want: []shape{
				{[]string{"# H", "a @0", "b @2", "c @4"}, false},
				{[]string{"d @80", "e @82", "f @84"}, true},
			},
		},
		{
			name: "uniform similarity is never cut",
			src:  "# H\n\na @0\n\nb @10\n\nc @20\n\nd @30\n\ne @40\n\nf @50\n",
			want: []shape{
				{[]string{"# H", "a @0", "b @10", "c @20", "d @30", "e @40", "f @50"}, false},
			},
		},
		{
			name: "sections stay separate and hard",
			src:  "# A\n\na @0\n\nb @1\n\n# B\n\nc @2\n\nd @3\n\ne @4\n\nf @5\n",
			want: []shape{
				{[]string{"# A", "a @0", "b @1"}, false},
				{[]string{"# B", "c @2", "d @3", "e @4", "f @5"}, false},
			},
		},
		{
			name: "too few pairs leaves the document alone",
			src:  "# H\n\na @0\n\nb @90\n\nc @0\n",
			want: []shape{
				{[]string{"# H", "a @0", "b @90", "c @0"}, false},
			},
		},
		{
			name: "code travels with the prose before it",
			src:  "# H\n\na @0\n\n```\ncode\n```\n\nb @1\n\nc @2\n\nd @90\n\n- item\n\ne @91\n\nf @92\n",
			want: []shape{
				{[]string{"# H", "a @0", "```", "b @1", "c @2"}, false},
				{[]string{"d @90", "- item", "e @91", "f @92"}, true},
			},
		},
		{
			name: "higher percentile cuts more",
			// Drops of 40, 2, 30, 2, 20 degrees: the 60th percentile catches
			// the three large ones.
			src: "# H\n\na @0\n\nb @40\n\nc @42\n\nd @72\n\ne @74\n\nf @94\n",
			opt: Options{Percentile: 60},
			want: []shape{
				{[]string{"# H", "a @0"}, false},
				{[]string{"b @40", "c @42"}, true},
				{[]string{"d @72", "e @74"}, true},
				{[]string{"f @94"}, true},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cands := structure(t, tt.src)
			got, err := Refine(context.Background(), cands, &angleEmbedder{}, tt.opt)
			if err != nil {
				t.Fatalf("Refine: %v", err)
			}
			if !reflect.DeepEqual(shapes(got), tt.want) {
				t.Errorf("candidates\n got: %+v\nwant: %+v", shapes(got), tt.want)
			}
			// No block may be lost, duplicated or reordered.
			var before, after []parse.Block
			for _, c := range cands {
				before = append(before, c.Blocks...)
			}
			for _, c := range got {
				after = append(after, c.Blocks...)
			}
			if !reflect.DeepEqual(before, after) {
				t.Error("Refine changed the sequence of blocks")
			}
		})
	}
}
