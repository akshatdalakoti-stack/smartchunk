package eval

import (
	"reflect"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
)

// structDoc has, by line:
//
//	1  # Title
//	3  intro
//	5-9   code block
//	11 ## Options
//	13-15 table
//	17-19 list
//	21 ## Empty tail
const structDoc = "# Title\n\nintro\n\n```go\nline one\nline two\nline three\n```\n\n## Options\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n- x\n- y\n- z\n\n## Empty tail\n"

func lines(ranges ...[2]int) []chunk.Chunk {
	out := make([]chunk.Chunk, len(ranges))
	for i, r := range ranges {
		out[i] = chunk.Chunk{StartLine: r[0], EndLine: r[1]}
	}
	return out
}

func TestMeasureStructure(t *testing.T) {
	blocks, err := parse.Parse([]byte(structDoc))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	base := Structural{CodeBlocks: 1, Tables: 1, Lists: 1, Headings: 2}

	tests := []struct {
		name   string
		chunks []chunk.Chunk
		want   Structural
	}{
		{
			name:   "one chunk holds everything",
			chunks: lines([2]int{1, 21}),
			want:   base,
		},
		{
			name:   "clean section boundaries",
			chunks: lines([2]int{1, 9}, [2]int{11, 21}),
			want:   base,
		},
		{
			name:   "code block cut in half",
			chunks: lines([2]int{1, 7}, [2]int{8, 21}),
			want:   with(base, func(s *Structural) { s.CodeCut, s.NeverIntact = 1, 1 }),
		},
		{
			name:   "table cut",
			chunks: lines([2]int{1, 13}, [2]int{14, 21}),
			want:   with(base, func(s *Structural) { s.TableCut, s.NeverIntact = 1, 1 }),
		},
		{
			name:   "list cut",
			chunks: lines([2]int{1, 18}, [2]int{19, 21}),
			want:   with(base, func(s *Structural) { s.ListCut, s.NeverIntact = 1, 1 }),
		},
		{
			name:   "chunk strictly inside a block is a cut",
			chunks: lines([2]int{6, 7}),
			want:   with(base, func(s *Structural) { s.CodeCut, s.NeverIntact = 1, 1 }),
		},
		{
			name:   "overlap leaves an intact copy",
			chunks: lines([2]int{1, 7}, [2]int{5, 21}),
			want:   with(base, func(s *Structural) { s.CodeCut = 1 }),
		},
		{
			name:   "heading separated from its content",
			chunks: lines([2]int{1, 1}, [2]int{3, 21}),
			want:   with(base, func(s *Structural) { s.OrphanHeadings, s.OrphansNeverJoined = 1, 1 }),
		},
		{
			name:   "second heading at the end of a chunk",
			chunks: lines([2]int{1, 11}, [2]int{13, 21}),
			want:   with(base, func(s *Structural) { s.OrphanHeadings, s.OrphansNeverJoined = 1, 1 }),
		},
		{
			name:   "overlap reunites an orphan with its content",
			chunks: lines([2]int{1, 11}, [2]int{11, 21}),
			want:   with(base, func(s *Structural) { s.OrphanHeadings = 1 }),
		},
		{
			name:   "dropped heading is not an orphan",
			chunks: lines([2]int{3, 9}, [2]int{11, 21}),
			want:   base,
		},
		{
			name:   "no chunks",
			chunks: nil,
			want:   base,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MeasureStructure(blocks, tt.chunks)
			if got != tt.want {
				t.Errorf("MeasureStructure\n got: %+v\nwant: %+v", got, tt.want)
			}
		})
	}
}

func with(s Structural, f func(*Structural)) Structural {
	f(&s)
	return s
}

func TestStructuralAdd(t *testing.T) {
	a := Structural{CodeBlocks: 1, Tables: 2, Lists: 3, CodeCut: 1, TableCut: 1, ListCut: 1, NeverIntact: 2, Headings: 4, OrphanHeadings: 2, OrphansNeverJoined: 1}
	sum := a
	sum.Add(a)
	want := Structural{CodeBlocks: 2, Tables: 4, Lists: 6, CodeCut: 2, TableCut: 2, ListCut: 2, NeverIntact: 4, Headings: 8, OrphanHeadings: 4, OrphansNeverJoined: 2}
	if sum != want {
		t.Errorf("Add = %+v, want %+v", sum, want)
	}
	if sum.Violations() != 6 || sum.AtomicBlocks() != 12 {
		t.Errorf("Violations = %d, AtomicBlocks = %d", sum.Violations(), sum.AtomicBlocks())
	}
}

func TestCutBy(t *testing.T) {
	blocks, err := parse.Parse([]byte(structDoc))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	tests := []struct {
		name  string
		chunk chunk.Chunk
		want  []parse.BlockType
	}{
		{"clean", chunk.Chunk{StartLine: 1, EndLine: 9}, nil},
		{"cuts code", chunk.Chunk{StartLine: 1, EndLine: 6}, []parse.BlockType{parse.Code}},
		{"cuts table and list", chunk.Chunk{StartLine: 14, EndLine: 17}, []parse.BlockType{parse.Table, parse.List}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CutBy(blocks, tt.chunk); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CutBy = %v, want %v", got, tt.want)
			}
		})
	}
}
