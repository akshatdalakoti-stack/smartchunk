package eval

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/akshatdalakoti-stack/smartchunk/internal/baseline"
	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/pipeline"
)

func ids(hits []Hit) []int {
	out := make([]int, len(hits))
	for i, h := range hits {
		out[i] = h.ID
	}
	return out
}

func TestIndexSearch(t *testing.T) {
	// Unit vectors at 0, 30, 60, 90 and 180 degrees, plus a zero vector.
	deg := func(d float64) []float32 {
		r := d * math.Pi / 180
		return []float32{float32(math.Cos(r)), float32(math.Sin(r))}
	}
	ix := NewIndex([][]float32{deg(90), deg(0), deg(180), deg(30), {0, 0}, deg(60)})

	tests := []struct {
		name  string
		query []float32
		k     int
		want  []int
	}{
		{"nearest first", deg(0), 3, []int{1, 3, 5}},
		{"top one", deg(85), 1, []int{0}},
		{"query length is irrelevant", []float32{50, 0}, 2, []int{1, 3}},
		{"k of zero returns everything", deg(0), 0, []int{1, 3, 5, 0, 4, 2}},
		{"k above size returns everything", deg(0), 99, []int{1, 3, 5, 0, 4, 2}},
		{"ties go to the lower id", deg(45), 2, []int{3, 5}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ix.Search(tt.query, tt.k)
			if !reflect.DeepEqual(ids(got), tt.want) {
				t.Errorf("Search = %v, want %v", ids(got), tt.want)
			}
			for i := 1; i < len(got); i++ {
				if got[i].Score > got[i-1].Score {
					t.Errorf("scores are not descending: %v", got)
				}
			}
		})
	}
	if ix.Len() != 6 {
		t.Errorf("Len = %d", ix.Len())
	}
	if got := NewIndex(nil).Search(deg(0), 5); len(got) != 0 {
		t.Errorf("empty index returned %v", got)
	}
}

// TestSearchMatchesFullSort checks the heap-based top-k against a full
// ranking on random data.
func TestSearchMatchesFullSort(t *testing.T) {
	vecs, _ := embed.Fake{Dim: 16}.Embed(context.Background(), manyTexts(300))
	ix := NewIndex(vecs)
	for q := 0; q < 20; q++ {
		full := ix.Search(vecs[q*7], 0)
		for _, k := range []int{1, 5, 17} {
			top := ix.Search(vecs[q*7], k)
			if !reflect.DeepEqual(top, full[:k]) {
				t.Fatalf("query %d: top-%d differs from the head of the full ranking", q, k)
			}
		}
	}
}

func manyTexts(n int) []string {
	out := make([]string, n)
	for i := range out {
		out[i] = fmt.Sprintf("alpha%d beta%d gamma%d shared words here", i%17, i%5, i)
	}
	return out
}

func TestRelevantAndComplete(t *testing.T) {
	q := Question{SourceFile: "a.md", AnswerStartLine: 10, AnswerEndLine: 14}
	tests := []struct {
		name         string
		c            chunk.Chunk
		wantRelevant bool
		wantComplete bool
	}{
		{"covers the answer", chunk.Chunk{SourceFile: "a.md", StartLine: 5, EndLine: 20}, true, true},
		{"exactly the answer", chunk.Chunk{SourceFile: "a.md", StartLine: 10, EndLine: 14}, true, true},
		{"first half", chunk.Chunk{SourceFile: "a.md", StartLine: 5, EndLine: 12}, true, false},
		{"touches the last line", chunk.Chunk{SourceFile: "a.md", StartLine: 14, EndLine: 30}, true, false},
		{"inside the answer", chunk.Chunk{SourceFile: "a.md", StartLine: 11, EndLine: 12}, true, false},
		{"just before", chunk.Chunk{SourceFile: "a.md", StartLine: 1, EndLine: 9}, false, false},
		{"just after", chunk.Chunk{SourceFile: "a.md", StartLine: 15, EndLine: 20}, false, false},
		{"other file", chunk.Chunk{SourceFile: "b.md", StartLine: 5, EndLine: 20}, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Relevant(tt.c, q); got != tt.wantRelevant {
				t.Errorf("Relevant = %v, want %v", got, tt.wantRelevant)
			}
			if got := Complete(tt.c, q); got != tt.wantComplete {
				t.Errorf("Complete = %v, want %v", got, tt.wantComplete)
			}
		})
	}
}

// tableEmbedder returns a fixed vector for each known text.
type tableEmbedder map[string][]float32

func (e tableEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, ok := e[t]
		if !ok {
			return nil, fmt.Errorf("no vector for %q", t)
		}
		out[i] = v
	}
	return out, nil
}

func TestMeasureRetrieval(t *testing.T) {
	// Four chunks along the unit circle; questions aimed at known angles.
	deg := func(d float64) []float32 {
		r := d * math.Pi / 180
		return []float32{float32(math.Cos(r)), float32(math.Sin(r))}
	}
	chunks := []chunk.Chunk{
		{Text: "c0", SourceFile: "a.md", StartLine: 1, EndLine: 10},
		{Text: "c1", SourceFile: "a.md", StartLine: 11, EndLine: 20},
		{Text: "c2", SourceFile: "a.md", StartLine: 21, EndLine: 30},
		{Text: "c3", SourceFile: "b.md", StartLine: 1, EndLine: 10},
		{Text: "c4", SourceFile: "b.md", StartLine: 11, EndLine: 20},
		{Text: "c5", SourceFile: "b.md", StartLine: 21, EndLine: 30},
	}
	emb := tableEmbedder{
		"c0": deg(0), "c1": deg(20), "c2": deg(40), "c3": deg(60), "c4": deg(80), "c5": deg(100),
		"q-first":   deg(1),  // nearest c0
		"q-second":  deg(1),  // nearest c0, answer in c1 -> rank 2
		"q-sixth":   deg(0),  // answer in c5 -> rank 6
		"q-missing": deg(0),  // answer lines no chunk covers
		"q-partial": deg(21), // nearest c1, answer spans c1 and c2
	}
	questions := []Question{
		{Question: "q-first", SourceFile: "a.md", AnswerStartLine: 2, AnswerEndLine: 4},
		{Question: "q-second", SourceFile: "a.md", AnswerStartLine: 12, AnswerEndLine: 13},
		{Question: "q-sixth", SourceFile: "b.md", AnswerStartLine: 25, AnswerEndLine: 26},
		{Question: "q-missing", SourceFile: "a.md", AnswerStartLine: 40, AnswerEndLine: 41},
		{Question: "q-partial", SourceFile: "a.md", AnswerStartLine: 18, AnswerEndLine: 23},
	}
	textOf := func(c chunk.Chunk) string { return c.Text }

	got, per, err := MeasureRetrieval(context.Background(), chunks, questions, emb, textOf)
	if err != nil {
		t.Fatalf("MeasureRetrieval: %v", err)
	}
	wantRanks := []int{1, 2, 6, 0, 1}
	wantComplete := []bool{true, true, false, false, false}
	for i, r := range per {
		if r.Rank != wantRanks[i] || r.Complete5 != wantComplete[i] {
			t.Errorf("question %q: rank %d complete %v, want rank %d complete %v",
				questions[i].Question, r.Rank, r.Complete5, wantRanks[i], wantComplete[i])
		}
	}
	want := Retrieval{
		Questions: 5,
		Hit1:      2.0 / 5,
		Hit3:      3.0 / 5,
		Hit5:      3.0 / 5,
		MRR:       (1 + 0.5 + 1.0/6 + 0 + 1) / 5,
		Complete5: 2.0 / 5,
	}
	if math.Abs(got.Hit1-want.Hit1) > 1e-9 || math.Abs(got.Hit3-want.Hit3) > 1e-9 || math.Abs(got.Hit5-want.Hit5) > 1e-9 ||
		math.Abs(got.MRR-want.MRR) > 1e-9 || math.Abs(got.Complete5-want.Complete5) > 1e-9 || got.Questions != want.Questions {
		t.Errorf("Retrieval\n got: %+v\nwant: %+v", got, want)
	}

	t.Run("errors", func(t *testing.T) {
		if _, _, err := MeasureRetrieval(context.Background(), chunks, questions, nil, textOf); err == nil {
			t.Error("nil embedder should be rejected")
		}
		unknown := []Question{{Question: "never seen", SourceFile: "a.md", AnswerStartLine: 1, AnswerEndLine: 1}}
		if _, _, err := MeasureRetrieval(context.Background(), chunks, unknown, emb, textOf); err == nil {
			t.Error("embedder failure should be reported")
		}
		r, per, err := MeasureRetrieval(context.Background(), chunks, nil, emb, textOf)
		if err != nil || r.Questions != 0 || per != nil {
			t.Errorf("no questions: %+v %v %v", r, per, err)
		}
	})
}

func TestMeasureTokens(t *testing.T) {
	mk := func(counts ...int) []chunk.Chunk {
		out := make([]chunk.Chunk, len(counts))
		for i, n := range counts {
			out[i] = chunk.Chunk{TokenCount: n}
		}
		return out
	}
	hundred := make([]int, 100)
	for i := range hundred {
		hundred[i] = i + 1
	}
	tests := []struct {
		name   string
		chunks []chunk.Chunk
		want   TokenStats
	}{
		{"none", nil, TokenStats{}},
		{"one", mk(42), TokenStats{Mean: 42, P95: 42, Max: 42}},
		{"unsorted input", mk(30, 10, 20), TokenStats{Mean: 20, P95: 30, Max: 30}},
		{"one to a hundred", mk(hundred...), TokenStats{Mean: 50.5, P95: 95, Max: 100}},
		{"outlier sets max, not p95", mk(append(make([]int, 99), 1000)...), TokenStats{Mean: 10, P95: 0, Max: 1000}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MeasureTokens(tt.chunks); got != tt.want {
				t.Errorf("MeasureTokens = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestMeasureDuplicates(t *testing.T) {
	long := func(seed string) string {
		var sb strings.Builder
		for i := 0; i < 60; i++ {
			fmt.Fprintf(&sb, "%s%d ", seed, i)
		}
		return sb.String()
	}
	tests := []struct {
		name      string
		texts     []string
		threshold float64
		wantCount int
		wantErr   bool
	}{
		{"none", nil, 0.85, 0, false},
		{"all distinct", []string{long("a"), long("b"), long("c")}, 0.85, 0, false},
		{"two copies of one", []string{long("a"), long("b"), long("a"), long("a")}, 0.85, 2, false},
		{"bad threshold", []string{long("a")}, 2, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks := make([]chunk.Chunk, len(tt.texts))
			for i, text := range tt.texts {
				chunks[i] = chunk.Chunk{Text: text}
			}
			count, rate, err := MeasureDuplicates(chunks, tt.threshold)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				return
			}
			if count != tt.wantCount {
				t.Errorf("count = %d, want %d", count, tt.wantCount)
			}
			if len(chunks) > 0 && math.Abs(rate-float64(tt.wantCount)/float64(len(chunks))) > 1e-9 {
				t.Errorf("rate = %v", rate)
			}
		})
	}
}

func TestLoadQuestions(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    int
		wantErr string
	}{
		{
			name:    "valid with blank lines",
			content: "{\"question\":\"q1\",\"source_file\":\"a.md\",\"answer_start_line\":1,\"answer_end_line\":3}\n\n{\"question\":\"q2\",\"source_file\":\"sub\\\\b.md\",\"answer_start_line\":5,\"answer_end_line\":5}\n",
			want:    2,
		},
		{"empty file", "", 0, "no questions"},
		{"bad json", "{nope}\n", 0, "line 1"},
		{"missing question", "{\"source_file\":\"a.md\",\"answer_start_line\":1,\"answer_end_line\":1}\n", 0, "question is empty"},
		{"missing file", "{\"question\":\"q\",\"answer_start_line\":1,\"answer_end_line\":1}\n", 0, "source_file is empty"},
		{"zero start", "{\"question\":\"q\",\"source_file\":\"a.md\",\"answer_start_line\":0,\"answer_end_line\":1}\n", 0, "not a valid range"},
		{"end before start", "{\"question\":\"q\",\"source_file\":\"a.md\",\"answer_start_line\":5,\"answer_end_line\":4}\n", 0, "not a valid range"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "q.jsonl")
			if err := os.WriteFile(path, []byte(tt.content), 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := LoadQuestions(path)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("err = %v, want it to mention %q", err, tt.wantErr)
				}
				return
			}
			if err != nil || len(got) != tt.want {
				t.Fatalf("got %d questions, err %v", len(got), err)
			}
			if got[1].SourceFile != "sub/b.md" {
				t.Errorf("backslashes were not normalised: %q", got[1].SourceFile)
			}
		})
	}
	if _, err := LoadQuestions(filepath.Join(t.TempDir(), "missing.jsonl")); err == nil {
		t.Error("missing file should be an error")
	}
}
