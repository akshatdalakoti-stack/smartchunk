package eval

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
)

// Question is one entry of the questions file: a question and the lines of
// the source document that answer it.
type Question struct {
	Question        string `json:"question"`
	SourceFile      string `json:"source_file"`
	AnswerStartLine int    `json:"answer_start_line"`
	AnswerEndLine   int    `json:"answer_end_line"`
}

// LoadQuestions reads a JSONL questions file and validates every entry.
func LoadQuestions(path string) ([]Question, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("eval: opening questions: %w", err)
	}
	defer f.Close()

	var out []Question
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64<<10), 4<<20)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var q Question
		if err := json.Unmarshal([]byte(raw), &q); err != nil {
			return nil, fmt.Errorf("eval: %s line %d: %w", path, line, err)
		}
		switch {
		case strings.TrimSpace(q.Question) == "":
			return nil, fmt.Errorf("eval: %s line %d: question is empty", path, line)
		case q.SourceFile == "":
			return nil, fmt.Errorf("eval: %s line %d: source_file is empty", path, line)
		case q.AnswerStartLine < 1 || q.AnswerEndLine < q.AnswerStartLine:
			return nil, fmt.Errorf("eval: %s line %d: answer lines %d-%d are not a valid range", path, line, q.AnswerStartLine, q.AnswerEndLine)
		}
		q.SourceFile = strings.ReplaceAll(q.SourceFile, `\`, "/")
		out = append(out, q)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("eval: reading %s: %w", path, err)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("eval: %s contains no questions", path)
	}
	return out, nil
}

// Retrieval holds retrieval quality averaged over all questions.
type Retrieval struct {
	Questions int
	// Hit1, Hit3 and Hit5 are the share of questions for which a relevant
	// chunk is among the top 1, 3 and 5 results.
	Hit1, Hit3, Hit5 float64
	// MRR is the mean reciprocal rank of the first relevant chunk, over
	// the full ranking.
	MRR float64
	// Complete5 is the share of questions for which one of the top 5
	// chunks holds the whole answer span, not just part of it.
	Complete5 float64
}

// QuestionResult is the outcome for one question.
type QuestionResult struct {
	Question Question
	// Rank is the 1-based position of the first relevant chunk, or 0 when
	// no chunk overlaps the answer at all.
	Rank int
	// Complete5 reports whether a top-5 chunk holds the whole answer.
	Complete5 bool
}

// Relevant reports whether a chunk comes from the question's document and
// overlaps its answer lines.
func Relevant(c chunk.Chunk, q Question) bool {
	return c.SourceFile == q.SourceFile && c.StartLine <= q.AnswerEndLine && c.EndLine >= q.AnswerStartLine
}

// Complete reports whether a chunk holds every answer line of the question.
func Complete(c chunk.Chunk, q Question) bool {
	return c.SourceFile == q.SourceFile && c.StartLine <= q.AnswerStartLine && c.EndLine >= q.AnswerEndLine
}

// MeasureRetrieval embeds the chunks and the questions, ranks the chunks for
// every question by cosine similarity, and scores the rankings. textOf
// chooses what is embedded for a chunk.
func MeasureRetrieval(ctx context.Context, chunks []chunk.Chunk, questions []Question, emb embed.Embedder, textOf func(chunk.Chunk) string) (Retrieval, []QuestionResult, error) {
	if emb == nil {
		return Retrieval{}, nil, fmt.Errorf("eval: retrieval needs an embedder")
	}
	if len(questions) == 0 {
		return Retrieval{}, nil, nil
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = textOf(c)
	}
	chunkVecs, err := emb.Embed(ctx, texts)
	if err != nil {
		return Retrieval{}, nil, fmt.Errorf("eval: embedding %d chunks: %w", len(chunks), err)
	}
	queries := make([]string, len(questions))
	for i, q := range questions {
		queries[i] = q.Question
	}
	queryVecs, err := emb.Embed(ctx, queries)
	if err != nil {
		return Retrieval{}, nil, fmt.Errorf("eval: embedding %d questions: %w", len(questions), err)
	}
	if len(chunkVecs) != len(chunks) || len(queryVecs) != len(questions) {
		return Retrieval{}, nil, fmt.Errorf("eval: embedder returned %d and %d vectors for %d chunks and %d questions",
			len(chunkVecs), len(queryVecs), len(chunks), len(questions))
	}

	index := NewIndex(chunkVecs)
	results := make([]QuestionResult, len(questions))
	r := Retrieval{Questions: len(questions)}
	for i, q := range questions {
		res := QuestionResult{Question: q}
		for pos, hit := range index.Search(queryVecs[i], 0) {
			c := chunks[hit.ID]
			if pos < 5 && Complete(c, q) {
				res.Complete5 = true
			}
			if res.Rank == 0 && Relevant(c, q) {
				res.Rank = pos + 1
			}
			if res.Rank != 0 && pos >= 4 {
				break
			}
		}
		results[i] = res
		if res.Rank > 0 {
			r.MRR += 1 / float64(res.Rank)
			if res.Rank <= 1 {
				r.Hit1++
			}
			if res.Rank <= 3 {
				r.Hit3++
			}
			if res.Rank <= 5 {
				r.Hit5++
			}
		}
		if res.Complete5 {
			r.Complete5++
		}
	}
	n := float64(len(questions))
	r.Hit1 /= n
	r.Hit3 /= n
	r.Hit5 /= n
	r.MRR /= n
	r.Complete5 /= n
	return r, results, nil
}
