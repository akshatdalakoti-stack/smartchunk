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
