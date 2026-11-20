// Package baseline is the naive splitter SmartChunk is measured against:
// fixed windows of N tokens with a fixed overlap, blind to document
// structure.
package baseline

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
)

// Options configures the fixed-size splitter.
type Options struct {
	// Size is the window length in tokens.
	Size int
	// Overlap is how many tokens consecutive windows share.
	Overlap int
	// Tokenizer defaults to chunk.ApproxTokenizer.
	Tokenizer chunk.Tokenizer
}

// DefaultOptions returns the settings used when the CLI is given no flags.
func DefaultOptions() Options {
	return Options{Size: 256, Overlap: 32, Tokenizer: chunk.ApproxTokenizer{}}
}

// Validate reports whether the options describe a splitter that terminates.
func (o Options) Validate() error {
	if o.Size <= 0 {
		return fmt.Errorf("baseline: size must be positive, got %d", o.Size)
	}
	if o.Overlap < 0 {
		return fmt.Errorf("baseline: overlap must not be negative, got %d", o.Overlap)
	}
	if o.Overlap >= o.Size {
		return fmt.Errorf("baseline: overlap (%d) must be smaller than size (%d)", o.Overlap, o.Size)
	}
	return nil
}

// Split cuts source into fixed-size token windows. Every chunk is a verbatim
// substring of the source, reported as prose because the splitter does not
// know any better.
func Split(sourceFile string, source []byte, opt Options) ([]chunk.Chunk, error) {
	if err := opt.Validate(); err != nil {
		return nil, err
	}
	if sourceFile == "" {
		return nil, errors.New("baseline: source file name is empty")
	}
	tok := opt.Tokenizer
	if tok == nil {
		tok = chunk.ApproxTokenizer{}
	}

	text := strings.ReplaceAll(string(source), "\r\n", "\n")
	spans := tok.Spans(text)
	if len(spans) == 0 {
		return nil, nil
	}

	// Offsets of every newline, to turn byte offsets into line numbers.
	var newlines []int
	for i := 0; i < len(text); i++ {
		if text[i] == '\n' {
			newlines = append(newlines, i)
		}
	}
	lineOf := func(off int) int {
		return sort.SearchInts(newlines, off) + 1
	}

	step := opt.Size - opt.Overlap
	var chunks []chunk.Chunk
	for start := 0; start < len(spans); start += step {
		end := start + opt.Size
		if end > len(spans) {
			end = len(spans)
		}
		from, to := spans[start].Start, spans[end-1].End
		body := text[from:to]
		startLine, endLine := lineOf(from), lineOf(to-1)
		chunks = append(chunks, chunk.Chunk{
			ID:          chunk.NewID(sourceFile, startLine, endLine, body),
			Text:        body,
			ChunkType:   chunk.Prose,
			HeadingPath: []string{},
			SourceFile:  sourceFile,
			StartLine:   startLine,
			EndLine:     endLine,
			TokenCount:  end - start,
			Flags:       []string{},
		})
		if end == len(spans) {
			break
		}
	}
	return chunks, nil
}
