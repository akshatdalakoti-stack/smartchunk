// Package sizing brings candidate chunks into a token range: it splits the
// ones that are too large and merges the ones that are too small.
//
// Splitting follows a strict order of preference. A candidate is cut between
// units first (a paragraph and the code that follows it stay together),
// then between blocks, and only then inside a prose block, at sentence
// boundaries. Code blocks, tables and lists are never cut: one that is over
// the limit on its own stays whole and is flagged "oversized".
package sizing

import (
	"fmt"
	"math"
	"strings"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/split"
)

// Options sets the token range.
type Options struct {
	// MaxTokens is the upper limit for a chunk.
	MaxTokens int
	// MinTokens is the size below which a chunk is merged into a neighbour.
	MinTokens int
	// Tokenizer defaults to chunk.ApproxTokenizer.
	Tokenizer chunk.Tokenizer
}

// DefaultOptions returns the range used when the CLI is given no flags.
func DefaultOptions() Options {
	return Options{MaxTokens: 256, MinTokens: 64, Tokenizer: chunk.ApproxTokenizer{}}
}

// Validate reports whether the range is usable.
func (o Options) Validate() error {
	if o.MaxTokens <= 0 {
		return fmt.Errorf("sizing: max tokens must be positive, got %d", o.MaxTokens)
	}
	if o.MinTokens < 0 {
		return fmt.Errorf("sizing: min tokens must not be negative, got %d", o.MinTokens)
	}
	if o.MinTokens >= o.MaxTokens {
		return fmt.Errorf("sizing: min tokens (%d) must be smaller than max tokens (%d)", o.MinTokens, o.MaxTokens)
	}
	return nil
}

// Apply splits oversized candidates and then merges undersized ones.
func Apply(cands []split.Candidate, opt Options) ([]split.Candidate, error) {
	if err := opt.Validate(); err != nil {
		return nil, err
	}
	if opt.Tokenizer == nil {
		opt.Tokenizer = chunk.ApproxTokenizer{}
	}
	s := sizer{opt: opt}

	var sized []split.Candidate
	for _, c := range cands {
		sized = append(sized, s.splitOversized(c)...)
	}
	return s.mergeUndersized(sized), nil
}
