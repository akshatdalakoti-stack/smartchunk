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
