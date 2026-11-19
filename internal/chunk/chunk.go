// Package chunk holds the types every stage of SmartChunk shares: the
// exported Chunk record and the Tokenizer interface used to measure it.
package chunk

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// Type classifies what a chunk contains.
type Type string

// Chunk types as they appear in the exported "chunk_type" field.
const (
	Code  Type = "code"
	Table Type = "table"
	Prose Type = "prose"
	Mixed Type = "mixed"
)

// Flags a chunk can carry.
const (
	// FlagOversized marks a chunk above the token limit that was kept whole
	// because it is a single code block, table or list.
	FlagOversized = "oversized"
	// FlagSentenceSplit marks a chunk cut out of a longer prose block at a
	// sentence boundary.
	FlagSentenceSplit = "sentence_split"
	// FlagUndersized marks a chunk below the minimum that had no neighbour
	// it could be merged into.
	FlagUndersized = "undersized"
	// FlagBoilerplateStripped marks a chunk that had repeated header/footer
	// lines removed from it.
	FlagBoilerplateStripped = "boilerplate_stripped"
)

// Chunk is one retrieval unit. Its JSON form is the JSONL export schema.
type Chunk struct {
	ID          string   `json:"id"`
	Text        string   `json:"text"`
	ChunkType   Type     `json:"chunk_type"`
	HeadingPath []string `json:"heading_path"`
	SourceFile  string   `json:"source_file"`
	StartLine   int      `json:"start_line"`
	EndLine     int      `json:"end_line"`
	TokenCount  int      `json:"token_count"`
	Flags       []string `json:"flags"`
}

// NewID derives a stable chunk id from where the chunk came from and what it
// says. The same input always produces the same id.
func NewID(sourceFile string, startLine, endLine int, text string) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\x00%d\x00%d\x00", sourceFile, startLine, endLine)
	h.Write([]byte(text))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// HasFlag reports whether the chunk carries the given flag.
func (c Chunk) HasFlag(flag string) bool {
	for _, f := range c.Flags {
		if f == flag {
			return true
		}
	}
	return false
}
