// Package export writes and reads chunks as JSONL, one chunk per line, and
// builds the text a vector database should embed for a chunk.
package export

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
)

// maxLineBytes is the longest JSONL line ReadJSONL accepts. An oversized
// code block can make a single chunk large.
const maxLineBytes = 64 << 20

// WriteJSONL writes one JSON object per chunk, one per line. The
// heading_path and flags fields are always arrays, never null, and HTML
// characters are left unescaped so code survives the round trip readably.
func WriteJSONL(w io.Writer, chunks []chunk.Chunk) error {
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	enc.SetEscapeHTML(false)
	for i, c := range chunks {
		if c.HeadingPath == nil {
			c.HeadingPath = []string{}
		}
		if c.Flags == nil {
			c.Flags = []string{}
		}
		if err := enc.Encode(c); err != nil {
			return fmt.Errorf("export: encoding chunk %d (%s): %w", i, c.ID, err)
		}
	}
	if err := bw.Flush(); err != nil {
		return fmt.Errorf("export: writing chunks: %w", err)
	}
	return nil
}

// WriteFile writes chunks as JSONL to path, creating parent directories.
// The path "-" means standard output.
func WriteFile(path string, chunks []chunk.Chunk) error {
	if path == "-" {
		return WriteJSONL(os.Stdout, chunks)
	}
	if path == "" {
		return errors.New("export: output path is empty")
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("export: creating %s: %w", dir, err)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("export: creating %s: %w", path, err)
	}
	if err := WriteJSONL(f, chunks); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("export: closing %s: %w", path, err)
	}
	return nil
}

// ReadJSONL reads chunks written by WriteJSONL. Blank lines are skipped.
func ReadJSONL(r io.Reader) ([]chunk.Chunk, error) {
	var chunks []chunk.Chunk
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), maxLineBytes)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var c chunk.Chunk
		if err := json.Unmarshal([]byte(raw), &c); err != nil {
			return nil, fmt.Errorf("export: line %d: %w", line, err)
		}
		chunks = append(chunks, c)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("export: reading chunks: %w", err)
	}
	return chunks, nil
}
