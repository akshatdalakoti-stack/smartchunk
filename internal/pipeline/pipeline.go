// Package pipeline runs the SmartChunk stages end to end:
//
//	parse -> structure split -> semantic boundaries -> size control
//	      -> noise removal -> chunks
//
// The first four stages work on one document at a time and run concurrently
// across documents. Noise removal needs to see the whole corpus (a footer is
// only a footer because it repeats), so it runs once at the end, over the
// candidates in document order. That ordering, not goroutine scheduling,
// decides the output, which makes every run reproducible.
package pipeline

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/akshatdalakoti-stack/smartchunk/internal/baseline"
	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/dedup"
	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/semantic"
	"github.com/akshatdalakoti-stack/smartchunk/internal/sizing"
	"github.com/akshatdalakoti-stack/smartchunk/internal/split"
)

// Doc is one Markdown document to chunk.
type Doc struct {
	// Name is the path reported as source_file: relative to the input
	// root, with forward slashes on every platform.
	Name string
	// Path is where to read the document from. It is ignored when Source
	// is set.
	Path string
	// Source holds the document's bytes when they are already in memory.
	Source []byte
}

// Read returns the document's bytes, loading them from Path if needed.
func (d Doc) Read() ([]byte, error) {
	if d.Source != nil {
		return d.Source, nil
	}
	data, err := os.ReadFile(d.Path)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", d.Name, err)
	}
	return data, nil
}

// Discover lists the Markdown documents at root, which may be a single file
// or a directory to walk. Hidden directories and node_modules are skipped.
// The result is sorted by name.
func Discover(root string) ([]Doc, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("pipeline: %w", err)
	}
	if !info.IsDir() {
		return []Doc{{Name: filepath.Base(root), Path: root}}, nil
	}
	var docs []Doc
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		name := d.Name()
		if d.IsDir() {
			if path != root && (strings.HasPrefix(name, ".") || name == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		if !isMarkdown(name) {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		docs = append(docs, Doc{Name: filepath.ToSlash(rel), Path: path})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("pipeline: walking %s: %w", root, err)
	}
	sort.Slice(docs, func(i, j int) bool { return docs[i].Name < docs[j].Name })
	return docs, nil
}

func isMarkdown(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".md", ".markdown":
		return true
	}
	return false
}
