package embed

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// cacheMagic identifies a cache file and its record layout.
const cacheMagic = "SMCHEMB1"

// maxCacheDim guards against reading a corrupt record as a huge vector.
const maxCacheDim = 1 << 16

type cacheKey [sha256.Size]byte

// Cached wraps an Embedder with a cache that survives across runs. Entries
// are keyed by the SHA-256 of the namespace (normally the model name) and
// the text, and stored in a single append-only file:
//
//	8-byte magic, then records of: 32-byte key | uint32 dim | dim float32s
//
// One file rather than one file per entry keeps the cache cheap to sync and
// to delete.
type Cached struct {
	inner     Embedder
	namespace string
	prepare   func(string) string

	mu     sync.Mutex
	mem    map[cacheKey][]float32
	file   *os.File
	hits   int
	misses int
}

// preparer is implemented by embedders that rewrite text before sending it.
type preparer interface {
	PrepareText(string) string
}

// NewCached opens (or creates) the cache file at path and wraps inner with
// it. The namespace keeps vectors from different models apart. A file with a
// damaged tail, as left by an interrupted run, is repaired by dropping the
// incomplete record.
func NewCached(inner Embedder, namespace, path string) (*Cached, error) {
	if inner == nil {
		return nil, errors.New("embed: cache needs an embedder to wrap")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("embed: creating cache directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, fmt.Errorf("embed: opening cache %s: %w", path, err)
	}
	c := &Cached{inner: inner, namespace: namespace, mem: map[cacheKey][]float32{}, file: f}
	if p, ok := inner.(preparer); ok {
		c.prepare = p.PrepareText
	}
	if err := c.load(); err != nil {
		f.Close()
		return nil, fmt.Errorf("embed: loading cache %s: %w", path, err)
	}
	return c, nil
}
