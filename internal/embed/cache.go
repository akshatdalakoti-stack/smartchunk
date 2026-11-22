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

// load reads every complete record and leaves the file positioned for
// appending.
func (c *Cached) load() error {
	info, err := c.file.Stat()
	if err != nil {
		return err
	}
	if info.Size() == 0 {
		_, err := c.file.Write([]byte(cacheMagic))
		return err
	}

	r := bufio.NewReaderSize(c.file, 1<<20)
	magic := make([]byte, len(cacheMagic))
	if _, err := io.ReadFull(r, magic); err != nil || string(magic) != cacheMagic {
		return errors.New("not a SmartChunk embedding cache (delete the file to rebuild it)")
	}

	good := int64(len(cacheMagic))
	for {
		var key cacheKey
		if _, err := io.ReadFull(r, key[:]); err != nil {
			break
		}
		var dimBuf [4]byte
		if _, err := io.ReadFull(r, dimBuf[:]); err != nil {
			break
		}
		dim := binary.LittleEndian.Uint32(dimBuf[:])
		if dim == 0 || dim > maxCacheDim {
			break
		}
		raw := make([]byte, 4*int(dim))
		if _, err := io.ReadFull(r, raw); err != nil {
			break
		}
		vec := make([]float32, dim)
		for i := range vec {
			vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		c.mem[key] = vec
		good += int64(len(key)) + 4 + int64(len(raw))
	}

	// Anything past the last complete record is a torn write: cut it off so
	// new records line up.
	if good < info.Size() {
		if err := c.file.Truncate(good); err != nil {
			return fmt.Errorf("repairing damaged tail: %w", err)
		}
	}
	_, err = c.file.Seek(good, io.SeekStart)
	return err
}

func (c *Cached) key(text string) cacheKey {
	h := sha256.New()
	h.Write([]byte(c.namespace))
	h.Write([]byte{0})
	h.Write([]byte(text))
	var k cacheKey
	h.Sum(k[:0])
	return k
}

// Embed implements Embedder. Only texts that are not in the cache reach the
// wrapped embedder, and identical texts within one call are embedded once.
func (c *Cached) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	keys := make([]cacheKey, len(texts))

	var missTexts []string
	var missKeys []cacheKey
	pending := map[cacheKey]bool{}

	c.mu.Lock()
	for i, t := range texts {
		if c.prepare != nil {
			t = c.prepare(t)
		}
		k := c.key(t)
		keys[i] = k
		if v, ok := c.mem[k]; ok {
			out[i] = v
			c.hits++
			continue
		}
		c.misses++
		if !pending[k] {
			pending[k] = true
			missTexts = append(missTexts, t)
			missKeys = append(missKeys, k)
		}
	}
	c.mu.Unlock()

	if len(missTexts) == 0 {
		return out, nil
	}

	vecs, err := c.inner.Embed(ctx, missTexts)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(missTexts) {
		return nil, fmt.Errorf("embed: embedder returned %d vectors for %d texts", len(vecs), len(missTexts))
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	var buf []byte
	for i, k := range missKeys {
		if _, ok := c.mem[k]; ok {
			continue // another goroutine stored it while we were embedding
		}
		c.mem[k] = vecs[i]
		buf = appendRecord(buf, k, vecs[i])
	}
	if len(buf) > 0 && c.file != nil {
		if _, err := c.file.Write(buf); err != nil {
			return nil, fmt.Errorf("embed: writing cache: %w", err)
		}
	}
	for i, k := range keys {
		if out[i] == nil {
			out[i] = c.mem[k]
		}
	}
	return out, nil
}

func appendRecord(buf []byte, k cacheKey, vec []float32) []byte {
	buf = append(buf, k[:]...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(vec)))
	for _, x := range vec {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(x))
	}
	return buf
}

// Stats returns how many lookups were served from the cache and how many
// had to be embedded.
func (c *Cached) Stats() (hits, misses int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses
}

// Len returns the number of cached vectors.
func (c *Cached) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.mem)
}

// Close flushes and closes the cache file. The cache must not be used
// afterwards.
func (c *Cached) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.file == nil {
		return nil
	}
	err := c.file.Close()
	c.file = nil
	if err != nil {
		return fmt.Errorf("embed: closing cache: %w", err)
	}
	return nil
}
