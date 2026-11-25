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

// Options configures a pipeline run.
type Options struct {
	// MaxTokens and MinTokens are the chunk size range. Zero means the
	// sizing defaults; a negative MinTokens turns merging off.
	MaxTokens int
	MinTokens int
	// DedupThreshold is the estimated Jaccard similarity at which a chunk
	// is dropped as a near-duplicate. Zero means the dedup default; a
	// negative value turns near-duplicate removal off.
	DedupThreshold float64
	// Percentile is the semantic boundary percentile. Zero means the
	// semantic default.
	Percentile float64
	// Workers is the number of documents processed at once. Zero means
	// one per CPU.
	Workers int
	// Embedder powers the semantic stage. Nil skips that stage.
	Embedder embed.Embedder
	// Tokenizer defaults to chunk.ApproxTokenizer.
	Tokenizer chunk.Tokenizer
	// Lines tunes repeated-line removal.
	Lines dedup.LineOptions
	// KeepRepeatedLines turns repeated-line removal off.
	KeepRepeatedLines bool
}

func (o Options) withDefaults() Options {
	d := sizing.DefaultOptions()
	if o.MaxTokens == 0 {
		o.MaxTokens = d.MaxTokens
	}
	switch {
	case o.MinTokens < 0:
		o.MinTokens = 0
	case o.MinTokens == 0:
		o.MinTokens = d.MinTokens
		if o.MinTokens >= o.MaxTokens {
			o.MinTokens = o.MaxTokens / 4
		}
	}
	if o.Workers <= 0 {
		o.Workers = runtime.NumCPU()
	}
	if o.Tokenizer == nil {
		o.Tokenizer = chunk.ApproxTokenizer{}
	}
	return o
}

func (o Options) sizing() sizing.Options {
	return sizing.Options{MaxTokens: o.MaxTokens, MinTokens: o.MinTokens, Tokenizer: o.Tokenizer}
}

// Stats summarises what a run did.
type Stats struct {
	Docs           int
	Chunks         int
	LinesStripped  int
	DuplicatesDrop int
	Oversized      int
}

// ChunkDoc runs the per-document stages on one document's source and
// returns its candidates.
func ChunkDoc(ctx context.Context, source []byte, opt Options) ([]split.Candidate, error) {
	return chunkDoc(ctx, source, opt.withDefaults())
}

// chunkDoc expects options that already have their defaults applied.
// Applying them twice would turn an explicit "no merging" back into the
// default minimum.
func chunkDoc(ctx context.Context, source []byte, opt Options) ([]split.Candidate, error) {
	blocks, err := parse.Parse(source)
	if err != nil {
		return nil, err
	}
	cands := split.Structure(blocks)
	cands, err = semantic.Refine(ctx, cands, opt.Embedder, semantic.Options{Percentile: opt.Percentile})
	if err != nil {
		return nil, err
	}
	return sizing.Apply(cands, opt.sizing())
}

// Run chunks every document and returns the chunks in document order.
func Run(ctx context.Context, docs []Doc, opt Options) ([]chunk.Chunk, Stats, error) {
	opt = opt.withDefaults()
	if err := opt.sizing().Validate(); err != nil {
		return nil, Stats{}, fmt.Errorf("pipeline: %w", err)
	}

	perDoc := make([][]split.Candidate, len(docs))
	err := ForEach(ctx, len(docs), opt.Workers, func(ctx context.Context, i int) error {
		src, err := docs[i].Read()
		if err != nil {
			return err
		}
		cands, err := chunkDoc(ctx, src, opt)
		if err != nil {
			return fmt.Errorf("chunking %s: %w", docs[i].Name, err)
		}
		perDoc[i] = cands
		return nil
	})
	if err != nil {
		return nil, Stats{}, fmt.Errorf("pipeline: %w", err)
	}

	stats := Stats{Docs: len(docs)}

	// Corpus-level noise removal, stage one: repeated lines.
	if !opt.KeepRepeatedLines {
		var all []split.Candidate
		for _, cands := range perDoc {
			all = append(all, cands...)
		}
		lines := dedup.RepeatedLines(all, opt.Lines)
		for i := range perDoc {
			var n int
			perDoc[i], n = dedup.StripLines(perDoc[i], lines)
			stats.LinesStripped += n
		}
	}

	var chunks []chunk.Chunk
	for i, cands := range perDoc {
		for _, c := range cands {
			chunks = append(chunks, c.Chunk(docs[i].Name, opt.Tokenizer))
		}
	}

	// Stage two: near-duplicate chunks. The first occurrence is kept.
	if opt.DedupThreshold >= 0 {
		texts := make([]string, len(chunks))
		for i, c := range chunks {
			texts[i] = c.Text
		}
		dupOf, err := dedup.FindDuplicates(texts, dedup.Options{Threshold: opt.DedupThreshold})
		if err != nil {
			return nil, Stats{}, fmt.Errorf("pipeline: %w", err)
		}
		kept := chunks[:0]
		for i, c := range chunks {
			if dupOf[i] >= 0 {
				stats.DuplicatesDrop++
				continue
			}
			kept = append(kept, c)
		}
		chunks = kept
	}

	for _, c := range chunks {
		if c.HasFlag(chunk.FlagOversized) {
			stats.Oversized++
		}
	}
	stats.Chunks = len(chunks)
	return chunks, stats, nil
}

// RunBaseline chunks every document with the fixed-size splitter and
// returns the chunks in document order.
func RunBaseline(ctx context.Context, docs []Doc, opt baseline.Options, workers int) ([]chunk.Chunk, error) {
	if err := opt.Validate(); err != nil {
		return nil, fmt.Errorf("pipeline: %w", err)
	}
	perDoc := make([][]chunk.Chunk, len(docs))
	err := ForEach(ctx, len(docs), workers, func(ctx context.Context, i int) error {
		src, err := docs[i].Read()
		if err != nil {
			return err
		}
		chunks, err := baseline.Split(docs[i].Name, src, opt)
		if err != nil {
			return fmt.Errorf("splitting %s: %w", docs[i].Name, err)
		}
		perDoc[i] = chunks
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("pipeline: %w", err)
	}
	var out []chunk.Chunk
	for _, chunks := range perDoc {
		out = append(out, chunks...)
	}
	return out, nil
}

// ForEach calls fn for every index in [0, n) using a pool of workers. It
// returns the first error and cancels the context passed to the remaining
// calls. fn must write its result somewhere indexed by i; ForEach itself
// imposes no ordering on when each call runs.
func ForEach(ctx context.Context, n, workers int, fn func(ctx context.Context, i int) error) error {
	if n == 0 {
		return ctx.Err()
	}
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers > n {
		workers = n
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	jobs := make(chan int)
	var (
		wg       sync.WaitGroup
		once     sync.Once
		firstErr error
	)
	fail := func(err error) {
		once.Do(func() {
			firstErr = err
			cancel()
		})
	}
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				if ctx.Err() != nil {
					continue // drain without working once something failed
				}
				if err := fn(ctx, i); err != nil {
					fail(err)
				}
			}
		}()
	}
feed:
	for i := 0; i < n; i++ {
		select {
		case jobs <- i:
		case <-ctx.Done():
			break feed
		}
	}
	close(jobs)
	wg.Wait()

	if firstErr != nil {
		return firstErr
	}
	return ctx.Err()
}
