package pipeline

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/akshatdalakoti-stack/smartchunk/internal/chunk"
	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
)

// corpusDir is where scripts/fetch_corpus.sh puts the evaluation corpus.
const corpusDir = "../../testdata/corpus"

// TestCorpusInvariants runs the pipeline over the real evaluation corpus,
// when it has been fetched, and checks the guarantees on documents nobody
// wrote with this parser in mind.
func TestCorpusInvariants(t *testing.T) {
	if _, err := os.Stat(corpusDir); err != nil {
		t.Skip("corpus not fetched; run scripts/fetch_corpus.sh to enable this test")
	}
	docs, err := Discover(corpusDir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if len(docs) == 0 {
		t.Skip("corpus directory is empty")
	}

	opt := Options{Embedder: embed.Fake{}}
	chunks, _, err := Run(context.Background(), docs, opt)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	byFile := map[string][]chunk.Chunk{}
	for _, c := range chunks {
		byFile[c.SourceFile] = append(byFile[c.SourceFile], c)
	}

	for _, d := range docs {
		src, err := d.Read()
		if err != nil {
			t.Fatal(err)
		}
		blocks, err := parse.Parse(src)
		if err != nil {
			t.Fatalf("%s: %v", d.Name, err)
		}
		lines := strings.Split(strings.ReplaceAll(string(src), "\r\n", "\n"), "\n")

		prevEnd := 0
		for i, b := range blocks {
			if b.StartLine <= prevEnd || b.EndLine < b.StartLine || b.EndLine > len(lines) {
				t.Errorf("%s: block %d has range %d-%d after a block ending at %d", d.Name, i, b.StartLine, b.EndLine, prevEnd)
				continue
			}
			prevEnd = b.EndLine
			// A block's text must be exactly the source lines it claims.
			if want := strings.Join(lines[b.StartLine-1:b.EndLine], "\n"); b.Text != want {
				t.Errorf("%s: block %d text does not match source lines %d-%d", d.Name, i, b.StartLine, b.EndLine)
			}
		}

		prevEnd = 0
		for _, c := range byFile[d.Name] {
			if c.StartLine <= prevEnd && prevEnd != 0 && !c.HasFlag(chunk.FlagSentenceSplit) {
				t.Errorf("%s: chunk %d-%d overlaps the chunk ending at %d", d.Name, c.StartLine, c.EndLine, prevEnd)
			}
			prevEnd = c.EndLine
			if c.TokenCount > 256 && !c.HasFlag(chunk.FlagOversized) {
				t.Errorf("%s: chunk %d-%d has %d tokens and no oversized flag", d.Name, c.StartLine, c.EndLine, c.TokenCount)
			}
			for _, b := range blocks {
				if !b.Type.Atomic() {
					continue
				}
				overlaps := c.StartLine <= b.EndLine && c.EndLine >= b.StartLine
				contains := c.StartLine <= b.StartLine && c.EndLine >= b.EndLine
				if overlaps && !contains {
					t.Errorf("%s: chunk %d-%d cuts the %s block at %d-%d", d.Name, c.StartLine, c.EndLine, b.Type, b.StartLine, b.EndLine)
				}
			}
		}
	}
}
