package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"
)

// Defaults for the Ollama client.
const (
	DefaultOllamaURL   = "http://localhost:11434"
	DefaultOllamaModel = "all-minilm"
	defaultBatchSize   = 32
	defaultMaxChars    = 8000
	defaultRetries     = 2
)

// Ollama embeds text through a local Ollama server's POST /api/embed
// endpoint. Requests are sent in batches.
type Ollama struct {
	// BaseURL is the server address, e.g. http://localhost:11434.
	BaseURL string
	// Model is the embedding model name.
	Model string
	// BatchSize is how many texts go into one request. Zero means 32.
	BatchSize int
	// MaxChars caps the length of each text sent to the server. The model
	// only reads its context window anyway, so the rest would be wasted
	// bandwidth. Zero means 8000.
	MaxChars int
	// Retries is how many times a failed request is retried when the
	// failure looks transient. Zero means 2; negative means none.
	Retries int
	// Client is the HTTP client. Nil means a client with a 5 minute timeout.
	Client *http.Client
}

// NewOllama returns a client for the given server and model, falling back to
// the defaults for empty arguments.
func NewOllama(baseURL, model string) *Ollama {
	if baseURL == "" {
		baseURL = DefaultOllamaURL
	}
	if model == "" {
		model = DefaultOllamaModel
	}
	return &Ollama{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Model:   model,
		Client:  &http.Client{Timeout: 5 * time.Minute},
	}
}

type ollamaRequest struct {
	Model    string   `json:"model"`
	Input    []string `json:"input"`
	Truncate bool     `json:"truncate"`
}

type ollamaResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Error      string      `json:"error"`
}

// Embed implements Embedder.
func (o *Ollama) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	batch := o.BatchSize
	if batch <= 0 {
		batch = defaultBatchSize
	}
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += batch {
		end := start + batch
		if end > len(texts) {
			end = len(texts)
		}
		vecs, err := o.embedBatch(ctx, texts[start:end])
		if err != nil {
			return nil, fmt.Errorf("embed: texts %d-%d of %d: %w", start, end-1, len(texts), err)
		}
		out = append(out, vecs...)
	}
	return out, nil
}
