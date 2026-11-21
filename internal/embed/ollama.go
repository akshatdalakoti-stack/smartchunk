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

// PrepareText returns the text exactly as it will be sent to the server, so
// that a cache in front of the client can key on what was really embedded.
func (o *Ollama) PrepareText(text string) string {
	limit := o.MaxChars
	if limit <= 0 {
		limit = defaultMaxChars
	}
	if len(text) > limit {
		cut := limit
		for cut > 0 && !utf8.RuneStart(text[cut]) {
			cut--
		}
		text = text[:cut]
	}
	if strings.TrimSpace(text) == "" {
		// The server rejects empty input; a lone placeholder keeps the
		// result aligned with the request.
		return "-"
	}
	return text
}

func (o *Ollama) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	input := make([]string, len(texts))
	for i, t := range texts {
		input[i] = o.PrepareText(t)
	}
	body, err := json.Marshal(ollamaRequest{Model: o.Model, Input: input, Truncate: true})
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}

	retries := o.Retries
	if retries == 0 {
		retries = defaultRetries
	}
	var lastErr error
	for attempt := 0; attempt <= retries || attempt == 0; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt) * 500 * time.Millisecond):
			}
		}
		vecs, retry, err := o.post(ctx, body, len(texts))
		if err == nil {
			return vecs, nil
		}
		lastErr = err
		if !retry || ctx.Err() != nil {
			break
		}
	}
	return nil, lastErr
}

// post sends one request. The second result says whether a failure is worth
// retrying.
func (o *Ollama) post(ctx context.Context, body []byte, want int) ([][]float32, bool, error) {
	url := strings.TrimRight(o.BaseURL, "/") + "/api/embed"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, false, fmt.Errorf("building request for %s: %w", url, err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := o.Client
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Minute}
	}
	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, false, ctx.Err()
		}
		return nil, true, fmt.Errorf("calling ollama at %s (is `ollama serve` running?): %w", url, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 256<<20))
	if err != nil {
		return nil, true, fmt.Errorf("reading ollama response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		msg := strings.TrimSpace(string(raw))
		if len(msg) > 300 {
			msg = msg[:300]
		}
		hint := ""
		if resp.StatusCode == http.StatusNotFound {
			hint = fmt.Sprintf(" (try `ollama pull %s`)", o.Model)
		}
		return nil, resp.StatusCode >= 500, fmt.Errorf("ollama returned %s for model %q%s: %s", resp.Status, o.Model, hint, msg)
	}

	var parsed ollamaResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, false, fmt.Errorf("decoding ollama response: %w", err)
	}
	if parsed.Error != "" {
		return nil, false, fmt.Errorf("ollama error: %s", parsed.Error)
	}
	if len(parsed.Embeddings) != want {
		return nil, false, fmt.Errorf("ollama returned %d embeddings for %d texts", len(parsed.Embeddings), want)
	}
	for i, v := range parsed.Embeddings {
		if len(v) == 0 {
			return nil, false, fmt.Errorf("ollama returned an empty embedding for text %d", i)
		}
	}
	return parsed.Embeddings, false, nil
}
