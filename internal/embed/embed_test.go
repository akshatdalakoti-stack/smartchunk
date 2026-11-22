package embed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func TestCosine(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64
	}{
		{"identical", []float32{1, 2, 3}, []float32{1, 2, 3}, 1},
		{"scaled", []float32{1, 2, 3}, []float32{2, 4, 6}, 1},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, -1},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"forty five degrees", []float32{1, 0}, []float32{1, 1}, math.Sqrt2 / 2},
		{"zero vector", []float32{0, 0}, []float32{1, 1}, 0},
		{"length mismatch", []float32{1, 2}, []float32{1, 2, 3}, 0},
		{"empty", nil, nil, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Cosine(tt.a, tt.b); math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("Cosine = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNormalizeAndDot(t *testing.T) {
	tests := []struct {
		name     string
		in       []float32
		wantNorm float64
	}{
		{"three four", []float32{3, 4}, 1},
		{"already unit", []float32{0, 1}, 1},
		{"zero stays zero", []float32{0, 0}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Normalize(tt.in)
			if norm := math.Sqrt(Dot(got, got)); math.Abs(norm-tt.wantNorm) > 1e-6 {
				t.Errorf("norm = %v, want %v", norm, tt.wantNorm)
			}
			if len(tt.in) > 0 && &got[0] == &tt.in[0] {
				t.Error("Normalize must return a copy")
			}
		})
	}
	if Dot([]float32{1}, []float32{1, 2}) != 0 {
		t.Error("Dot of mismatched lengths should be 0")
	}
}

func TestFakeEmbedder(t *testing.T) {
	ctx := context.Background()
	f := Fake{}
	texts := []string{
		"install the package with pip",
		"install the package using pip",
		"quantum chromodynamics lattice simulation",
		"",
	}
	a, err := f.Embed(ctx, texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	b, _ := f.Embed(ctx, texts)
	if !reflect.DeepEqual(a, b) {
		t.Error("fake embedder is not deterministic")
	}
	if len(a) != len(texts) || len(a[0]) != 64 {
		t.Fatalf("got %d vectors of dim %d", len(a), len(a[0]))
	}
	near, far := Cosine(a[0], a[1]), Cosine(a[0], a[2])
	if near <= far {
		t.Errorf("similar texts scored %v, unrelated texts %v", near, far)
	}
	if Cosine(a[3], a[0]) != 0 {
		t.Error("empty text should embed to the zero vector")
	}
	if v, _ := (Fake{Dim: 8}).Embed(ctx, []string{"x"}); len(v[0]) != 8 {
		t.Errorf("Dim not honoured: %d", len(v[0]))
	}

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := f.Embed(cancelled, texts); !errors.Is(err, context.Canceled) {
		t.Errorf("cancelled context: err = %v", err)
	}
}

// fakeOllama serves /api/embed, returning for each input a 2-d vector of
// (its length, its index in the batch). It records the batch sizes it saw.
func fakeOllama(t *testing.T, batches *[]int, mu *sync.Mutex) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" || r.Method != http.MethodPost {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		var req ollamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if !req.Truncate || req.Model == "" {
			http.Error(w, "truncate and model are required", http.StatusBadRequest)
			return
		}
		mu.Lock()
		*batches = append(*batches, len(req.Input))
		mu.Unlock()
		resp := ollamaResponse{}
		for i, in := range req.Input {
			resp.Embeddings = append(resp.Embeddings, []float32{float32(len(in)), float32(i)})
		}
		json.NewEncoder(w).Encode(resp)
	}))
}

func TestOllamaBatching(t *testing.T) {
	tests := []struct {
		name        string
		texts       int
		batchSize   int
		wantBatches []int
	}{
		{"no texts", 0, 4, nil},
		{"single batch", 3, 4, []int{3}},
		{"exact multiple", 8, 4, []int{4, 4}},
		{"remainder", 10, 4, []int{4, 4, 2}},
		{"default batch size", 40, 0, []int{32, 8}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var batches []int
			var mu sync.Mutex
			srv := fakeOllama(t, &batches, &mu)
			defer srv.Close()

			o := NewOllama(srv.URL+"/", "test-model")
			o.BatchSize = tt.batchSize
			texts := make([]string, tt.texts)
			for i := range texts {
				texts[i] = strings.Repeat("x", i+1)
			}
			vecs, err := o.Embed(context.Background(), texts)
			if err != nil {
				t.Fatalf("Embed: %v", err)
			}
			if len(vecs) != tt.texts {
				t.Fatalf("got %d vectors, want %d", len(vecs), tt.texts)
			}
			for i, v := range vecs {
				if int(v[0]) != i+1 {
					t.Errorf("vector %d belongs to a text of length %v: order was not preserved", i, v[0])
				}
			}
			if !reflect.DeepEqual(batches, tt.wantBatches) {
				t.Errorf("batches = %v, want %v", batches, tt.wantBatches)
			}
		})
	}
}

func TestOllamaErrors(t *testing.T) {
	tests := []struct {
		name      string
		handler   http.HandlerFunc
		wantErr   string
		wantCalls int32
	}{
		{
			name: "model not found is not retried",
			handler: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, `{"error":"model not found"}`, http.StatusNotFound)
			},
			wantErr:   "ollama pull",
			wantCalls: 1,
		},
		{
			name: "server error is retried",
			handler: func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "boom", http.StatusInternalServerError)
			},
			wantErr:   "500",
			wantCalls: 2,
		},
		{
			name: "wrong number of embeddings",
			handler: func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, `{"embeddings":[[1,2]]}`)
			},
			wantErr:   "1 embeddings for 2 texts",
			wantCalls: 1,
		},
		{
			name: "malformed json",
			handler: func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, `{"embeddings":`)
			},
			wantErr:   "decoding",
			wantCalls: 1,
		},
		{
			name: "error field",
			handler: func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, `{"error":"out of memory"}`)
			},
			wantErr:   "out of memory",
			wantCalls: 1,
		},
		{
			name: "empty embedding",
			handler: func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, `{"embeddings":[[1],[]]}`)
			},
			wantErr:   "empty embedding",
			wantCalls: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&calls, 1)
				tt.handler(w, r)
			}))
			defer srv.Close()

			o := NewOllama(srv.URL, "m")
			o.Retries = 1
			_, err := o.Embed(context.Background(), []string{"a", "b"})
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("err = %v, want it to contain %q", err, tt.wantErr)
			}
			if got := atomic.LoadInt32(&calls); got != tt.wantCalls {
				t.Errorf("server saw %d calls, want %d", got, tt.wantCalls)
			}
		})
	}
}

func TestOllamaUnreachable(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	url := srv.URL
	srv.Close() // nothing listens here any more

	o := NewOllama(url, "m")
	o.Retries = -1
	_, err := o.Embed(context.Background(), []string{"a"})
	if err == nil || !strings.Contains(err.Error(), "ollama serve") {
		t.Errorf("err = %v, want a hint about `ollama serve`", err)
	}
}

func TestOllamaContextCancel(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-started
		cancel()
	}()
	_, err := NewOllama(srv.URL, "m").Embed(ctx, []string{"a"})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func TestOllamaPrepareText(t *testing.T) {
	o := &Ollama{MaxChars: 5}
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"short text untouched", "abc", "abc"},
		{"long text capped", "abcdefgh", "abcde"},
		{"cut lands on a rune boundary", "abcdé", "abcd"},
		{"empty becomes placeholder", "", "-"},
		{"whitespace becomes placeholder", " \n\t", "-"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := o.PrepareText(tt.in); got != tt.want {
				t.Errorf("PrepareText(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
