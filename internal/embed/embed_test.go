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
