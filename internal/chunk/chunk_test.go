package chunk

import (
	"strings"
	"testing"
)

func TestApproxTokenizerCount(t *testing.T) {
	tests := []struct {
		name string
		text string
		want int
	}{
		{"empty", "", 0},
		{"whitespace only", "  \n\t ", 0},
		{"single word", "hello", 1},
		{"words", "the quick brown fox", 4},
		{"punctuation counts", "Hello, world!", 4},
		{"code", "x := foo(a, b)", 9},
		{"numbers", "version 1.22 rocks", 5},
		{"unicode letters", "naïve café", 2},
		{"cjk run is one word", "日本語 text", 2},
		{"underscore splits", "snake_case", 3},
		{"newlines separate", "a\nb\n\nc", 3},
		{"fence is one token", "```go", 2},
		{"repeated punctuation is one token", "a --- b === c", 5},
		{"different punctuation is separate", "a->b", 4},
		{"table delimiter row", "|-----|-----|", 5},
		{"space breaks a punctuation run", "- - -", 3},
	}
	tok := ApproxTokenizer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tok.Count(tt.text); got != tt.want {
				t.Errorf("Count(%q) = %d, want %d", tt.text, got, tt.want)
			}
			spans := tok.Spans(tt.text)
			if len(spans) != tt.want {
				t.Errorf("len(Spans(%q)) = %d, want %d", tt.text, len(spans), tt.want)
			}
		})
	}
}

func TestApproxTokenizerSpans(t *testing.T) {
	tok := ApproxTokenizer{}
	text := "Go's fmt.Println(\"héllo\")"
	want := []string{"Go", "'", "s", "fmt", ".", "Println", "(", "\"", "héllo", "\"", ")"}
	spans := tok.Spans(text)
	if len(spans) != len(want) {
		t.Fatalf("got %d spans, want %d", len(spans), len(want))
	}
	prevEnd := 0
	for i, s := range spans {
		if s.Start < prevEnd || s.End <= s.Start || s.End > len(text) {
			t.Fatalf("span %d = %+v is out of order or out of range", i, s)
		}
		if got := text[s.Start:s.End]; got != want[i] {
			t.Errorf("span %d = %q, want %q", i, got, want[i])
		}
		prevEnd = s.End
	}
}

func TestNewID(t *testing.T) {
	base := NewID("a.md", 1, 5, "text")
	tests := []struct {
		name     string
		id       string
		wantSame bool
	}{
		{"identical input", NewID("a.md", 1, 5, "text"), true},
		{"different file", NewID("b.md", 1, 5, "text"), false},
		{"different start", NewID("a.md", 2, 5, "text"), false},
		{"different end", NewID("a.md", 1, 6, "text"), false},
		{"different text", NewID("a.md", 1, 5, "other"), false},
	}
	if len(base) != 16 || strings.Trim(base, "0123456789abcdef") != "" {
		t.Fatalf("id %q is not 16 hex characters", base)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if same := tt.id == base; same != tt.wantSame {
				t.Errorf("id equal = %v, want %v", same, tt.wantSame)
			}
		})
	}
}

func TestHasFlag(t *testing.T) {
	c := Chunk{Flags: []string{FlagOversized}}
	tests := []struct {
		flag string
		want bool
	}{
		{FlagOversized, true},
		{FlagUndersized, false},
		{"", false},
	}
	for _, tt := range tests {
		if got := c.HasFlag(tt.flag); got != tt.want {
			t.Errorf("HasFlag(%q) = %v, want %v", tt.flag, got, tt.want)
		}
	}
}
