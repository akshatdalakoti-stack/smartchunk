package chunk

import "unicode"

// Span is the byte range [Start, End) of one token inside a text.
type Span struct {
	Start int
	End   int
}

// Tokenizer measures text in tokens. Implementations must be safe for
// concurrent use.
type Tokenizer interface {
	// Count returns the number of tokens in text.
	Count(text string) int
	// Spans returns the byte range of every token, in order.
	Spans(text string) []Span
}

// ApproxTokenizer is the default tokenizer. It needs no vocabulary:
//
//   - a run of letters and digits is one token;
//   - a run of the same punctuation character is one token, so "```",
//     "---" and "==" each count once, as they do in subword vocabularies;
//   - every other punctuation character is a token of its own;
//   - whitespace separates tokens and is not counted.
//
// It is an approximation. Subword tokenizers split rare words further, so
// this undercounts English prose by roughly 20-30%; on code, where most
// tokens are punctuation, it is close.
type ApproxTokenizer struct{}

// The tokenizer is a three-state machine over runes.
const (
	inSpace = iota
	inWord
	inPunct
)

// Count implements Tokenizer without allocating.
func (ApproxTokenizer) Count(text string) int {
	n := 0
	state := inSpace
	var punct rune
	for _, r := range text {
		switch {
		case isWordRune(r):
			if state != inWord {
				n++
			}
			state = inWord
		case unicode.IsSpace(r):
			state = inSpace
		default:
			if state != inPunct || punct != r {
				n++
			}
			state, punct = inPunct, r
		}
	}
	return n
}

// Spans implements Tokenizer.
func (ApproxTokenizer) Spans(text string) []Span {
	spans := make([]Span, 0, len(text)/4)
	start := -1
	flush := func(end int) {
		if start >= 0 {
			spans = append(spans, Span{start, end})
			start = -1
		}
	}
	state := inSpace
	var punct rune
	for i, r := range text {
		switch {
		case isWordRune(r):
			if state != inWord {
				flush(i)
				start = i
			}
			state = inWord
		case unicode.IsSpace(r):
			flush(i)
			state = inSpace
		default:
			if state != inPunct || punct != r {
				flush(i)
				start = i
			}
			state, punct = inPunct, r
		}
	}
	flush(len(text))
	return spans
}

func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}
