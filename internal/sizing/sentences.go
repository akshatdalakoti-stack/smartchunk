package sizing

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// abbreviations that end in a period without ending a sentence.
var abbreviations = []string{
	"e.g.", "i.e.", "vs.", "cf.", "mr.", "mrs.", "ms.", "dr.", "prof.",
	"fig.", "approx.", "st.", "jr.", "sr.", "inc.", "ltd.",
}

// sentenceSpans returns byte ranges that partition text into sentences. Each
// range runs up to the start of the next sentence, so joining them gives the
// text back unchanged.
//
// A sentence ends at '.', '!' or '?' (plus any closing quotes or brackets)
// followed by whitespace and then something that can start a sentence.
// Periods inside inline code, after common abbreviations, after initials and
// after list-style numbers are not sentence ends.
func sentenceSpans(text string) [][2]int {
	var spans [][2]int
	start := 0
	inCode := false
	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		switch {
		case r == '`':
			inCode = !inCode
		case !inCode && (r == '.' || r == '!' || r == '?'):
			if next, ok := boundaryAfter(text, i, r); ok {
				spans = append(spans, [2]int{start, next})
				start = next
				i = next
				continue
			}
		}
		i += size
	}
	if start < len(text) {
		spans = append(spans, [2]int{start, len(text)})
	}
	return spans
}
