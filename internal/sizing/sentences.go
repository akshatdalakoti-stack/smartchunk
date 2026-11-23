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

// boundaryAfter decides whether the terminator at offset i ends a sentence.
// If it does, it returns the offset where the next sentence starts.
func boundaryAfter(text string, i int, term rune) (int, bool) {
	j := i + 1
	for j < len(text) && strings.ContainsRune(`)"']*_`, rune(text[j])) {
		j++
	}
	if j >= len(text) || !isSpace(text[j]) {
		return 0, false
	}
	k := j
	for k < len(text) && isSpace(text[k]) {
		k++
	}
	if k >= len(text) {
		return 0, false
	}
	next, _ := utf8.DecodeRuneInString(text[k:])
	if !unicode.IsUpper(next) && !unicode.IsDigit(next) && !strings.ContainsRune("\"'([`*_#<>", next) {
		return 0, false
	}
	if term == '.' && isFalsePeriod(text, i) {
		return 0, false
	}
	return k, true
}

// isFalsePeriod reports whether the period at offset i belongs to an
// abbreviation, an initial or a list-style number.
func isFalsePeriod(text string, i int) bool {
	head := strings.ToLower(text[:i+1])
	for _, abbr := range abbreviations {
		if strings.HasSuffix(head, abbr) {
			before := len(head) - len(abbr)
			if before == 0 || !isWordByte(head[before-1]) {
				return true
			}
		}
	}
	// The word directly in front of the period.
	w := i
	for w > 0 && isWordByte(text[w-1]) {
		w--
	}
	word := text[w:i]
	if len(word) == 1 && unicode.IsUpper(rune(word[0])) {
		return true // an initial, as in "J. Smith"
	}
	if word != "" && strings.Trim(word, "0123456789") == "" {
		// "1. " at the start of a line is a list marker, not a sentence.
		if w == 0 || text[w-1] == '\n' {
			return true
		}
	}
	return false
}

func isSpace(b byte) bool { return b == ' ' || b == '\n' || b == '\t' || b == '\r' }

func isWordByte(b byte) bool {
	return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' || b >= 0x80
}

// SplitSentences returns the sentences of text with surrounding whitespace
// trimmed.
func SplitSentences(text string) []string {
	var out []string
	for _, s := range sentenceSpans(text) {
		if t := strings.TrimSpace(text[s[0]:s[1]]); t != "" {
			out = append(out, t)
		}
	}
	return out
}
