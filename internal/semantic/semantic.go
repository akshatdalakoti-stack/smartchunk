// Package semantic refines structural candidates by cutting sections where
// the topic shifts.
//
// Every unit of a document (a prose block plus the code, tables and lists
// attached to it) is embedded. The cosine similarity of each neighbouring
// pair is computed, and a section is cut wherever that similarity falls
// below a percentile of the document's own similarity distribution. Using a
// per-document percentile instead of a fixed number means a tightly focused
// reference page and a rambling tutorial each get cuts at their own weakest
// joins.
package semantic

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/akshatdalakoti-stack/smartchunk/internal/embed"
	"github.com/akshatdalakoti-stack/smartchunk/internal/parse"
	"github.com/akshatdalakoti-stack/smartchunk/internal/split"
)

// cutEpsilon is how far below the threshold a similarity must fall to count
// as a boundary. Without it, a document whose joins are all equally strong
// would be cut on float32 rounding noise.
const cutEpsilon = 1e-5

// Options tunes boundary detection.
type Options struct {
	// Percentile (0-100) of the document's neighbour similarities below
	// which a boundary is cut. Zero means 25.
	Percentile float64
	// MinPairs is the number of neighbouring pairs a document needs before
	// its distribution is trusted. Shorter documents are left as they are.
	// Zero means 4.
	MinPairs int
}

func (o Options) withDefaults() Options {
	if o.Percentile <= 0 {
		o.Percentile = 25
	}
	if o.Percentile > 100 {
		o.Percentile = 100
	}
	if o.MinPairs <= 0 {
		o.MinPairs = 4
	}
	return o
}
