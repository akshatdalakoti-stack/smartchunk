# SmartChunk build targets.
#
#   make build   compile the CLI into bin/
#   make test    go vet, then all tests with the race detector
#   make bench   benchmarks for parse, dedup and the full pipeline
#   make eval    fetch the corpus if needed and run the full evaluation
#
# The race detector needs cgo (a C compiler on PATH). Use `make test RACE=`
# to run without it.

GO      ?= go
BIN     := bin/smartchunk$(shell $(GO) env GOEXE)
CORPUS  ?= testdata/corpus
QUESTIONS ?= testdata/questions.jsonl
REPORT  ?= report
RACE    ?= -race
EVAL_FLAGS ?=

.PHONY: all build test vet bench corpus eval clean

all: build

build:
	$(GO) build -o $(BIN) ./cmd/smartchunk

vet:
	$(GO) vet ./...

test: vet
	$(GO) test $(RACE) ./...

bench:
	$(GO) test -run '^$$' -bench . -benchmem ./internal/parse ./internal/dedup ./internal/pipeline

corpus:
	bash scripts/fetch_corpus.sh $(CORPUS)

# Needs a running Ollama with the all-minilm model:
#   ollama serve &  ollama pull all-minilm
# Add EVAL_FLAGS="--embedder fake" for an offline run with the test embedder.
eval: build corpus
	$(BIN) eval --corpus $(CORPUS) --questions $(QUESTIONS) --out $(REPORT) $(EVAL_FLAGS)

clean:
	rm -rf bin $(REPORT)
