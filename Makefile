.PHONY: all figures clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

all: figures  ## Generate all figures from data

figures:  ## Generate all 4 PDF figures from result data
	python scripts/generate_figures.py

clean:  ## Remove generated figures
	rm -f figures/*.pdf
