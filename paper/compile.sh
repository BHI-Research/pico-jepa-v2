#!/bin/bash
set -e
cd "$(dirname "$0")"

# Bibliography is embedded via a manual thebibliography environment,
# so bibtex is not needed (and would error: no \bibdata command).
pdflatex -interaction=nonstopmode paper_pico-JEPA-v2.tex
pdflatex -interaction=nonstopmode paper_pico-JEPA-v2.tex

echo "Done: paper_pico-JEPA-v2.pdf"
