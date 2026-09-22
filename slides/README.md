# Slides — pico-JEPA v2

Beamer deck (Spanish, 16:9) for the 15-minute conference talk, built from the
paper in [`../paper`](../paper). 19 slides (~13:30) plus 3 backup slides for
questions.

Nothing is duplicated in this repo: the UTN·BHI theme
([custom-beamer](https://github.com/javierip/custom-beamer) v2.0.0) is cloned
by `compile.sh` into `.custom-beamer/` and used from there — `TEXINPUTS`
finds the `.sty`, `\utnbhiassetpath` finds its images — and `\graphicspath`
pulls the three figures from `../paper/`. All of it is git-ignored.

## Build

```bash
cd slides
./compile.sh          # Linux, WSL or Git Bash
```

```powershell
# Windows, via WSL
wsl -- bash -c "cd /mnt/c/path/to/pico-jepa-v2/slides && ./compile.sh"

# Windows, native MiKTeX / TeX Live (run ./compile.sh once first to clone
# the theme; note the ';' path separator)
cd slides
$env:TEXINPUTS = ".custom-beamer;"
pdflatex -interaction=nonstopmode slides_pico-JEPA-v2.tex
pdflatex -interaction=nonstopmode slides_pico-JEPA-v2.tex
```

Needs `git` and network access on the first build, and `pdflatex` with
`beamer`, `tikz`, `booktabs`, `helvet` and Spanish `babel`; on Debian/Ubuntu,
`texlive-latex-recommended texlive-latex-extra texlive-lang-spanish
texlive-pictures`.

To pick up a newer theme release: `rm -rf .custom-beamer` and build again.

## Before presenting

- Set `\date{}` (commented out in the preamble) — otherwise the title slide
  falls back to `\today`.
- Each frame has a `% ~m:ss` time budget and a `\note{}` for the speaker.
  To show notes on a second screen, uncomment the two `pgfpages` /
  `\setbeameroption` lines in the preamble.
- The theme renders the same in 4:3 — drop `aspectratio=169` from
  `\documentclass` if the venue needs it, then re-check the figure widths.
