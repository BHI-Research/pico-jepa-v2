# Slides — pico-JEPA v2

Beamer deck (Spanish, 4:3, 18 slides) for the 15-minute conference talk,
built from the paper in [`../paper`](../paper).

Nothing is duplicated in this repo: the UTN·BHI theme
([custom-beamer](https://github.com/javierip/custom-beamer)) is cloned by
`compile.sh` on the first build and is git-ignored, and `\graphicspath`
pulls the three figures from `../paper/`.

## Build

```bash
cd slides
./compile.sh          # Linux, WSL or Git Bash
```

```powershell
# Windows, via WSL
wsl -- bash -c "cd /mnt/c/path/to/pico-jepa-v2/slides && ./compile.sh"

# Windows, native MiKTeX / TeX Live (after one ./compile.sh run,
# or clone the theme by hand — see below)
cd slides
pdflatex -interaction=nonstopmode slides_pico-JEPA-v2.tex
pdflatex -interaction=nonstopmode slides_pico-JEPA-v2.tex
```

Needs `git` and network access on the first build, and `pdflatex` with
`beamer`, `tikz`, `booktabs` and Spanish `babel`; on Debian/Ubuntu,
`texlive-latex-recommended texlive-latex-extra texlive-lang-spanish
texlive-pictures`.

To refresh the theme, `rm -rf .custom-beamer` and build again. To place it
by hand instead, copy `beamerthemeUTN-BHI.sty` and `theme/` from the
template repo next to the `.tex` — `\usetheme` reads the `.sty` from the
working directory and the `.sty` loads its background as `theme/…`.

## Before presenting

- Set `\date{}` (commented out in the preamble) — otherwise the title slide
  falls back to `\today`.
- Each frame has a `% ~m:ss` time budget and a `\note{}` for the speaker.
  To show notes on a second screen, uncomment the two `pgfpages` /
  `\setbeameroption` lines in the preamble.
