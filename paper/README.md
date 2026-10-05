# Paper — pico-JEPA v2

LaTeX sources for *"pico-JEPA v2: ¿Cuándo Superan Muchos Modelos Pequeños a Uno
Grande?"* (Springer LNCS, Spanish). Source: `paper_pico-JEPA-v2.tex`, plus the
bundled `llncs.cls` and the three `fig*.png` figures.

## Requirements

A TeX distribution with `pdflatex`, Spanish `babel` and `cm-super` fonts —
a full TeX Live or MiKTeX install covers everything.

```bash
# Debian/Ubuntu (also inside WSL)
sudo apt install texlive-latex-recommended texlive-latex-extra \
                 texlive-lang-spanish texlive-fonts-recommended cm-super
```

The bibliography is embedded in the `.tex`, so BibTeX is not needed: two
`pdflatex` passes are enough.

## Build

**Linux, WSL or Git Bash**

```bash
cd paper
./compile.sh
```

**Windows, from PowerShell** — via WSL:

```powershell
wsl -- bash -c "cd /mnt/c/path/to/pico-jepa-v2/paper && ./compile.sh"
```

or with a native MiKTeX / TeX Live (two lines: Windows PowerShell has no `&&`):

```powershell
cd paper
pdflatex -interaction=nonstopmode paper_pico-JEPA-v2.tex
pdflatex -interaction=nonstopmode paper_pico-JEPA-v2.tex
```

Output: `paper_pico-JEPA-v2.pdf`, 10 pages. A few warnings are normal; only
lines starting with `!` are errors. The built PDF is committed to the repo, so
rebuild it whenever the `.tex` changes; the `.aux`/`.log`/`.out` files are
git-ignored.
