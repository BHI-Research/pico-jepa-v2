#!/bin/bash
set -e
cd "$(dirname "$0")"

THEME_REPO="https://github.com/javierip/custom-beamer.git"
THEME_SRC=".custom-beamer"

# El tema UTN-BHI no se versiona en este repositorio: se obtiene de su origen.
# Para actualizarlo: rm -rf .custom-beamer && ./compile.sh
if [ ! -d "$THEME_SRC" ]; then
  echo "--- Obteniendo el tema UTN-BHI desde $THEME_REPO ---"
  git clone --depth 1 "$THEME_REPO" "$THEME_SRC"
fi

# beamer busca beamerthemeUTN-BHI.sty en el directorio de trabajo, y ese .sty
# carga el fondo como theme/utn-bhi-page (ruta relativa): los dos tienen que
# quedar junto al .tex. Ambos destinos estan en .gitignore.
cp "$THEME_SRC/beamerthemeUTN-BHI.sty" .
mkdir -p theme
cp "$THEME_SRC"/theme/*.png theme/

# Dos pasadas: la segunda resuelve las referencias internas de beamer.
pdflatex -interaction=nonstopmode slides_pico-JEPA-v2.tex
pdflatex -interaction=nonstopmode slides_pico-JEPA-v2.tex

echo "Listo: slides_pico-JEPA-v2.pdf"
