#!/bin/bash
set -e
cd "$(dirname "$0")"

THEME_REPO="https://github.com/javierip/custom-beamer.git"
THEME_SRC=".custom-beamer"

# El tema UTN-BHI no se versiona ni se copia en este repositorio: se clona y
# se usa desde su propio directorio.
# Para actualizarlo: rm -rf .custom-beamer && ./compile.sh
if [ ! -d "$THEME_SRC" ]; then
  echo "--- Obteniendo el tema UTN-BHI desde $THEME_REPO ---"
  git clone --depth 1 "$THEME_REPO" "$THEME_SRC"
fi

# Asi encuentra beamerthemeUTN-BHI.sty sin copiarlo. Sus imagenes las ubica
# \utnbhiassetpath, definido en el preambulo del .tex. El separador final
# conserva las rutas propias de TeX.
export TEXINPUTS="$THEME_SRC:$TEXINPUTS"

# Dos pasadas: la segunda resuelve las referencias internas de beamer.
pdflatex -interaction=nonstopmode slides_pico-JEPA-v2.tex
pdflatex -interaction=nonstopmode slides_pico-JEPA-v2.tex

echo "Listo: slides_pico-JEPA-v2.pdf"
