# Project page — pico-JEPA v2

The GitHub Pages site: <https://bhi-research.github.io/pico-jepa-v2/>

A one-page English summary of the paper, for visitors arriving from the
repository or the conference. Plain HTML and CSS — no Jekyll, no build step, no
external scripts or fonts. Edit `index.html`, push, done.

```
index.html                       the whole page
static/css/style.css             tokens + layout
.nojekyll                        serve the files as-is
```

## Enabling it (one time)

Settings → Pages → *Deploy from a branch* → branch `main`, folder `/docs`.
The content must be on `main`, not `dev`.

## Maintenance notes

- **The paper and slide PDFs are linked by GitHub URL, not by relative path.**
  Pages serves `docs/` as the site root, so `../paper/…` would not resolve. The
  links point at `blob/main/…`, which GitHub renders in-browser.
- **The results chart is inline HTML/CSS, not an image.** The paper's own
  `fig2_aggregations.png` is labelled in Spanish, so it was rebuilt here in
  English; it scales, themes and prints with the rest of the page.
- **Numbers are hard-coded** from the paper's result tables. If the results
  change, update `index.html` and the paper together.
- **Colors come from the lab's data-viz palette** — surfaces, text ink, one
  accent, and the fixed status roles (`good` / `critical`). Dark mode is a
  selected set of values, not an automatic inversion. Every verdict is a text
  label, so color never carries meaning on its own.

## Checking a change locally

```bash
cd docs
python -m http.server 8000   # then open http://localhost:8000
```
