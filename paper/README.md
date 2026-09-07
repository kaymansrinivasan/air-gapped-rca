# airgap-rca working paper template
paper/Screenshot 2026-09-07 232125.png
## Overleaf

Use New Project > Upload Project and select the supplied ZIP. Alternatively,
extract the ZIP and upload its contents into your existing project, preserving
`figures/` and `generated/`. Replace the starter main.tex. Select main.tex as the
main document and pdfLaTeX as the compiler, then Recompile.

The source is the authoritative manuscript. The included PDF is a compiled
preview. This is a working template, not an 8–12-page completed technical paper.
It will grow with the literature review, system details, and measured results.

## Local build

Run `latexmk -pdf main.tex` from this folder with a TeX installation that includes
IEEEtran, TikZ, booktabs, tabularx, listings, xurl, and hyperref. No shell escape or
network access is required to compile once these packages are installed.
Download TeX dependencies in the connected staging environment in advance.

## Contents

- main.tex: IEEE conference manuscript, initial prose, formulas, architecture,
  schema example, hardware table, and writing prompts.
- figures/: three vector chart placeholders; no fabricated data points.
- generated/: quality, performance, and ablation table holders.
- generate_placeholders.py: standard-library-only script that reproduces those
  empty holders. It is NOT a benchmark exporter and must not be used to overwrite
  future measured results.
- IEEEtran.cls, if bundled: unchanged third-party IEEE class with its original
  license header. Overleaf also supplies IEEEtran through TeX Live.

## Layout versus results

This template uses the IEEEtran conference class. It is not an endorsed or
accepted IEEE publication. The architecture is a proposed design. Equations are
proposed scoring and measurement definitions. All results await experiments.

Template reference: https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/
Class distribution: https://ctan.org/pkg/ieeetran

Bundled class provenance: IEEEtran V1.8b (2015/08/26), retrieved unchanged from https://github.com/bardsoftware/template-ieee-transactions/blob/master/IEEEtran.cls .
