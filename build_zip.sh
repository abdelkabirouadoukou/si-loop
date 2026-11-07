#!/usr/bin/env bash
# Rebuild the PDF and the bundle:  bash build_zip.sh
set -e
pdflatex -interaction=nonstopmode paper.tex >/dev/null; bibtex paper >/dev/null
pdflatex -interaction=nonstopmode paper.tex >/dev/null; pdflatex -interaction=nonstopmode paper.tex >/dev/null
rm -f si_loop_study1.zip
zip -r si_loop_study1.zip paper.tex paper.pdf references.bib README.md LICENSE requirements.txt .gitignore \
  si_loop_toy.py run_parallel.py analyze.py docs/hypotheses.md docs/hypotheses_prerun.md docs/dev_diagnostics.md \
  results/ablation.json results/analysis.json results/analysis_output.txt results/config_hashes.txt \
  results/final_hard.png results/curves.png build_zip.sh
