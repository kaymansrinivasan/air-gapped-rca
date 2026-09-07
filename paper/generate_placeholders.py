"""Regenerate empty manuscript holders. This script does not process benchmarks.

Replace its role with the benchmark's real export script when results exist.
Never overwrite measured outputs with this script: it refuses non-placeholder files.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
MARK = '% PLACEHOLDER ONLY: no experimental measurements.\n'

def write(name, body):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not p.read_text().startswith(MARK):
        raise SystemExit(f'Refusing to overwrite non-placeholder file: {p}')
    p.write_text(MARK + body)

def table(name, caption, label, rows):
    body = '\\begin{table}[t]\n\\caption{' + caption + '}\\label{' + label + '}\n\\centering\\footnotesize\n'
    body += '\\begin{tabularx}{\\columnwidth}{@{}Yll@{}}\\toprule\nMetric & Jetson & RB3\\\\\\midrule\n'
    for row in rows:
        body += row + ' & \\pending & \\pending\\\\\n'
    body += '\\bottomrule\\end{tabularx}\n\\end{table}\n'
    write('generated/' + name + '.tex', body)

table('quality', 'Quality holders. Rates require explicit denominators and uncertainty.', 'tab:quality', [
'Answerable / unanswerable count', 'Top-1 / top-3 accuracy (\\%)', 'False answer rate (\\%)',
'Error among answers (\\%)', 'Any-wrong-candidate rate (\\%)', 'Coverage / refusal rate (\\%)',
'Correct refusal rate (\\%)', 'Citation pass rate, pre-filter (\\%)', 'Post-filter citation integrity (\\%)', 'Execution-error rate (\\%)'])
table('performance', 'Performance holders. Record repetitions and power boundary.', 'tab:performance', [
'Latency: median / p95 (s)', 'Time to first token (s)', 'Decode throughput (tokens/s)',
'Average / peak power (W)', 'Mean energy per triage (J)', 'Electricity price (currency/kWh)',
'Cost per 1,000 (currency)', 'Cold-start / indexing time (s)'])
body = r'''\begin{table}[t]
\caption{Ablation holders. J/R denotes separate Jetson and RB3 results in each cell. All entries await measurements.}
\label{tab:ablation}\centering\scriptsize
\begin{tabularx}{\columnwidth}{@{}Yccc@{}}\toprule
Configuration & FAR (\%) & Coverage (\%) & Acc@1 (\%)\\\midrule
Full pipeline & --/-- & --/-- & --/--\\
Without keywords & --/-- & --/-- & --/--\\
Without verification & --/-- & --/-- & --/--\\
Without refusal & --/-- & --/-- & --/--\\\bottomrule
\end{tabularx}
\end{table}
'''
write('generated/ablation.tex',body)
for name,xlabel,ylabel,categories in [
    ('tradeoff','Median end-to-end latency (s)','Top-1 accuracy (\\%)',[]),
    ('energy','Platform','Mean energy per triage (J)',['Jetson','RB3']),
    ('far','Verification configuration','False answer rate (\\%)',['With verification','Without verification'])]:
    body = r'''\begin{tikzpicture}[font=\scriptsize]
\draw[gray!60] (0,0) rectangle (6.5,3.1);
\node[align=center,text=accent] at (3.25,1.65) {Awaiting benchmark output\\No measured values};
'''
    body += '\\node[rotate=90] at (-0.48,1.55) {'+ylabel+'};\n'
    body += '\\node at (3.25,-0.65) {'+xlabel+'};\n'
    if categories:
        for x,cat in zip([1.65,4.85],categories): body += f'\\node at ({x},-0.22) {{{cat}}};\n'
    body += '\\end{tikzpicture}\n'
    write('figures/'+name+'.tex',body)
print('Generated three pending tables and three empty chart holders.')
