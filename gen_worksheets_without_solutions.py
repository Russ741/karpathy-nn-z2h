#!/usr/bin/env python
from os import chdir, listdir

WORKSHEET_DIR = "worksheets"

chdir(WORKSHEET_DIR)
filenames = listdir()

SOLN_INFIX = "-solutions"
SOLN_SUFFIX = f"{SOLN_INFIX}.py"
NO_SOLN_SUFFIX = ".py"

SOLN_HEADER = "# Solution code"
SOLN_FOOTER = "# End solution code"

worksheet_prefixes = []

for filename in filenames:
    if not filename.endswith(SOLN_SUFFIX):
        continue
    file_prefix = filename.removesuffix(SOLN_SUFFIX)
    worksheet_prefixes.append(file_prefix)
    outfile_name = file_prefix + NO_SOLN_SUFFIX

    is_solution = False
    with open(filename) as infile, open(outfile_name, 'w') as outfile:
        for line in infile:
            print(f"{line=}")
            if line.startswith(SOLN_HEADER):
                is_solution = True
                outfile.write("# TODO: Implement solution here\n")
            elif line.startswith(SOLN_FOOTER):
                is_solution = False
            elif not is_solution:
                outfile.write(line)

COLAB_PREFIX = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Russ741/karpathy-nn-z2h/blob/main/{WORKSHEET_DIR}"
COLAB_SUFFIX = ")"

print("Worksheet | Colab | .ipynb | Colab (w/solutions) | .ipynb (w/solutions)")
print("--- | --- | --- | --- | ---")
for worksheet_prefix in worksheet_prefixes:
    print(
        f"{worksheet_prefix} | "
        f"{COLAB_PREFIX}/{worksheet_prefix}.ipynb{COLAB_SUFFIX} | "
        f"[link]({WORKSHEET_DIR}/{worksheet_prefix}.py) | "
        f"{COLAB_PREFIX}/{worksheet_prefix}{SOLN_INFIX}.ipynb{COLAB_SUFFIX} | "
        f"[link]({WORKSHEET_DIR}/{worksheet_prefix}{SOLN_INFIX}.py)"
    )
