#!/usr/bin/env python
from os import chdir, listdir

chdir('./02-building-makemore')
filenames = listdir()

SOLN_SUFFIX = "-solutions.py"
NO_SOLN_SUFFIX = ".py"

SOLN_HEADER = "# Solution code"
SOLN_FOOTER = "# End solution code"

for filename in filenames:
    if not filename.endswith(SOLN_SUFFIX):
        continue
    print(filename)
    file_prefix = filename.removesuffix(SOLN_SUFFIX)
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
