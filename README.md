# Karpathy Neural Networks: Zero To Hero Worksheets

Interactive worksheets to practice the concepts in Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) video series.

## How To Use

I recommend starting with the versions without solutions, and only consulting the provided solution for a given cell once 1) you've completed it or 2) you're stuck despite reviewing the relevant section of video.

### Google Colab

**IMPORTANT NOTE**: Make sure to save a copy of the worksheet (```File -> Save a copy in Drive```) before starting work on it, and verify that you can find the copy you've made.

Worksheet | Colab | .py | Colab (w/solutions) | .py (w/solutions)
--- | --- | --- | --- | ---
02a-building_makemore_probability_table | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Russ741/karpathy-nn-z2h/blob/main/worksheets/02a-building_makemore_probability_table.ipynb) | [link](worksheets/02a-building_makemore_probability_table.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Russ741/karpathy-nn-z2h/blob/main/worksheets/02a-building_makemore_probability_table-solutions.ipynb) | [link](worksheets/02a-building_makemore_probability_table-solutions.py)
02b-building_makemore_slp | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Russ741/karpathy-nn-z2h/blob/main/worksheets/02b-building_makemore_slp.ipynb) | [link](worksheets/02b-building_makemore_slp.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Russ741/karpathy-nn-z2h/blob/main/worksheets/02b-building_makemore_slp-solutions.ipynb) | [link](worksheets/02b-building_makemore_slp-solutions.py)
03-building_makemore_mlp | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Russ741/karpathy-nn-z2h/blob/main/worksheets/03-building_makemore_mlp.ipynb) | [link](worksheets/03-building_makemore_mlp.py) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Russ741/karpathy-nn-z2h/blob/main/worksheets/03-building_makemore_mlp-solutions.ipynb) | [link](worksheets/03-building_makemore_mlp-solutions.py)

### Local

```console
git clone git@github.com:Russ741/karpathy-nn-z2h.git
cd karpathy-nn-z2h
conda env create -f environment.yml
conda activate karpathy-nn-z2h
pre-commit install
jupyter notebook
```

Then open Jupyter in your browser using the link it printed out in the console, and navigate to the desired worksheet.

To update the conda environment when the environment.yml file is updated:
```
conda env update -f environment.yml
```
