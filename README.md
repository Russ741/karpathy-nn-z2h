```
conda env create -f environment.yml
conda activate karpathy-nn-z2h
curl https://raw.githubusercontent.com/karpathy/makemore/master/names.txt > names.txt
jupyter notebook

Navigate to http://localhost:8888/notebooks/02-building-makemore/02-building_makemore.ipynb
```

To update the conda environment when the environment.yml file is updated:
```
conda env update -f environment.yml
```