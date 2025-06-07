---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Documentation

+++

### HTML

```{code-cell} ipython3
%%bash

make html
find build -type f -name '*.ipynb' -delete
```

### Clear

```{code-cell} ipython3
%%bash
rm -rf build/html/*
```

```{code-cell} ipython3
%%bash
eval "$(conda shell.bash hook)"
conda activate /home/jupyter-blankjul/anaconda3/envs/pypi
python -m ipykernel install --user --name=pipy
```

## Notebooks

+++

### Serve

```{code-cell} ipython3
%%bash

python3 -m http.server --directory  build/html
```

### Clear

```{code-cell} ipython3
%%bash

jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace  source/algorithms/moo/nsga2.ipynb
```

```{code-cell} ipython3
%%bash

for file in $(find source -name '*.ipynb');
do
    echo $file
    python ./clear.py  $file
done;
```

### Run

```{code-cell} ipython3
%%bash
eval "$(conda shell.bash hook)"
conda activate /home/jupyter-blankjul/anaconda3/envs/pypi

for file in $(find source -name '*.ipynb');
do
    jupyter nbconvert --execute --to notebook --inplace $file
done;
echo 'DONE'
```
