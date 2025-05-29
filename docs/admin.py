# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Documentation

# %% [markdown]
# ### HTML

# %% language="bash"
#
# make html
# find build -type f -name '*.ipynb' -delete

# %% [markdown]
# ### Clear

# %% language="bash"
# rm -rf build/html/*

# %% language="bash"
# eval "$(conda shell.bash hook)"
# conda activate /home/jupyter-blankjul/anaconda3/envs/pypi
# python -m ipykernel install --user --name=pipy

# %% [markdown]
# ## Notebooks

# %% [markdown]
# ### Serve

# %% language="bash"
#
# python3 -m http.server --directory  build/html

# %% [markdown]
# ### Clear

# %% language="bash"
#
# jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace  source/algorithms/moo/nsga2.ipynb

# %% language="bash"
#
# for file in $(find source -name '*.ipynb');
# do
#     echo $file
#     python ./clear.py  $file
# done;

# %% [markdown]
# ### Run

# %% language="bash"
# eval "$(conda shell.bash hook)"
# conda activate /home/jupyter-blankjul/anaconda3/envs/pypi
#
# for file in $(find source -name '*.ipynb');
# do
#     jupyter nbconvert --execute --to notebook --inplace $file
# done;
# echo 'DONE'
