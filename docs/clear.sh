find . -type f -name '*.ipynb' -exec sh -c '
for pathname do
  jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$pathname"
done' sh {} +
