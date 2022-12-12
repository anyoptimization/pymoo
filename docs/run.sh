for file in $(find source -name '*.ipynb');
do
    jupyter nbconvert --execute --to notebook --inplace $file
done;
