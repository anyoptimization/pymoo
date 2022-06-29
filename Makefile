clean:
	rm -rf build dist pymoo.egg-info

clean-ext:
	rm -f pymoo/cython/*.c
	rm -f pymoo/cython/*.so
	rm -f pymoo/cython/*.cpp
	rm -f pymoo/cython/*.html

compile:
	python setup.py build_ext --inplace

compile-without-cython:
	python setup.py build_ext --inplace --no-cython

dist:
	python setup.py sdist

install:
	python setup.py install

