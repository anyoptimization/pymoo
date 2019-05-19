clean:
	rm -rf build dist pymoo.egg-info

clean-ext:
	rm pymoo/cython/*.c
	rm pymoo/cython/*.so
	rm pymoo/cython/*.cpp
	rm pymoo/cython/*.html

compile:
	python setup.py build_ext --inplace

dist:
	python setup.py sdist

install:
	python setup.py install
	