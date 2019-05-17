clean:
	rm -rf build dist pymoo.egg-info

compile:
	python setup.py build_ext --inplace

dist:
	python setup.py sdist

install:
	python setup.py install
