.PHONY: clean
clean:
	rm -rf dist/
	find . -name '*.egg-info' -exec rm -rf {} +
	rm -f pymoo/cython/*.c
	rm -f pymoo/cython/*.so
	rm -f pymoo/cython/*.cpp
	rm -f pymoo/cython/*.html
	find . -name '*.so' -delete

.PHONY: compile
compile:
	python setup.py build_ext --inplace

.PHONY: dist
dist:
	python setup.py sdist

.PHONY: install
install:
	python setup.py install

