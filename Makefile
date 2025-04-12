.PHONY: clean
clean: clean-ext clean-pyc
	rm -rf build/
	rm -rf dist/
	find . -name '*.egg-info' -exec rm -rf {} +

.PHONY: clean-ext
clean-ext:
	rm -f src/pymoo/cython/*.c
	rm -f src/pymoo/cython/*.so
	rm -f src/pymoo/cython/*.cpp
	rm -f src/pymoo/cython/*.html
	find . -name '*.so' -delete

.PHONY: clean-pyc
clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '__pycache__' -exec rm -rf {} +

.PHONY: compile
compile:
	python setup.py build_ext --inplace

.PHONY: dist
dist:
	python setup.py sdist

.PHONY: install
install:
	python setup.py install

