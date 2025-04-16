.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	find . -name '*.egg-info' -exec rm -rf {} +

.PHONY: clean-ext
clean-ext:
	rm -f pymoo/cython/*.c
	rm -f pymoo/cython/*.so
	rm -f pymoo/cython/*.cpp
	rm -f pymoo/cython/*.html

.PHONY: compile
compile:
	python setup.py build_ext --inplace

.PHONY: dist
dist:
	python setup.py sdist

.PHONY: install
install:
	python setup.py install

