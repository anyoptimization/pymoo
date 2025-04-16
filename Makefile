.PHONY: clean
clean:
	rm -rf build dist pymoo.egg-info

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

