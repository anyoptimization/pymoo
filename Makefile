.PHONY: clean
clean:
	rm -rf build dist pymoo.egg-info

.PHONY: clean-ext
clean-ext:
	rm -f pymoo/functions/compiled/*.c
	rm -f pymoo/functions/compiled/*.so
	rm -f pymoo/functions/compiled/*.cpp
	rm -f pymoo/functions/compiled/*.html

.PHONY: compile
compile:
	python setup.py build_ext --inplace

.PHONY: dist
dist:
	python setup.py sdist

.PHONY: install
install:
	python setup.py install

