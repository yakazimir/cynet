.PHONY: clean-pyc clean-build

help:
	@echo "clean-build -- remove build scaffolding"
	@echo "clean-pyc -- remove auxiliary python files"
	@echo "clean-cython -- remove cython auxiliary files"
	@echo "clean -- total cleaning of project files"
	@echo "build-ext - locally build and compile the cython sources"

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	rm -rf zubr_env
	rm -rf html_env

clean-pyc:
	find cynet/ -name '*.pyc' -exec rm -f {} +
	find cynet/ -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean-cython:
	find cynet/ -name '*.so' -exec rm -f {} +
	find cynet/ -name '*.c' -exec rm -f {} +

clean: clean-build clean-pyc clean-cython

build-ext:
	python setup.py build_ext --inplace --dynet=/media/sf_projects/cynet/dynet/ --eigen=/media/sf_projects/eigen --boost=/usr/include/
