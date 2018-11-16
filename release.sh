#!/bin/bash

# remove old wheels
sudo rm -rf dist/*

# Build Python 2 & 3 wheels for current version
sudo python2 setup.py sdist bdist_wheel
sudo python3 setup.py sdist bdist_wheel

# Upload to PyPI with twine. Needs full "skymind" credentials in ~/.pypirc
twine upload dist/*