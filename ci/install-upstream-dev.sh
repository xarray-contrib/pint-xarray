#!/usr/bin/env bash
python -m pip install \
  -i https://pypi.anaconda.org/scipy-wheels-nightly/simple \
  --no-deps \
  --pre \
  --upgrade \
  numpy
python -m pip install --upgrade \
  git+https://github.com/hgrecco/pint \
  git+https://github.com/pydata/xarray
