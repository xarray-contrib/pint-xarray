#!/usr/bin/env bash
python -m pip install \
  -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
  --no-deps \
  --pre \
  --upgrade \
  numpy \
  scipy  # until `scipy` has released a version compatible with `numpy>=2.0`
python -m pip install --upgrade \
  git+https://github.com/hgrecco/pint \
  git+https://github.com/pydata/xarray
