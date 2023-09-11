Contributing
============
``pint-xarray`` is developed on `github <https://github.com/xarray-contrib/pint-xarray>`_.

Commit message tags
-------------------
By default, the upstream dev CI is disabled on pull request and push events. You can
override this behavior per commit by adding a <tt>[test-upstream]</tt> tag to the first
line of the commit message.

Linters / Autoformatters
------------------------
In order to keep code consistent, we use

- `Black <https://black.readthedocs.io/en/stable/>`_ for standardized code formatting
- `blackdoc <https://blackdoc.readthedocs.io/en/stable/>`_ for standardized code formatting in documentation
- `Flake8 <https://flake8.pycqa.org/en/latest/>`_ for general code quality
- `isort <https://github.com/PyCQA/isort>`_ for standardized order in imports. See also `flake8-isort <https://github.com/gforcada/flake8-isort>`_.
