from setuptools import setup, find_packages

setup(
    name='pint-xarray',
    version='0.0.1',
    author='Tom Nicholas',
    author_email='tomnicholas1@googlemail.com',
    description='Physical units interface to xarray using Pint',
    packages=find_packages(),
    url='https://github.com/TomNicholas/pint-xarray',
    install_requires=['numpy >= 1.17.1',
                      'xarray >= 0.15.1',
                      'pint >= 0.12'],
)
