from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(['primes_python.py',        # Cython code file with primes() function
                           'primes_python_cy.pyx'],  # Python code file with primes_python_compiled() function
                          annotate=True),        # enables generation of the html annotation file
)