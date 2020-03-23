from distutils.core import setup
from setuptools import find_packages

setup(
    name='cfm',
    version='0.1.1',
    packages=find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
)