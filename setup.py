#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='gnosis',
    version='0.1.0',
    description="Code for 'Does Knowledge Distillation Really Work?'",
    author='Samuel Stanton, Pavel Izmailov, and Polina Kirichenko',
    author_email='ss13641@nyu.edu',
    url='https://github.com/samuelstanton/gnosis.git',
    packages=find_packages(include=['gnosis', 'gnosis.*', 'cka', 'cka.*']),
)
