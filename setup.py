#!/usr/bin/env python
# -*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: Ruifeng96150
# Mail: ruifeng96150@163.com
# Created Time:  2018-10-21 19:17:34
#############################################


from setuptools import setup, find_packages

setup(
    name="easy_sklearn",
    version="0.1.4",
    keywords=("sklearn", "python", "machine learning"),
    description="This is a python library base on sklearn",
    long_description="This is a python library, which can be easier to build sklearn classification and regressor models.",
    license="MIT Licence",

    url="https://github.com/ruifeng96150/easy_sklearn.git",
    author="ruifeng96150",
    author_email="ruifeng96150@163.com",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python',
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ],

    package_dir={'': '.'},
    packages=find_packages('.'),
    include_package_data=True,
    platforms="any",
    install_requires=['numpy', 'pandas', 'scikit-learn']
)
