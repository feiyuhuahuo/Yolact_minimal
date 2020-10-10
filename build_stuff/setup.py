#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize("cython_nms.pyx"),
      include_dirs=[np.get_include()])
