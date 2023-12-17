# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 12:08:15 2022

@author: ninad gaikwad 
"""

# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import numpy as np
import pandas as pd
import scipy.io
import re
import shutil
import matplotlib.pyplot as plt

# Custom Modules

# Testing Matplotlib
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
x = range(0,10)
y = [t**2 for t in x]
z = [t**2+1 for t in x]
plt.plot(x, y, label = r'$\beta=\alpha^2$')
plt.plot(x, z, label = r'$\beta=\alpha^2+1$')
plt.xlabel(r'$\alpha$')
plt.xticks(rotation=30)
plt.ylabel(r'$\beta$')
plt.legend(loc=0)
plt.savefig('test_plot.png')
plt.show()

plt.plot(datetimelist_1[0:10],range(10))
plt.xticks(rotation=30)
plt.show()
# plt.rcParams['text.usetex'] = True


t = np.linspace(0.0, 1.0, 100)
s = np.cos(4 * np.pi * t) + 2

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
ax.plot(t, s)



ax.set_xlabel(r'\textbf{time (s)}')
ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}', fontsize=16)
ax.set_title(r'\TeX\ is Number $\sum_{n=1}^\infty'
              r'\frac{-e^{i\pi}}{2^n}$!', fontsize=16, color='r')

plt.savefig('test_plot.png')
plt.show()