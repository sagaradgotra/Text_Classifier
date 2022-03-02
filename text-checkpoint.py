# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:48:48 2022

@author: sagar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_20newsgroups

data=fetch_20newsgroups()
print(data.target_names) 


