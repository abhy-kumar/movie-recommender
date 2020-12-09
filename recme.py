import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens()

print(repr(data['train']))
print(repr(data['test']))
