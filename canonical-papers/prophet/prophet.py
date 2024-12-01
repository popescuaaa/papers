"""
Facebook Prophet

Generalized additive models: modeling a linear function with splines

from:

f(x) = sum_{i=1}^{n} alpha_i * x_i  to
f(x) = sum_{i=1}^{n} s_i * x_i  to

where the s_i represents an arbitrary spline ( function ).

Basically prophet is a generalized additive model with 5 different components:

1. Trend
2. Seasonality
3. Holidays
4. Autoregressive
5. Error

"""
import numpy as np
import pandas as pd
from sklearn.datasets import l

class Prophet:
	def __init__(self, input_data, horizon: int) -> None:
		self.horizon = horizon
		self.input_data = input_data

	def trend(self,) -> None:
		pass

	def seasonality(self,) -> None:
		pass

	def holidays(self,) -> None:
		pass

