from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math

scaler = StandardScaler()

# These functions will be passed a single column to be processed by its own process

# this function takes a numerical column, imputes the mean into empty areas, and standard scales it
def scale_nums(series):
    series = series.fillna(series.mean())
    return pd.Series(np.transpose(scaler.fit_transform(pd.DataFrame(series)))[0])

# scale_cats takes a categorical dummy column and divides by the sqrt of the probability of the class
def scale_cats(series):
    series /= math.sqrt((sum(series) / len(series)))
    return series
