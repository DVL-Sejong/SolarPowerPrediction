from src.util import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plant_number = 126
spot_index = 174
str_date = "20190820"
date = datetime.strptime(str_date, "%Y%m%d")
duration = 100

fig, ax = plt.subplots()

correlations, power_data, weather_data_list = get_pearson_correlations(date, plant_number, spot_index, duration)
correlations = np.array(correlations)
correlations = correlations.reshape((1, len(correlations)))
print(correlations.shape)

ax.matshow(correlations, cmap=plt.cm.Blues)

for i, correlation in enumerate(correlations[0]):
    print(correlation)
    ax.text(i, 0, str(correlation), va='center', ha='center')


plt.show()
