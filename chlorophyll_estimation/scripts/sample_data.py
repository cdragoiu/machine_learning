import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def sample_data(file_name, max_samples=1566):
    data = pd.read_csv(file_name)
    data = shuffle(data)
    data = data.reset_index()
    data.pop('index')
    sampled_data_dict = {}
    counter = -1
    bins = np.zeros(50, dtype=int)
    for i in range(data.shape[0]):
        bin_index = math.floor(0.5 * data[data.columns[-1]][i])
        if bins[bin_index] >= max_samples:
            continue
        bins[bin_index] += 1
        counter += 1
        sampled_data_dict[counter] = data.iloc[i].to_dict()
    sampled_data = pd.DataFrame.from_dict(sampled_data_dict, orient='index')
    sampled_data.to_csv(file_name.replace('.csv', '_sampled.csv'), index=False)

if __name__ == '__main__':
    sample_data('data/training.csv')
