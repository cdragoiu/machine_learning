import numpy as np
import pandas as pd

class DataLabels:
    def __init__(self):
        self.features = ['WL400', 'WL412', 'WL442', 'WL490',
                         'WL510', 'WL560', 'WL620', 'WL665',
                         'WL673', 'WL681', 'WL708', 'WL753',
                         'WL761', 'WL764', 'WL767', 'WL778']
        self.features_ratio = ['WL778/WL761', 'WL778/WL681', 'WL778/WL620', 'WL778/WL490',
                               'WL764/WL708', 'WL764/WL665', 'WL764/WL510', 'WL764/WL412',
                               'WL753/WL673', 'WL753/WL560', 'WL753/WL442', 'WL681/WL620',
                               'WL681/WL490', 'WL665/WL510', 'WL665/WL412', 'WL560/WL442']
        self.target = 'CHL'
        self.ratio = '/'

class DataProcessor:
    def __init__(self):
        self.feature_scale_factor = 100.0
        self.target_scale_factor = 20.0

    def transform_feature(self, data):
        return np.log(self.feature_scale_factor * data)

    def transform_target(self, data):
        return np.log(data / self.target_scale_factor)

    def recover_target(self, data):
        return self.target_scale_factor * np.exp(data)

def augment_data(data):
    data_labels = DataLabels()
    feature_size = len(data_labels.features)
    target_in = data_labels.target in data.columns
    for i in range(feature_size - 1):
        for j in range(i + 1, feature_size):
            index = data.columns.size - target_in
            name = data_labels.features[j] + data_labels.ratio + data_labels.features[i]
            value = data[data_labels.features[j]] / data[data_labels.features[i]]
            data.insert(index, name, value)

def transform_data(data):
    data_labels = DataLabels()
    data_processor = DataProcessor()
    for label in data.columns:
        if label == data_labels.target:
            continue
        data[label] = data_processor.transform_feature(data[label])
    if data_labels.target in data.columns:
        data[data_labels.target] = data_processor.transform_target(data[data_labels.target])

def process_data(file_name):
    data = pd.read_csv(file_name)
    augment_data(data)
    transform_data(data)
    data.to_csv(file_name.replace('.csv', '_processed.csv'), index=False)

if __name__ == '__main__':
    process_data('data/training_sampled.csv')
    process_data('data/validation.csv')
    process_data('data/testing.csv')
