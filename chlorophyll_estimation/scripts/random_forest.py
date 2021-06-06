import numpy as np
import pandas as pd
from models import random_forest, plot_results
from process_data import DataLabels, DataProcessor

def run_model(train_file_name, valid_file_name, test_file_name):
    data_labels = DataLabels()
    train_features = pd.read_csv(train_file_name)
    train_target = train_features.pop(data_labels.target)
    valid_features = pd.read_csv(valid_file_name)
    valid_target = valid_features.pop(data_labels.target)
    test_features = pd.read_csv(test_file_name)
    model = random_forest(n_estimators=64, min_samples_leaf=4, max_samples=0.5, verbose=2)
    model.fit(train_features, train_target)
    valid_predic = model.predict(valid_features)
    data_processor = DataProcessor()
    valid_target_raw = data_processor.recover_target(valid_target)
    valid_predic_raw = data_processor.recover_target(valid_predic)
    plot_results(valid_target_raw, valid_predic_raw, 'random_forest_results')
    predictions = []
    for estimator in model.estimators_:
        test_predic = estimator.predict(test_features)
        test_predic_raw = data_processor.recover_target(test_predic)
        predictions.append(test_predic_raw)
    predictions = np.array(predictions, dtype=float)
    predic_mean = predictions.mean(axis=0)
    predic_std = predictions.std(axis=0)
    results = pd.DataFrame({'CHL_LEVEL': predic_mean, 'UNCERTAINTY': predic_std})
    results.to_csv('results/random_forest_results.csv', index=False)

if __name__ == '__main__':
    run_model('data/training_sampled_processed.csv',
              'data/validation_processed.csv',
              'data/testing_processed.csv')
