import numpy as np
import pandas as pd
import tensorflow.keras as keras
from models import neural_network, plot_results, plot_loss
from process_data import DataLabels, DataProcessor

def run_model(train_file_name, valid_file_name, test_file_name):
    data_labels = DataLabels()
    train_features = pd.read_csv(train_file_name)
    train_target = train_features.pop(data_labels.target)
    valid_features = pd.read_csv(valid_file_name)
    valid_target = valid_features.pop(data_labels.target)
    test_features = pd.read_csv(test_file_name)
    for label in data_labels.features:
        train_features.pop(label)
        valid_features.pop(label)
        test_features.pop(label)
    model = neural_network(train_features.shape[1],
                           units=256,
                           regularization=0.01,
                           dropout=0.1,
                           learning_rate=0.00001,
                           verbose=True)
    history = model.fit(train_features,
                        train_target,
                        batch_size=30,
                        epochs=120,
                        verbose=1,
                        validation_data=(valid_features, valid_target))
    valid_predic = model.predict(valid_features).flatten()
    data_processor = DataProcessor()
    valid_target_raw = data_processor.recover_target(valid_target)
    valid_predic_raw = data_processor.recover_target(valid_predic)
    plot_loss(history.history, 'neural_network_loss')
    plot_results(valid_target_raw, valid_predic_raw, 'neural_network_results')

if __name__ == '__main__':
    run_model('data/training_sampled_processed.csv',
              'data/validation_processed.csv',
              'data/testing_processed.csv')
