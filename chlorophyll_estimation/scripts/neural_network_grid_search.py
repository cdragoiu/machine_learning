import pandas as pd
from sklearn.metrics import mean_squared_error
from models import neural_network
from process_data import DataLabels, DataProcessor

def grid_search(train_file_name, valid_file_name):
    data_labels = DataLabels()
    train_features = pd.read_csv(train_file_name)
    train_target = train_features.pop(data_labels.target)
    valid_features = pd.read_csv(valid_file_name)
    valid_target = valid_features.pop(data_labels.target)
    for label in data_labels.features:
        train_features.pop(label)
        valid_features.pop(label)
    data_processor = DataProcessor()
    results = []
    best_result_index = 0
    for units in [64, 128, 256]:
        for regularization in [0.1, 0.01, 0.001]:
            for dropout in [0.1, 0.3, 0.5]:
                for learning_rate in [0.001, 0.0001, 0.00001]:
                    model = neural_network(
                        train_features.shape[1], units, regularization, dropout, learning_rate)
                    print('>>> testing {} : {} : {} : {}'.format(
                        units, regularization, dropout, learning_rate))
                    history = model.fit(train_features, train_target,
                                        batch_size=30,
                                        epochs=90,
                                        verbose=0,
                                        validation_data=(valid_features, valid_target))
                    train_predic = model.predict(train_features).flatten()
                    valid_predic = model.predict(valid_features).flatten()
                    train_target_raw = data_processor.recover_target(train_target)
                    train_predic_raw = data_processor.recover_target(train_predic)
                    valid_target_raw = data_processor.recover_target(valid_target)
                    valid_predic_raw = data_processor.recover_target(valid_predic)
                    train_loss = history.history['loss'][-1]
                    train_mse = mean_squared_error(train_target_raw, train_predic_raw)
                    valid_loss = history.history['val_loss'][-1]
                    valid_mse = mean_squared_error(valid_target_raw, valid_predic_raw)
                    results.append({'units': units,
                                    'regularization': regularization,
                                    'dropout': dropout,
                                    'learning_rate': learning_rate,
                                    'train_loss': train_loss,
                                    'train_mse': train_mse,
                                    'valid_loss': valid_loss,
                                    'valid_mse': valid_mse})
                    if valid_mse < results[best_result_index]['valid_mse']:
                        best_result_index = len(results) - 1
    return results, best_result_index

if __name__ == '__main__':
    results, best_result_index = grid_search('data/training_sampled_processed.csv',
                                             'data/validation_processed.csv')
    print('\n>>> Test Parameters <<<\n')
    for res in results:
        for key, val in res.items():
            print('{:<20s} : {}'.format(key, val))
        print()
    print('>>> Best Parameters <<<\n')
    for key, val in results[best_result_index].items():
        print('{:<20s} : {}'.format(key, val))
    print()
