import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from models import random_forest
from process_data import DataLabels, DataProcessor

def grid_search(train_file_name, valid_file_name):
    data_labels = DataLabels()
    train_features = pd.read_csv(train_file_name)
    train_target = train_features.pop(data_labels.target)
    valid_features = pd.read_csv(valid_file_name)
    valid_target = valid_features.pop(data_labels.target)
    data_processor = DataProcessor()
    results = []
    best_result_index = 0
    for n_estimators in [32, 64, 128]:
        for min_samples_leaf in [2, 4, 8]:
            for max_samples in [0.5, 0.7, 0.9]:
                print('>>> testing {} : {} : {}'.format(
                    n_estimators, min_samples_leaf, max_samples))
                model = random_forest(n_estimators, min_samples_leaf, max_samples)
                model.fit(train_features, train_target)
                train_predic = model.predict(train_features)
                valid_predic = model.predict(valid_features)
                train_target_raw = data_processor.recover_target(train_target)
                train_predic_raw = data_processor.recover_target(train_predic)
                valid_target_raw = data_processor.recover_target(valid_target)
                valid_predic_raw = data_processor.recover_target(valid_predic)
                train_r2 = r2_score(train_target_raw, train_predic_raw)
                train_mse = mean_squared_error(train_target_raw, train_predic_raw)
                valid_r2 = r2_score(valid_target_raw, valid_predic_raw)
                valid_mse = mean_squared_error(valid_target_raw, valid_predic_raw)
                results.append({'n_estimators': n_estimators,
                                'min_samples_leaf': min_samples_leaf,
                                'max_samples': max_samples,
                                'train_r2': train_r2,
                                'train_mse': train_mse,
                                'valid_r2': valid_r2,
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
