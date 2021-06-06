import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pandas as pd
from process_data import DataLabels

def xaxis_tkr_format(val, pos):
    if abs(val) < 1e-6:
        return '0'
    return '{:.2f}'.format(val)

def yaxis_tkr_format(val, pos):
    if abs(val) < 1e-6:
        return '0'
    if val < 1e3:
        return '{:.0f}'.format(val)
    return '{:.0f}K'.format(val * 1e-3)

def plot_features(file_name, features, save_tag='features'):
    data = pd.read_csv(file_name)
    fig = plt.figure(figsize=(7.0, 7.0))
    for i in range(len(features)):
        axs = plt.subplot(4, 4, i + 1)
        x = data[features[i]]
        axs.hist(x, bins=50, range=(x.min(), x.max()), color='#06b', alpha=0.7)
        axs.xaxis.set_major_formatter(tkr.FuncFormatter(xaxis_tkr_format))
        axs.yaxis.set_major_formatter(tkr.FuncFormatter(yaxis_tkr_format))
        axs.text(0.03, 0.97, features[i],
            ha='left', va='top', fontsize=8.0, color='#d00', transform=plt.gca().transAxes)
    plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.04, wspace=0.4, hspace=0.2)
    plt.savefig(file_name.replace('data', 'plots').replace('.csv', '_' + save_tag))
    plt.close()

if __name__ == '__main__':
    data_labels = DataLabels()
    features = data_labels.features
    plot_features('data/training.csv', features)
    plot_features('data/training_sampled.csv', features)
    plot_features('data/training_sampled_processed.csv', features)
    plot_features('data/validation.csv', features)
    plot_features('data/validation_processed.csv', features)
    plot_features('data/testing.csv', features)
    plot_features('data/testing_processed.csv', features)
    features = data_labels.features_ratio
    plot_features('data/training_sampled_processed.csv', features, 'features_ratio')
