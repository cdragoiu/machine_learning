from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
from process_data import DataLabels

def plot_features_vs_target(file_name, features, target, save_tag='features_vs_target'):
    data = pd.read_csv(file_name)
    fig = plt.figure(figsize=(7.0, 7.0))
    for i in range(len(features)):
        axs = plt.subplot(4, 4, i + 1)
        x = data[target]
        y = data[features[i]]
        axs.hist2d(x, y, bins=50, range=[(x.min(), x.max()), (y.min(), y.max())],
                   cmap='rainbow', norm=LogNorm(), alpha=0.7)
        axs.text(0.03, 0.97, features[i] + ' vs ' + target,
            ha='left', va='top', fontsize=6.0, color='#d00', transform=plt.gca().transAxes)
    plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.04, wspace=0.4, hspace=0.2)
    plt.savefig(file_name.replace('data', 'plots').replace('.csv', '_' + save_tag))
    plt.close()

if __name__ == '__main__':
    data_labels = DataLabels()
    features = data_labels.features
    target = data_labels.target
    plot_features_vs_target('data/training.csv', features, target)
    plot_features_vs_target('data/training_sampled.csv', features, target)
    plot_features_vs_target('data/training_sampled_processed.csv', features, target)
    plot_features_vs_target('data/validation.csv', features, target)
    plot_features_vs_target('data/validation_processed.csv', features, target)
    features = data_labels.features_ratio
    plot_features_vs_target('data/training_sampled_processed.csv', features, target,
                            'features_ratio_vs_target')
