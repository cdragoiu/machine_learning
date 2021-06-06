import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import pandas as pd
from plot_features import yaxis_tkr_format
from process_data import DataLabels

def plot_target(file_name, target):
    data = pd.read_csv(file_name)
    fig = plt.figure(figsize=(4.0, 3.0))
    axs = plt.subplot(1, 1, 1)
    x = data[target]
    axs.hist(x, bins=50, range=(x.min(), x.max()), color='#0b6', alpha=0.7)
    axs.set_xlabel(target + ' Level')
    axs.yaxis.set_major_formatter(tkr.FuncFormatter(yaxis_tkr_format))
    plt.subplots_adjust(left=0.13, right=0.95, top=0.93, bottom=0.15)
    plt.savefig(file_name.replace('data', 'plots').replace('.csv', '_target'))
    plt.close()

if __name__ == '__main__':
    data_labels = DataLabels()
    target = data_labels.target
    plot_target('data/training.csv', target)
    plot_target('data/training_sampled.csv', target)
    plot_target('data/training_sampled_processed.csv', target)
    plot_target('data/validation.csv', target)
    plot_target('data/validation_processed.csv', target)
