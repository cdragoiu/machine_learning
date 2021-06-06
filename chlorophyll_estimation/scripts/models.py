import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def random_forest(n_estimators, min_samples_leaf, max_samples, verbose=0):
    return RandomForestRegressor(n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf,
                                 max_samples=max_samples,
                                 random_state=0,
                                 n_jobs=4,
                                 verbose=verbose)

def neural_network(features, units, regularization, dropout, learning_rate, verbose=False):
    model = keras.Sequential([
        keras.Input(shape=(features,)),
        keras.layers.Dense(
            units=units,
            activation=keras.activations.relu,
            kernel_regularizer=keras.regularizers.l2(regularization)
        ),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(
            units=units,
            activation=keras.activations.relu,
            kernel_regularizer=keras.regularizers.l2(regularization)
        ),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate),
                  loss=keras.losses.MeanSquaredError())
    if verbose:
        model.summary()
    return model

def plot_results(target, prediction, save_name):
    fig = plt.figure(figsize=(4.0, 4.0))
    axs = plt.subplot(1, 1, 1)
    axs.plot(target, prediction, color='#b06', alpha=0.7, marker='o', linewidth=0.0)
    axs.set_xlabel('True CHL')
    axs.set_ylabel('Predicted CHL')
    axs.set_xlim(0.0, 50.0)
    axs.set_ylim(0.0, 50.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    mse = mean_squared_error(target, prediction)
    axs.text(0.97, 0.97, 'MSE = {:.2f}'.format(mse),
        ha='right', va='top', fontsize=10.0, color='#00d', transform=plt.gca().transAxes)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.12)
    plt.savefig('plots/' + save_name)
    plt.close()

def plot_loss(history, save_name):
    fig = plt.figure(figsize=(4.0, 4.0))
    axs = plt.subplot(1, 1, 1)
    axs.plot(history['loss'], color='#00b', alpha=0.7, linewidth=2, label='training')
    axs.plot(history['val_loss'], color='#b00', alpha=0.7, linewidth=2, label='validation')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('MSE')
    plt.grid(True)
    plt.legend(frameon=False)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    plt.savefig('plots/' + save_name)
    plt.close()
