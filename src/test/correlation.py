import argparse

from src.constant import FeatureType
from src.loader import Power, Weather
from src.model import *
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


def get_pearson_correlations(args, power_data, weather_data):
    correlations = []
    for feature in args.features:
        feature_data = weather_data[feature]
        corr, _ = pearsonr(power_data, feature_data)
        correlations.append(corr)

    return correlations


def plot_correlations(correlations):
    correlations = np.array(correlations)
    correlations = correlations.reshape((1, len(correlations)))

    fig, ax = plt.subplots()
    ax.matshow(correlations, cmap=plt.cm.Blues)

    for i, correlation in enumerate(correlations[0]):
        ax.text(i, 0, "%.03f" % correlation, va='center', ha='center')

    plt.show()


if __name__ == '__main__':
    # ====== Random Seed Initialization ====== #
    seed = 1234
    tf.random.set_seed(seed)

    parser = argparse.ArgumentParser()
    args = parser.parse_args("")
    args.exp_name = "solar power prediction using weather features"

    # ====== Data Loading ====== #
    args.plant = 126
    args.spot = 174
    args.start_date = "20190820"
    args.end_date = "20200809"
    args.batch_size = 64
    args.x_frames = 3
    args.y_frames = 1

    # ====== Model Capacity ===== #
    args.hid_dim = 256

    # ====== Optimizer & Training ====== #
    args.optim = 'RMSprop'
    args.activation = 'relu'
    args.lss = 'MSE'
    args.lr = 0.001
    args.epochs = 256
    args.early_stop = 30
    args.evaluation = 'NRMSE'

    # ====== Experiment Variable ====== #
    name_var1 = 'features'

    # ====== Setting Attributes ====== #
    features = [feature for feature in FeatureType]
    setattr(args, name_var1, features)

    power_loader = Power(args)
    power_data = power_loader.get_data()

    weather_loader = Weather(args)
    weather_data = weather_loader.get_data()

    correlations = get_pearson_correlations(args, power_data, weather_data)
    plot_correlations(correlations)

    for i in range(len(correlations)):
        print(features[i].value, correlations[i])
