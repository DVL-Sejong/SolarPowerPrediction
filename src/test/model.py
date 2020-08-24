import argparse

from src.constant import FeatureType
from src.loader import Power, Weather, Loader
from src.model import *
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt


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
    args.sequence_len = 72
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
    list_var1 = [FeatureType.SUNSHINE,
                 FeatureType.HUMIDITY,
                 FeatureType.WIND_SPEED,
                 FeatureType.VISIBILITY,
                 FeatureType.GROUND_TEMPERATURE,
                 FeatureType.WIND_DIRECTION,
                 FeatureType.STEAM_PRESSURE,
                 FeatureType.TEMPERATURE,
                 FeatureType.PRECIPITATION,
                 FeatureType.DEW_POINT_TEMPERATURE,
                 FeatureType.ATMOSPHERIC_PRESSURE]

    for i in range(len(list_var1)):
        sub_list = list_var1[:i + 1]

        setattr(args, name_var1, str(sub_list))
        print(args)

        setattr(args, name_var1, sub_list)
        loader = Loader(args)
        dataset = loader.get_dataset()
        break
