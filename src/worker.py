import argparse
import hashlib
import json
import os
from pathlib import Path

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import keras
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

import logging

from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

from src.constant import FeatureType
from src.loader import Power, Weather

logging.basicConfig(level=logging.DEBUG)


class SolarWorker(Worker):
    def __init__(self,  args, dataset, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.dataset = dataset

    def compute(self, config, budget, **kwargs):
        print("compute")

        X_train, y_train = self.dataset['train']
        X_val, y_val = self.dataset['val']
        X_test, y_test = self.dataset['test']

        with K.tf_ops.device('/GPU:0'):
            model = Sequential()
            optimizer = RMSProp(learning_rate=config['learning_rate'])

            model.add(LSTM(256, input_shape=(self.args.frame_in, len(self.args.features))))
            model.add(RepeatVector(self.args.frame_out))
            model.add(LSTM(256, return_sequences=True))
            model.add(TimeDistributed(Dense(256, activation='relu')))
            model.add(TimeDistributed(Dense(1)))
            model.compile(loss='mse', optimizer=optimizer)

            hash_key = hashlib.sha1(str(self.args).encode()).hexdigest()[:6]
            print("model name: ", hash_key)

            model_root_path = os.path.join(self.args.root, 'models', self.args.name, hash_key)
            Path(model_root_path).mkdir(parents=True, exist_ok=True)
            checkpoint_path = os.path.join(model_root_path, 'model-{epoch:03d}-{val_loss:03f}.h5')
            model_path = os.path.join(model_root_path, '%s.h5' % 'hash_key')

            callback = EarlyStopping(monitor='val_loss', patience=config['patience'])
            checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_loss', save_best_only=True)

            history = model.fit(X_train, y_train, batch_size=config['batch_size'], epochs=budget,
                                validation_data=(X_val, y_val), callbacks=[callback, checkpoint], shuffle=config['shuffle'])
            model.save(model_path)

            train_score = model.evaluate(X_train, y_train, verbose=0)
            validation_score = model.evaluate(X_val, y_val, verbose=0)
            test_score = model.evaluate(X_test, y_test, verbose=0)
            count_params = model.count_params()

        result = {
            'loss': 1 - validation_score[1],
            'info': {
                'test accuracy': test_score[1],
                'train accuracy': train_score[1],
                'validation accuracy': validation_score[1],
                'number of parameters': count_params,
                'epochs': str(budget)
            }
        }

        file1 = json.dumps(config)
        file2 = json.dumps(result)
        self.write_result([file1, file2, "\n"])

        return result

    @staticmethod
    def get_configspace():
        configuration_space = CS.ConfigurationSpace()

        # training configurations
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1e-1,
                                                       default_value='1e-2', log=True)
        batch_size = CSH.CategoricalHyperparameter('batch_size', [1, 2, 4, 8, 16, 32, 64, 128])
        patience = CSH.CategoricalHyperparameter('patience', [10, 20, 30, 40])
        shuffle = CSH.CategoricalHyperparameter('shuffle', [True, False])

        configuration_space.add_hyperparameter(learning_rate)
        configuration_space.add_hyperparameter(batch_size)
        configuration_space.add_hyperparameter(patience)
        configuration_space.add_hyperparameter(shuffle)

        print("configuration space ended")
        print(configuration_space)

        return configuration_space

    def write_result(self, contents):
        path = os.path.join(self.args.root, 'results', 'automl', '%s setting.txt' % self.args.name)
        if not os.path.exists(path):
            os.mknod(path)

        with open(path, 'a+') as file:
            for content in contents:
                file.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    # ====== Path ====== #
    args.root = Path(os.getcwd()).parent

    # ====== Model ====== #
    args.frame_in = 72
    args.frame_out = 24

    # ====== Data ====== #
    args.years = [2017, 2018, 2019]
    args.region = "Jindo"
    args.station = 192

    # ====== Features ====== #
    features = [FeatureType.SUNSHINE,
                FeatureType.GROUND_TEMPERATURE,
                FeatureType.HUMIDITY,
                FeatureType.WIND_SPEED,
                FeatureType.WIND_DIRECTION,
                FeatureType.TEMPERATURE,
                FeatureType.VISIBILITY,
                FeatureType.PRECIPITATION,
                FeatureType.STEAM_PRESSURE,
                FeatureType.DEW_POINT_TEMPERATURE,
                FeatureType.ATMOSPHERIC_PRESSURE]

    power = Power(args)
    power_data = power.get_data()

    weather = Weather(args, features)

    for i in range(len(features)):
        weather_data = weather.get_data(i+1)
        setattr(object, 'name', 'model %d' % (i + 1))

        train = [power_data['train'], weather_data['train']]
        val = [power_data['val'], weather_data['val']]
        test = [power_data['test'], weather_data['test']]

        dataset = {'train': train, 'val': val, 'test': test}

        worker = SolarWorker(run_id='all cases', args=args, dataset=dataset)
        config_space = worker.get_configspace()
        config = config_space.sample_configuration().get_dictionary()
        print(config)
        result = worker.compute(config=config, budget=256)
        print(result)
