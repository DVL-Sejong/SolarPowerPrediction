import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB

import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

import logging

from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

from sklearn.metrics import mean_squared_error

from constant import FeatureType
from loader import Power, Weather

logging.basicConfig(level=logging.DEBUG)


class SolarWorker(Worker):
    def __init__(self,  *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_interval = sleep_interval

    def set_dataset(self, args, dataset, scaler):
        self.args = args
        self.dataset = dataset
        self.scaler = scaler

    def compute(self, config, budget, **kwargs):
        print("compute")

        X_train, y_train = self.dataset['train']
        X_val, y_val = self.dataset['val']
        X_test, y_test = self.dataset['test']

        setattr(self.args, 'batch_size', config['batch_size'])
        setattr(self.args, 'vector_size', config['vector_size'])
        setattr(self.args, 'shuffle', config['shuffle'])
        setattr(self.args, 'epochs', int(budget))

        with tf.device('/GPU:0'):
            model = Sequential()
            optimizer = RMSProp(learning_rate=self.args.learning_rate)

            model.add(LSTM(config['vector_size'], input_shape=(self.args.frame_in, self.args.feature_len)))
            model.add(RepeatVector(self.args.frame_out))
            model.add(LSTM(config['vector_size'], return_sequences=True))
            model.add(TimeDistributed(Dense(config['vector_size'], activation='relu')))
            model.add(TimeDistributed(Dense(1)))
            model.compile(loss='mse', optimizer=optimizer)

            model_path = os.path.join(self.args.root, 'results', self.args.experiment_name, self.args.name)
            Path(model_path).mkdir(parents=True, exist_ok=True)

            file_len = len([name for name in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, name))])
            model_path = os.path.join(model_path, 'model-%03d' % (file_len + 1))
            Path(model_path).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(model_path, 'model-{epoch:03d}-{val_loss:03f}.h5')
            model_saving_path = os.path.join(model_path, 'model.h5')

            callback = EarlyStopping(monitor='val_loss', patience=self.args.patience)
            checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_loss', save_best_only=True)

            history = model.fit(X_train, y_train, batch_size=config['batch_size'], epochs=int(budget),
                                validation_data=(X_val, y_val), callbacks=[callback, checkpoint], shuffle=config['shuffle'])
            model.save(model_saving_path)

            train_score = self.evaluate(model, X_train, y_train)
            val_score = self.evaluate(model, X_val, y_val)
            test_score = self.evaluate(model, X_test, y_test)
            count_params = model.count_params()
            self.write_result(model_path, train_score, val_score, test_score, count_params)
            self.write_args(model_path)

            print("train result")
            train_score = model.evaluate(X_train, y_train, verbose=0)
            print("validation result")
            val_score = model.evaluate(X_val, y_val, verbose=0)
            print("test result")
            test_score = model.evaluate(X_test, y_test, verbose=0)

            result = {
                'loss': 1 - val_score,
                'info': {
                    'test accuracy': test_score,
                    'train accuracy': train_score,
                    'validation accuracy': val_score,
                    'number of parameters': count_params,
                    'epochs': str(int(budget))
                }
            }

        return result

    def evaluate(self, model, X_data, y_data):
        y_data = y_data.reshape((y_data.shape[0] * y_data.shape[1], y_data.shape[2]))
        y_data = self.scaler.inverse_transform(y_data)

        prediction = model.predict(X_data)
        prediction = prediction.reshape((-1, 1))
        prediction = self.scaler.inverse_transform(prediction)

        zero_indices = np.where(y_data == 0)
        non_zero_indices = np.where(y_data != 0)
        y_data_adjusted = np.delete(y_data, zero_indices)
        y_pred_adjusted = np.delete(prediction, zero_indices)

        rmse = np.sqrt(mean_squared_error(y_data_adjusted, y_pred_adjusted))
        max_min = np.max(y_data_adjusted) - np.min(y_data_adjusted)
        nrmse = rmse / max_min

        pred_zero_indices = np.where(prediction == 0)
        pred_non_zero_indices = np.where(prediction != 0)
        true_positive = np.intersect1d(zero_indices, pred_zero_indices).size
        true_negative = np.intersect1d(non_zero_indices, pred_non_zero_indices).size
        false_positive = np.intersect1d(non_zero_indices, pred_zero_indices).size
        false_negative = np.intersect1d(zero_indices, pred_non_zero_indices).size

        print('in test datsaet, zeros: %d, non_zeros: %d' % (zero_indices[0].size, non_zero_indices[0].size))
        print('in prediction, zeros: %d, non_zeros: %d' % (pred_zero_indices[0].size, pred_non_zero_indices[0].size))

        print('true_positive: %d, true_negative: %d, false_positive: %d, false_negative: %d' %
              (true_positive, true_negative, false_positive, false_negative))

        all = true_positive + true_negative + false_positive + false_negative
        accuracy = (true_positive + true_negative) / all if all != 0 else -1
        true_predics = true_positive + false_positive
        precision = true_positive / true_predics if true_predics != 0 else -1
        true_cases = true_positive + false_negative
        recall = true_positive / true_cases if true_cases != 0 else -1
        if precision != -1 and recall != -1:
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            f1_score = -1

        print('nrmse: %lf, accuracy: %lf, f1_score: %lf' % (nrmse, accuracy, f1_score))

        result = dict()
        result.update({'nrmse': nrmse, 'accuracy': accuracy, 'f1_score': f1_score})
        print(result)
        return result

    @staticmethod
    def get_configspace():
        configuration_space = CS.ConfigurationSpace()

        # training configurations
        batch_size = CSH.CategoricalHyperparameter('batch_size', [1, 2, 4, 8, 16, 32, 64, 128])
        vector_size = CSH.CategoricalHyperparameter('vector_size', [16, 32, 64, 128, 256, 512, 1024])
        shuffle = CSH.CategoricalHyperparameter('shuffle', [True, False])

        configuration_space.add_hyperparameter(batch_size)
        configuration_space.add_hyperparameter(vector_size)
        configuration_space.add_hyperparameter(shuffle)

        print("configuration space ended")
        print(configuration_space)

        return configuration_space

    def write_result(self, path, train_score, val_score, test_score, count_params):
        columns = ['train_nrmse', 'train_accuracy', 'train_f1_score',
                   'val_nrmse', 'val_accuracy', 'val_f1_score',
                   'test_nrmse', 'test_accuracy', 'test_f1_score', 'count_params']

        result = pd.DataFrame(index=[0], columns=columns)
        result['train_nrmse'] = train_score['nrmse']
        result['train_accuracy'] = train_score['accuracy']
        result['train_f1_score'] = train_score['f1_score']
        result['val_nrmse'] = val_score['nrmse']
        result['val_accuracy'] = val_score['accuracy']
        result['val_f1_score'] = val_score['f1_score']
        result['test_nrmse'] = test_score['nrmse']
        result['test_accuracy'] = test_score['accuracy']
        result['test_f1_score'] = test_score['f1_score']
        result['count_params'] = count_params

        result.to_csv(os.path.join(path, 'result.csv'), index=False)
        print('Saving %s' % os.path.join(path, 'result.csv'))

    def write_args(self, path):
        print('Saving %s' % os.path.join(path, 'setting.csv'))
        columns = []
        for arg in vars(self.args):
            columns.append(arg)

        arguments = pd.DataFrame(index=[0], columns=columns)
        for arg in vars(self.args):
            arguments.loc[0][arg] = getattr(self.args, arg)
            print('argument %s, %s' % (arg, arguments.loc[0][arg]))

        arguments.to_csv(os.path.join(path, 'setting.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    # ====== Path ====== #
    args.root = Path(os.getcwd())
    args.experiment_name = "automl-003"

    # ====== Model ====== #
    args.frame_in = 72
    args.frame_out = 24
    args.nameserver = '127.0.0.1'
    args.n_iterations = 10
    args.min_budget = 32
    args.max_budget = 512
    args.learning_rate = 0.001
    args.patience = 30

    # ====== Data ====== #
    args.years = [2017, 2018, 2019]
    args.region = "Jindo"
    args.station = 192
    args.ratio = [0.6, 0.2, 0.2]

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
        setattr(args, 'feature_len', i + 1)
        setattr(args, 'name', 'feature-%02d' % (i + 1))

        train = [weather_data['train'], power_data['train']]
        val = [weather_data['val'], power_data['val']]
        test = [weather_data['test'], power_data['test']]

        dataset = {'train': train, 'val': val, 'test': test}

        print('New nameserver %s starts' % args.name)
        name_server = hpns.NameServer(run_id=args.name, host=args.nameserver, port=None)
        name_server.start()

        worker = SolarWorker(sleep_interval=0, nameserver=args.nameserver, run_id=args.name)
        worker.set_dataset(args, dataset, power_data['scaler'])
        worker.run(background=True)

        bohb = BOHB(configspace=worker.get_configspace(),
                    run_id=args.name, nameserver=args.nameserver,
                    min_budget=args.min_budget, max_budget=args.max_budget)

        result = bohb.run(n_iterations=args.n_iterations)

        bohb.shutdown(shutdown_workers=True)
        name_server.shutdown()

        id2config = result.get_id2config_mapping()
        incumbent = result.get_incumbent_id()

        print('Best found configurations:', id2config[incumbent]['config'])
        print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
        print('A total of %i runs where excuted.' % len(result.get_all_runs()))
        print('Total budget corresponds to %.1f full function evaluations.' % (
                    sum([r.budget for r in result.get_all_runs()]) / args.max_budget))

        best_path = os.path.join(args.root, 'results', args.experiment_name, args.name, 'best_hyper_parameter.json')
        with open(best_path, 'w') as file:
            json.dump(id2config[incumbent]['config'], file)
