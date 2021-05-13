from pathlib import Path
from datetime import timedelta

from constant import FeatureType
from loader import Power, Weather

from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error

import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


class Model:
    def __init__(self, scaler):
        self.scaler = scaler

    def train(self, args, dataset):
        X_train, y_train = dataset['train']
        X_val, y_val = dataset['val']
        X_test, y_test = dataset['test']

        with tf.device('/GPU:0'):
            model = Sequential()
            optimizer = RMSProp(learning_rate=args.learning_rate)

            model.add(LSTM(args.vector_size, input_shape=(args.frame_in, args.feature_len)))
            model.add(RepeatVector(args.frame_out))
            model.add(LSTM(args.vector_size, return_sequences=True))
            model.add(TimeDistributed(Dense(args.vector_size, activation='relu')))
            model.add(TimeDistributed(Dense(1)))
            model.compile(loss='mse', optimizer=optimizer)

            model_path = os.path.join(args.root, 'results', args.experiment_name, args.name)
            Path(model_path).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(model_path, 'model-{epoch:03d}-{val_loss:03f}.h5')
            model_saving_path = os.path.join(model_path, 'model.h5')

            callback = EarlyStopping(monitor='val_loss', patience=args.patience)
            checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_loss', save_best_only=True)

            history = model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
                                validation_data=(X_val, y_val), callbacks=[callback, checkpoint], shuffle=args.shuffle)
            model.save(model_saving_path)

            print('train_score')
            train_score = self.evaluate(model, X_train, y_train)
            print('val_score')
            val_score = self.evaluate(model, X_val, y_val)
            print('test_score')
            test_score = self.evaluate(model, X_test, y_test)
            count_params = model.count_params()
            print('count_params: %d' % count_params)

            self.write_result(model_path, train_score, val_score, test_score, count_params)
            self.write_args(model_path, args)

            return model

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

    def save_result(self, y_test, y_pred_list, args):
        y_test = y_test.reshape((y_test.shape[0] * y_test.shape[1], y_test.shape[2]))
        y_test = self.scaler.inverse_transform(y_test)
        y_test = y_test.reshape((y_test.shape[0]))

        for i in range(len(y_pred_list)):
            y_pred_list[i] = y_pred_list[i].reshape(-1, 1)
            y_pred_list[i] = self.scaler.inverse_transform(y_pred_list[i])
            y_pred_list[i] = y_pred_list[i].reshape((y_pred_list[i].shape[0]))

        df = pd.DataFrame()
        for i in range(len(args.features)):
            df['%dth model' % (i + 1)] = y_pred_list[i].tolist()
        df['y_test'] = y_test.tolist()

        full_idx = pd.date_range(start=args.test_start + timedelta(days=3), end=args.test_end, freq='H')
        full_idx = full_idx[:y_test.shape[0]]
        df['time'] = full_idx
        df = df.set_index('time')

        result_path = os.path.join(args.root, 'results', args.experiment_name, 'result.csv')
        df.to_csv(result_path)

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

    def write_args(self, path, args):
        print('Saving %s' % os.path.join(path, 'setting.csv'))
        columns = []
        for arg in vars(args):
            columns.append(arg)

        arguments = pd.DataFrame(index=[0], columns=columns)
        for arg in vars(args):
            arguments.loc[0][arg] = getattr(args, arg)
            print('argument %s, %s' % (arg, arguments.loc[0][arg]))

        arguments.to_csv(os.path.join(path, 'setting.csv'), index=False)

    def read_setting(self, args):
        setting_path = os.path.join(args.root, 'results', args.automl_name, args.name)
        setting_path = os.path.join(setting_path, 'best_hyper_parameter.json')

        with open(setting_path) as json_file:
            setting = json.load(json_file)

        setattr(args, 'batch_size', setting['batch_size'])
        setattr(args, 'shuffle', setting['shuffle'])
        setattr(args, 'vector_size', setting['vector_size'])

        return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    # ====== Path ====== #
    args.root = Path(os.getcwd())
    args.experiment_name = "LSTM-003"
    args.automl_name = 'automl-003'

    # ====== Model ====== #
    args.frame_in = 72
    args.frame_out = 24
    args.epochs = 200
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
    weather = Weather(args, features)

    power_data = power.get_data()
    model_manager = Model(power_data['scaler'])
    setattr(args, 'test_start', power.test_start)
    setattr(args, 'test_end', power.test_end)

    y_pred_list = []
    results = []
    for i in range(len(features)):
        weather_data = weather.get_data(i + 1)
        setattr(args, 'feature_len', i + 1)
        setattr(args, 'name', 'feature-%02d' % (i + 1))

        train = [weather_data['train'], power_data['train']]
        val = [weather_data['val'], power_data['val']]
        test = [weather_data['test'], power_data['test']]

        dataset = {'train': train, 'val': val, 'test': test}
        args = model_manager.read_setting(args)
        model = model_manager.train(args, dataset)
        y_pred = model.predict(dataset['test'][0])
        y_pred_list.append(y_pred)
        result = model_manager.evaluate(model, dataset['test'][0], dataset['test'][1])
        results.append(result)

    setattr(args, 'features', features)
    model_manager.save_result(dataset['test'][1], y_pred_list, args)
