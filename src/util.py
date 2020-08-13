from src.loader import *
from datetime import datetime, timedelta, time
from matplotlib.dates import DateFormatter
from matplotlib import pyplot
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.constant import *
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input, LSTM, Dense, RepeatVector
import tensorflow as tf
from keras.models import Sequential
import math
import argparse


def plot_day_power(plant_number, date, duration=1):
    day_power = get_power_data(plant_number, date, duration)
    x_value = [date + timedelta(hours=i) for i in range(duration * 24)]
    y_value = [value for value in day_power]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_value, y_value)
    ax.set(xlabel="Time",
           ylabel="Solar Power",
           title="UR00000%s %s" % (str(plant_number), date.strftime("%Y-%m-%d")))

    date_form = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    plt.show()


def get_power_data(plant_number, date, duration=1):
    json_data = read_json(plant_number, date)['hrPow']
    for i in range(1, duration):
        date = date + timedelta(days=1)
        new_data = read_json(plant_number, date)['hrPow']
        json_data = json_data.append(new_data, ignore_index=True)

    power_data = json_data.interpolate(method='linear')
    power_data = normalize(power_data.to_numpy())
    return power_data


def get_weather_data(spot_index, date, feature_type, duration=1):
    csv_data = read_csv(spot_index, date, duration=duration)
    weather_data = interpolate_weather(date, csv_data, feature_type)
    weather_data = normalize(weather_data.to_numpy())
    weather_data = np.nan_to_num(weather_data)

    return weather_data


def plot_day_weather(spot_index, date, feature_type, duration=1):
    weather_data = get_weather_data(spot_index, date, feature_type, duration=duration)

    x_value = [date + timedelta(hours=i) for i in range(duration * 24)]
    unit = feature_type.value
    unit = unit[unit.find("(") + 1:unit.find(")")]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_value, weather_data)
    ax.set(xlabel="Time",
           ylabel=unit,
           title="%s %s" % (date.strftime("%Y-%m-%d"), feature_type.name))

    date_form = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    plt.show()


def insert_row(idx, df, df_insert):
    begin = df.iloc[:idx, ]
    end = df.iloc[idx:, ]
    df_insert = pd.DataFrame(list(df_insert.values()), columns=['일시'])
    return pd.concat([begin, df_insert, end])


def interpolate_weather(date, csv_data, feature_type):
    str_date = date.strftime("%Y-%m-%d %H:%M")

    for i in range(23):
        if csv_data.iloc[i:i+1]['일시'].values[0] != str_date:
            row = {'일시': str_date}
            csv_data = insert_row(i, csv_data, row)
        date = date + timedelta(hours=1)
        str_date = date.strftime("%Y-%m-%d %H:%M")

    weather_data = csv_data[feature_type].interpolate(method='linear')
    return weather_data


def normalize(data):
    normalized = (data-min(data))/(max(data)-min(data))
    return normalized


def get_pearson_correlations(date, plant_number, spot_index, duration=1):
    power_data = get_power_data(plant_number, date, duration=duration)
    attributes = [attribute.value for attribute in FeatureType]

    correlations = []
    weather_data_list = []
    for attribute in attributes:
        weather_data = get_weather_data(spot_index, date, attribute, duration=duration)
        all_zeros = not np.any(weather_data)
        if all_zeros:
            corr = np.nan
        else:
            corr, _ = pearsonr(power_data, weather_data)
        correlations.append(corr)
        weather_data_list.append(weather_data)

    return correlations, power_data, weather_data_list


def plot_correlations(correlations, power_data, weather_data_list):
    pyplot.scatter(power_data, weather_data_list[0])
    pyplot.show()


def get_input_data(plant_number, spot_index, starting_date, features, duration):
    input_data = np.empty([len(features) + 1, 24 * duration])

    input_data[0] = get_power_data(plant_number, starting_date, duration)
    for i in range(0, len(features)):
        input_data[i + 1] = get_weather_data(spot_index, starting_date, features[i].value, duration)

    return input_data


# def get_model():
#     with tf.device('/GPU:0'):
#         seq = Sequential()
#         seq.add(LSTM(256, return_state=True))
#         seq.add(RepeatVector(24))
#         seq.add(LSTM(24))
#         seq.add(Dense(24, activation='relu'))
#
#         rmsprop = RMSprop(lr=0.001)
#         seq.compile(optimizer=rmsprop, loss=tf.keras.losses.MeanSquaredError)
#
#         return seq
#
#
# def train(trainset, valset, epochs, batsh_size):
#     X_train = trainset[0]
#     y_train = trainset[1]
#     X_val = valset[0]
#     y_val = valset[1]
#
#     seq = get_model()
#     seq.fit(X_train, y_train,
#             validation_data=(X_val, y_val),
#             epochs=epochs, batch_size=batsh_size,
#             shuffle='batch')
#
#     seq.summary()
#
#     score = seq.evaluate(X_val, y_val)
#     return score


def generate_model(num_features, latent_dim):
    with tf.device('/GPU:0'):
        encoder_inputs = Input(shape=(num_features, 72))

        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        repeat_vector = RepeatVector(24)
        repeat_vector_outputs = repeat_vector(encoder_outputs)

        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(repeat_vector_outputs, initial_state=encoder_states)

        decoder_dense = Dense(1, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        rmsprop = RMSprop(lr=0.001)


        model = Model(encoder_inputs, decoder_outputs)
        model.compile(optimizer=rmsprop, loss=tf.keras.losses.MeanSquaredError(), metrics='accuracy')

    return model


def train(X_train, y_train, X_val, y_val, num_features, latent_dim):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)
    history = generate_model(num_features, latent_dim)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[callback])
    model.summary()
    model.save('models.h5')

    return history


class Dataset:
    def __init__(self, plant_number, spot_index, feature_types, start, end):
        self.plant_number = plant_number
        self.spot_index = spot_index
        self.feature_types = feature_types
        self.start = datetime.strptime(start, "%Y%m%d")
        self.end = datetime.strptime(end, "%Y%m%d")
        self.x_frames = 3
        self.y_frames = 1
        self.initialize()

    def initialize(self):
        self.duration = (self.end - self.start).days + 1
        self.data = get_input_data(self.plant_number, self.spot_index,
                                   self.start, self.feature_types, self.duration)

    def get_durations(self):
        train_duration = math.floor(self.duration * 0.75)

        val_duration = math.floor(self.duration * 0.125)
        test_duration = math.floor(self.duration * 0.125)
        return train_duration, val_duration, test_duration

    def get_dates(self):
        train_duration, val_duration, test_duration = self.get_durations()
        train_start = self.start
        train_end = train_start + timedelta(days=train_duration - 1)
        val_start = train_end + timedelta(days=1)
        val_end = val_start + timedelta(days=val_duration - 1)
        test_start = val_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_duration - 1)
        return [train_start, train_end], [val_start, val_end], [test_start, test_end]

    def get_item(self, start):
        index = (start - self.start).days * 24
        X = [self.data[i+1][index:index+(self.x_frames * 24)] for i in range(len(self.feature_types))]
        y = self.data[0][index+(self.x_frames * 24):index+((self.x_frames + self.y_frames) * 24)]
        return np.asarray(X), y

    def get_items(self, start, end):
        len_samples = (end - start).days + 1 - (self.x_frames + self.y_frames) + 1

        X = []
        y = []
        for i in range(len_samples):
            X_item, y_item = self.get_item(start + timedelta(i))
            X.append(X_item)
            y.append(y_item)

        return np.asarray(X), np.asarray(y)

    def get_dataset(self):
        train, val, test = self.get_dates()
        X_train, y_train = self.get_items(train[0], train[1])
        X_val, y_val = self.get_items(val[0], val[1])
        X_test, y_test = self.get_items(test[0], test[1])

        return [X_train, y_train], [X_val, y_val], [X_test, y_test]


plant_number = 126
spot_index = 174
str_date = "20190820"
date = datetime.strptime(str_date, "%Y%m%d")
duration = 100

start = "20190820"
end = "20190825"

epochs = 256
batch_size = 64

feature_types = [FeatureType.TEMPERATURE,
                 FeatureType.SUNSHINE]

dataset = Dataset(plant_number, spot_index, feature_types, start, end)
train_set, validation_set, test_set = dataset.get_dataset()
X_train, y_train = train_set
X_val, y_val = validation_set
X_test, y_test = test_set

model = generate_model(X_train, y_train, X_val, y_val, len(feature_types), 256)

# train(train_set, validation_set, epochs, batch_size)

#
# attributes = [attribute.value for attribute in FeatureType]
# correlations, power_data, weather_data_list = get_pearson_correlations(date, plant_number, spot_index, duration)
#
# # plot_correlations(correlations, power_data, weather_data_list)
#
# input_data = get_input_data(plant_number, spot_index, date, feature_types, 100)
# print(input_data.shape)

