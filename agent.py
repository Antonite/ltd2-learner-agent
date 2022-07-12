import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup
import glob
import sample_generator as sg

# dont print annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


units_file_url = "../ltd2-game-parser/units.csv"
unitDf = pd.read_csv(units_file_url, dtype={"units": str, "leak": int})
unitDf.fillna('', inplace=True)

sends_file_url = "../ltd2-game-parser/sends.csv"
sendsDf = pd.read_csv(sends_file_url, dtype={"sends": str, "leak": int})
sendsDf.fillna('', inplace=True)

waves_file_url = "../ltd2-game-parser/waves.csv"
waveDf = pd.read_csv(waves_file_url, dtype={"wave": int, "leak": int})


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("leak")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


# Categorical Datasets
waves_ds = dataframe_to_dataset(waveDf)
units_ds = dataframe_to_dataset(unitDf)
sends_ds = dataframe_to_dataset(sendsDf)


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


# Categorical features encoded as integers
wave = keras.Input(shape=(1,), name="wave", dtype="int64")

# Categorical feature encoded as string
sends = keras.Input(shape=(1,), name="sends", dtype="string")

all_inputs = [
    wave,
    sends,
]

# Integer categorical features
wave_encoded = encode_categorical_feature(wave, "wave", waves_ds, False)

# String categorical features
sends_encoded = encode_categorical_feature(sends, "sends", sends_ds, True)


features = [
    wave_encoded,
    sends_encoded
]

for col in ["0.5|0.5", "0.5|1", "0.5|1.5", "0.5|10", "0.5|10.5", "0.5|11", "0.5|11.5", "0.5|12", "0.5|12.5", "0.5|13", "0.5|13.5", "0.5|2", "0.5|2.5", "0.5|3", "0.5|3.5", "0.5|4", "0.5|4.5", "0.5|5", "0.5|5.5", "0.5|6", "0.5|6.5", "0.5|7", "0.5|7.5", "0.5|8", "0.5|8.5", "0.5|9", "0.5|9.5", "1.5|0.5", "1.5|1", "1.5|1.5", "1.5|10", "1.5|10.5", "1.5|11", "1.5|11.5", "1.5|12", "1.5|12.5", "1.5|13", "1.5|13.5", "1.5|2", "1.5|2.5", "1.5|3", "1.5|3.5", "1.5|4", "1.5|4.5", "1.5|5", "1.5|5.5", "1.5|6", "1.5|6.5", "1.5|7", "1.5|7.5", "1.5|8", "1.5|8.5", "1.5|9", "1.5|9.5", "1|0.5", "1|1", "1|1.5", "1|10", "1|10.5", "1|11", "1|11.5", "1|12", "1|12.5", "1|13", "1|13.5", "1|2", "1|2.5", "1|3", "1|3.5", "1|4", "1|4.5", "1|5", "1|5.5", "1|6", "1|6.5", "1|7", "1|7.5", "1|8", "1|8.5", "1|9", "1|9.5", "2.5|0.5", "2.5|1", "2.5|1.5", "2.5|10", "2.5|10.5", "2.5|11", "2.5|11.5", "2.5|12", "2.5|12.5", "2.5|13", "2.5|13.5", "2.5|2", "2.5|2.5", "2.5|3", "2.5|3.5", "2.5|4", "2.5|4.5", "2.5|5", "2.5|5.5", "2.5|6", "2.5|6.5", "2.5|7", "2.5|7.5", "2.5|8", "2.5|8.5", "2.5|9", "2.5|9.5", "2|0.5", "2|1", "2|1.5", "2|10", "2|10.5", "2|11", "2|11.5", "2|12", "2|12.5", "2|13", "2|13.5", "2|2", "2|2.5", "2|3", "2|3.5", "2|4", "2|4.5", "2|5", "2|5.5", "2|6", "2|6.5", "2|7", "2|7.5", "2|8", "2|8.5", "2|9", "2|9.5", "3.5|0.5", "3.5|1", "3.5|1.5", "3.5|10", "3.5|10.5", "3.5|11", "3.5|11.5", "3.5|12", "3.5|12.5", "3.5|13", "3.5|13.5", "3.5|2", "3.5|2.5", "3.5|3", "3.5|3.5", "3.5|4", "3.5|4.5", "3.5|5", "3.5|5.5", "3.5|6", "3.5|6.5", "3.5|7", "3.5|7.5", "3.5|8", "3.5|8.5", "3.5|9", "3.5|9.5", "3|0.5", "3|1", "3|1.5", "3|10", "3|10.5", "3|11", "3|11.5", "3|12", "3|12.5", "3|13", "3|13.5", "3|2", "3|2.5", "3|3", "3|3.5", "3|4", "3|4.5", "3|5", "3|5.5", "3|6", "3|6.5", "3|7", "3|7.5", "3|8", "3|8.5", "3|9", "3|9.5", "4.5|0.5", "4.5|1", "4.5|1.5", "4.5|10", "4.5|10.5", "4.5|11", "4.5|11.5", "4.5|12", "4.5|12.5", "4.5|13", "4.5|13.5", "4.5|2", "4.5|2.5", "4.5|3", "4.5|3.5", "4.5|4", "4.5|4.5", "4.5|5", "4.5|5.5", "4.5|6", "4.5|6.5", "4.5|7", "4.5|7.5", "4.5|8", "4.5|8.5", "4.5|9", "4.5|9.5", "4|0.5", "4|1", "4|1.5", "4|10", "4|10.5", "4|11", "4|11.5", "4|12", "4|12.5", "4|13", "4|13.5", "4|2", "4|2.5", "4|3", "4|3.5", "4|4", "4|4.5", "4|5", "4|5.5", "4|6", "4|6.5", "4|7", "4|7.5", "4|8", "4|8.5", "4|9", "4|9.5", "5.5|0.5", "5.5|1", "5.5|1.5", "5.5|10", "5.5|10.5", "5.5|11", "5.5|11.5", "5.5|12", "5.5|12.5", "5.5|13", "5.5|13.5", "5.5|2", "5.5|2.5", "5.5|3", "5.5|3.5", "5.5|4", "5.5|4.5", "5.5|5", "5.5|5.5", "5.5|6", "5.5|6.5", "5.5|7", "5.5|7.5", "5.5|8", "5.5|8.5", "5.5|9", "5.5|9.5", "5|0.5", "5|1", "5|1.5", "5|10", "5|10.5", "5|11", "5|11.5", "5|12", "5|12.5", "5|13", "5|13.5", "5|2", "5|2.5", "5|3", "5|3.5", "5|4", "5|4.5", "5|5", "5|5.5", "5|6", "5|6.5", "5|7", "5|7.5", "5|8", "5|8.5", "5|9", "5|9.5", "6.5|0.5", "6.5|1", "6.5|1.5", "6.5|10", "6.5|10.5", "6.5|11", "6.5|11.5", "6.5|12", "6.5|12.5", "6.5|13", "6.5|13.5", "6.5|2", "6.5|2.5", "6.5|3", "6.5|3.5", "6.5|4", "6.5|4.5", "6.5|5", "6.5|5.5", "6.5|6", "6.5|6.5", "6.5|7", "6.5|7.5", "6.5|8", "6.5|8.5", "6.5|9", "6.5|9.5", "6|0.5", "6|1", "6|1.5", "6|10", "6|10.5", "6|11", "6|11.5", "6|12", "6|12.5", "6|13", "6|13.5", "6|2", "6|2.5", "6|3", "6|3.5", "6|4", "6|4.5", "6|5", "6|5.5", "6|6", "6|6.5", "6|7", "6|7.5", "6|8", "6|8.5", "6|9", "6|9.5", "7.5|0.5", "7.5|1", "7.5|1.5", "7.5|10", "7.5|10.5", "7.5|11", "7.5|11.5", "7.5|12", "7.5|12.5", "7.5|13", "7.5|13.5", "7.5|2", "7.5|2.5", "7.5|3", "7.5|3.5", "7.5|4", "7.5|4.5", "7.5|5", "7.5|5.5", "7.5|6", "7.5|6.5", "7.5|7", "7.5|7.5", "7.5|8", "7.5|8.5", "7.5|9", "7.5|9.5", "7|0.5", "7|1", "7|1.5", "7|10", "7|10.5", "7|11", "7|11.5", "7|12", "7|12.5", "7|13", "7|13.5", "7|2", "7|2.5", "7|3", "7|3.5", "7|4", "7|4.5", "7|5", "7|5.5", "7|6", "7|6.5", "7|7", "7|7.5", "7|8", "7|8.5", "7|9", "7|9.5", "8.5|0.5", "8.5|1", "8.5|1.5", "8.5|10", "8.5|10.5", "8.5|11", "8.5|11.5", "8.5|12", "8.5|12.5", "8.5|13", "8.5|13.5", "8.5|2", "8.5|2.5", "8.5|3", "8.5|3.5", "8.5|4", "8.5|4.5", "8.5|5", "8.5|5.5", "8.5|6", "8.5|6.5", "8.5|7", "8.5|7.5", "8.5|8", "8.5|8.5", "8.5|9", "8.5|9.5", "8|0.5", "8|1", "8|1.5", "8|10", "8|10.5", "8|11", "8|11.5", "8|12", "8|12.5", "8|13", "8|13.5", "8|2", "8|2.5", "8|3", "8|3.5", "8|4", "8|4.5", "8|5", "8|5.5", "8|6", "8|6.5", "8|7", "8|7.5", "8|8", "8|8.5", "8|9", "8|9.5"]:
    inp = keras.Input(shape=(1,), name=col, dtype="string")
    all_inputs.append(inp)
    enc = encode_categorical_feature(inp, "units", units_ds, True)
    features.append(enc)


all_features = layers.concatenate(features)

l1 = layers.Dense(308, activation="relu")(all_features)
l1 = layers.Dropout(0.5)(l1)
l2 = layers.Dense(308, activation="relu")(l1)
l2 = layers.Dropout(0.5)(l2)
output = layers.Dense(1, activation="sigmoid")(l2)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

checkpoint_path = "trainings/june152022.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.load_weights(checkpoint_path)

_, _, files = next(os.walk("data"))
file_count = len(files)

# * means all if need specific format then *.csv
list_of_files = glob.glob('models/june152022/*')
latest_file = max(list_of_files, key=os.path.getctime)
start = int(str.split(str.split(latest_file, "-")[1], ".")[0])
i = start

for e in range(100):
    print("Starting era {0}".format(e))
    while i < file_count+1:
        print("loading dataset {0}".format(i))
        file_url = "data/{0}.csv".format(i)
        dataframe = pd.read_csv(file_url, dtype={"wave": int, "sends": str, "leak": int, "0.5|0.5": str, "0.5|1": str, "0.5|1.5": str, "0.5|10": str, "0.5|10.5": str, "0.5|11": str, "0.5|11.5": str, "0.5|12": str, "0.5|12.5": str, "0.5|13": str, "0.5|13.5": str, "0.5|2": str, "0.5|2.5": str, "0.5|3": str, "0.5|3.5": str, "0.5|4": str, "0.5|4.5": str, "0.5|5": str, "0.5|5.5": str, "0.5|6": str, "0.5|6.5": str, "0.5|7": str, "0.5|7.5": str, "0.5|8": str, "0.5|8.5": str, "0.5|9": str, "0.5|9.5": str, "1.5|0.5": str, "1.5|1": str, "1.5|1.5": str, "1.5|10": str, "1.5|10.5": str, "1.5|11": str, "1.5|11.5": str, "1.5|12": str, "1.5|12.5": str, "1.5|13": str, "1.5|13.5": str, "1.5|2": str, "1.5|2.5": str, "1.5|3": str, "1.5|3.5": str, "1.5|4": str, "1.5|4.5": str, "1.5|5": str, "1.5|5.5": str, "1.5|6": str, "1.5|6.5": str, "1.5|7": str, "1.5|7.5": str, "1.5|8": str, "1.5|8.5": str, "1.5|9": str, "1.5|9.5": str, "1|0.5": str, "1|1": str, "1|1.5": str, "1|10": str, "1|10.5": str, "1|11": str, "1|11.5": str, "1|12": str, "1|12.5": str, "1|13": str, "1|13.5": str, "1|2": str, "1|2.5": str, "1|3": str, "1|3.5": str, "1|4": str, "1|4.5": str, "1|5": str, "1|5.5": str, "1|6": str, "1|6.5": str, "1|7": str, "1|7.5": str, "1|8": str, "1|8.5": str, "1|9": str, "1|9.5": str, "2.5|0.5": str, "2.5|1": str, "2.5|1.5": str, "2.5|10": str, "2.5|10.5": str, "2.5|11": str, "2.5|11.5": str, "2.5|12": str, "2.5|12.5": str, "2.5|13": str, "2.5|13.5": str, "2.5|2": str, "2.5|2.5": str, "2.5|3": str, "2.5|3.5": str, "2.5|4": str, "2.5|4.5": str, "2.5|5": str, "2.5|5.5": str, "2.5|6": str, "2.5|6.5": str, "2.5|7": str, "2.5|7.5": str, "2.5|8": str, "2.5|8.5": str, "2.5|9": str, "2.5|9.5": str, "2|0.5": str, "2|1": str, "2|1.5": str, "2|10": str, "2|10.5": str, "2|11": str, "2|11.5": str, "2|12": str, "2|12.5": str, "2|13": str, "2|13.5": str, "2|2": str, "2|2.5": str, "2|3": str, "2|3.5": str, "2|4": str, "2|4.5": str, "2|5": str, "2|5.5": str, "2|6": str, "2|6.5": str, "2|7": str, "2|7.5": str, "2|8": str, "2|8.5": str, "2|9": str, "2|9.5": str, "3.5|0.5": str, "3.5|1": str, "3.5|1.5": str, "3.5|10": str, "3.5|10.5": str, "3.5|11": str, "3.5|11.5": str, "3.5|12": str, "3.5|12.5": str, "3.5|13": str, "3.5|13.5": str, "3.5|2": str, "3.5|2.5": str, "3.5|3": str, "3.5|3.5": str, "3.5|4": str, "3.5|4.5": str, "3.5|5": str, "3.5|5.5": str, "3.5|6": str, "3.5|6.5": str, "3.5|7": str, "3.5|7.5": str, "3.5|8": str, "3.5|8.5": str, "3.5|9": str, "3.5|9.5": str, "3|0.5": str, "3|1": str, "3|1.5": str, "3|10": str, "3|10.5": str, "3|11": str, "3|11.5": str, "3|12": str, "3|12.5": str, "3|13": str, "3|13.5": str, "3|2": str, "3|2.5": str, "3|3": str, "3|3.5": str, "3|4": str, "3|4.5": str, "3|5": str, "3|5.5": str, "3|6": str, "3|6.5": str, "3|7": str, "3|7.5": str, "3|8": str, "3|8.5": str, "3|9": str, "3|9.5": str, "4.5|0.5": str, "4.5|1": str, "4.5|1.5": str, "4.5|10": str, "4.5|10.5": str, "4.5|11": str, "4.5|11.5": str, "4.5|12": str, "4.5|12.5": str, "4.5|13": str, "4.5|13.5": str, "4.5|2": str, "4.5|2.5": str, "4.5|3": str, "4.5|3.5": str, "4.5|4": str, "4.5|4.5": str, "4.5|5": str, "4.5|5.5": str, "4.5|6": str, "4.5|6.5": str, "4.5|7": str, "4.5|7.5": str, "4.5|8": str, "4.5|8.5": str, "4.5|9": str, "4.5|9.5": str, "4|0.5": str, "4|1": str, "4|1.5": str, "4|10": str, "4|10.5": str, "4|11": str, "4|11.5": str, "4|12": str, "4|12.5": str,
                                "4|13": str, "4|13.5": str, "4|2": str, "4|2.5": str, "4|3": str, "4|3.5": str, "4|4": str, "4|4.5": str, "4|5": str, "4|5.5": str, "4|6": str, "4|6.5": str, "4|7": str, "4|7.5": str, "4|8": str, "4|8.5": str, "4|9": str, "4|9.5": str, "5.5|0.5": str, "5.5|1": str, "5.5|1.5": str, "5.5|10": str, "5.5|10.5": str, "5.5|11": str, "5.5|11.5": str, "5.5|12": str, "5.5|12.5": str, "5.5|13": str, "5.5|13.5": str, "5.5|2": str, "5.5|2.5": str, "5.5|3": str, "5.5|3.5": str, "5.5|4": str, "5.5|4.5": str, "5.5|5": str, "5.5|5.5": str, "5.5|6": str, "5.5|6.5": str, "5.5|7": str, "5.5|7.5": str, "5.5|8": str, "5.5|8.5": str, "5.5|9": str, "5.5|9.5": str, "5|0.5": str, "5|1": str, "5|1.5": str, "5|10": str, "5|10.5": str, "5|11": str, "5|11.5": str, "5|12": str, "5|12.5": str, "5|13": str, "5|13.5": str, "5|2": str, "5|2.5": str, "5|3": str, "5|3.5": str, "5|4": str, "5|4.5": str, "5|5": str, "5|5.5": str, "5|6": str, "5|6.5": str, "5|7": str, "5|7.5": str, "5|8": str, "5|8.5": str, "5|9": str, "5|9.5": str, "6.5|0.5": str, "6.5|1": str, "6.5|1.5": str, "6.5|10": str, "6.5|10.5": str, "6.5|11": str, "6.5|11.5": str, "6.5|12": str, "6.5|12.5": str, "6.5|13": str, "6.5|13.5": str, "6.5|2": str, "6.5|2.5": str, "6.5|3": str, "6.5|3.5": str, "6.5|4": str, "6.5|4.5": str, "6.5|5": str, "6.5|5.5": str, "6.5|6": str, "6.5|6.5": str, "6.5|7": str, "6.5|7.5": str, "6.5|8": str, "6.5|8.5": str, "6.5|9": str, "6.5|9.5": str, "6|0.5": str, "6|1": str, "6|1.5": str, "6|10": str, "6|10.5": str, "6|11": str, "6|11.5": str, "6|12": str, "6|12.5": str, "6|13": str, "6|13.5": str, "6|2": str, "6|2.5": str, "6|3": str, "6|3.5": str, "6|4": str, "6|4.5": str, "6|5": str, "6|5.5": str, "6|6": str, "6|6.5": str, "6|7": str, "6|7.5": str, "6|8": str, "6|8.5": str, "6|9": str, "6|9.5": str, "7.5|0.5": str, "7.5|1": str, "7.5|1.5": str, "7.5|10": str, "7.5|10.5": str, "7.5|11": str, "7.5|11.5": str, "7.5|12": str, "7.5|12.5": str, "7.5|13": str, "7.5|13.5": str, "7.5|2": str, "7.5|2.5": str, "7.5|3": str, "7.5|3.5": str, "7.5|4": str, "7.5|4.5": str, "7.5|5": str, "7.5|5.5": str, "7.5|6": str, "7.5|6.5": str, "7.5|7": str, "7.5|7.5": str, "7.5|8": str, "7.5|8.5": str, "7.5|9": str, "7.5|9.5": str, "7|0.5": str, "7|1": str, "7|1.5": str, "7|10": str, "7|10.5": str, "7|11": str, "7|11.5": str, "7|12": str, "7|12.5": str, "7|13": str, "7|13.5": str, "7|2": str, "7|2.5": str, "7|3": str, "7|3.5": str, "7|4": str, "7|4.5": str, "7|5": str, "7|5.5": str, "7|6": str, "7|6.5": str, "7|7": str, "7|7.5": str, "7|8": str, "7|8.5": str, "7|9": str, "7|9.5": str, "8.5|0.5": str, "8.5|1": str, "8.5|1.5": str, "8.5|10": str, "8.5|10.5": str, "8.5|11": str, "8.5|11.5": str, "8.5|12": str, "8.5|12.5": str, "8.5|13": str, "8.5|13.5": str, "8.5|2": str, "8.5|2.5": str, "8.5|3": str, "8.5|3.5": str, "8.5|4": str, "8.5|4.5": str, "8.5|5": str, "8.5|5.5": str, "8.5|6": str, "8.5|6.5": str, "8.5|7": str, "8.5|7.5": str, "8.5|8": str, "8.5|8.5": str, "8.5|9": str, "8.5|9.5": str, "8|0.5": str, "8|1": str, "8|1.5": str, "8|10": str, "8|10.5": str, "8|11": str, "8|11.5": str, "8|12": str, "8|12.5": str, "8|13": str, "8|13.5": str, "8|2": str, "8|2.5": str, "8|3": str, "8|3.5": str, "8|4": str, "8|4.5": str, "8|5": str, "8|5.5": str, "8|6": str, "8|6.5": str, "8|7": str, "8|7.5": str, "8|8": str, "8|8.5": str, "8|9": str, "8|9.5": str})
        dataframe.fillna('', inplace=True)

        val_dataframe = dataframe.sample(frac=0.2, random_state=1)
        train_dataframe = dataframe.drop(val_dataframe.index)

        train_ds = dataframe_to_dataset(train_dataframe)
        val_ds = dataframe_to_dataset(val_dataframe)

        train_ds = train_ds.batch(4096)
        val_ds = val_ds.batch(4096)

        model.fit(train_ds, epochs=5, validation_data=val_ds,
                  callbacks=[cp_callback], verbose=1)

        if i == file_count:
            start = 0
        i += 1


# while True:
#     print("enter a test")
#     input()
#     input_dict = {name: tf.convert_to_tensor(
#         [value]) for name, value in sg.generateSample().items()}
#     predictions = model.predict(input_dict)

#     print(
#         "%.1f%% chance to leak " % (100 * predictions[0][0],)
#     )

#     print(predictions)
