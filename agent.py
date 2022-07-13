import os

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers

import glob
import sample_generator as sg

# dont print annoying warnings
from utils import Utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

units_file_url = "../ltd2-game-parser/units.csv"
unitDf = pd.read_csv(units_file_url, dtype={"units": str, "leak": int})
unitDf.fillna('', inplace=True)

sends_file_url = "../ltd2-game-parser/sends.csv"
sendsDf = pd.read_csv(sends_file_url, dtype={"sends": str, "leak": int})
sendsDf.fillna('', inplace=True)

waves_file_url = "../ltd2-game-parser/waves.csv"
waveDf = pd.read_csv(waves_file_url, dtype={"wave": int, "leak": int})

checkpoint_path = "trainings/june152022.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Categorical Datasets
waves_ds = Utils.dataframe_to_dataset(waveDf)
units_ds = Utils.dataframe_to_dataset(unitDf)
sends_ds = Utils.dataframe_to_dataset(sendsDf)

# Categorical features encoded as integers
wave = keras.Input(shape=(1,), name="wave", dtype="int64")

# Categorical feature encoded as string
sends = keras.Input(shape=(1,), name="sends", dtype="string")

all_inputs = [
    wave,
    sends,
]

# Integer categorical features
wave_encoded = Utils.encode_categorical_feature(wave, "wave", waves_ds, False)

# String categorical features
sends_encoded = Utils.encode_categorical_feature(
    sends, "sends", sends_ds, True)

features = [
    wave_encoded,
    sends_encoded
]

for col in Utils.GAME_BOARD_COLUMNS:
    inp = keras.Input(shape=(1,), name=col, dtype="string")
    all_inputs.append(inp)
    enc = Utils.encode_categorical_feature(inp, "units", units_ds, True)
    features.append(enc)

all_features = layers.concatenate(features)

l1 = layers.Dense(308, activation="relu", kernel_regularizer=regularizers.L2(0.001))(all_features)
l1 = layers.Dropout(0.2)(l1)

l2 = layers.Dense(308, activation="relu", kernel_regularizer=regularizers.L2(0.001))(l1)
l2 = layers.Dropout(0.2)(l2)

output = layers.Dense(1, activation="sigmoid")(l2)
model = keras.Model(all_inputs, output)
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=opt)

model.load_weights(checkpoint_path)

_, _, files = next(os.walk("data"))
file_count = len(files)

# * means all if need specific format then *.csv
list_of_files = glob.glob('models/june152022/*')
latest_file = max(list_of_files, key=os.path.getctime)

i = int(str.split(str.split(latest_file, "-")[1], ".")[0])
for e in range(100):
    print("Starting era {0}".format(e))
    while i < file_count + 1:
        print("loading dataset {0}".format(i))
        file_url = "data/{0}.csv".format(i)
        dataframe = pd.read_csv(file_url, dtype=Utils.DTYPE_ARG_MAP)
        dataframe.fillna('', inplace=True)

        val_dataframe = dataframe.sample(frac=0.2, random_state=1)
        train_dataframe = dataframe.drop(val_dataframe.index)

        train_ds = Utils.dataframe_to_dataset(train_dataframe)
        val_ds = Utils.dataframe_to_dataset(val_dataframe)

        train_ds = train_ds.batch(32)
        val_ds = val_ds.batch(32)

        # model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=[], verbose=1)  # NO CHECKPOINTING **
        model.fit(train_ds, epochs=1, validation_data=val_ds,
                  callbacks=[cp_callback], verbose=1)
        i += 1

    i = 1  # Reset i to first value after first iteration

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
