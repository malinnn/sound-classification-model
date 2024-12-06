import os
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import modelReteaNeuronala
import tkinter as tk
from tkinter import Tk, filedialog
from tkinter.filedialog import askopenfilename

yamnet_model_handle = 'https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Convertire fisiere audio in float si resamplare audio
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def load_wav_for_map(filename, label, fold):
    return load_wav_16k_mono(filename), label, fold


def extract_embedding(wav_data, label, fold):
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    num_embeddings = tf.shape(embeddings)[0]
    return (embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings))


def plot_training_history(history):
    plt.figure(figsize=(10, 5))

    plt.plot(history.history['accuracy'], label='Acuratete')

    plt.plot(history.history['val_accuracy'], label='Acuratete validare')

    plt.xlabel('Epoca')
    plt.ylabel('Acuratete')
    plt.title('Antrenarea modelului de retea neuronala')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_history(history):
    plt.figure(figsize=(10, 5))

    plt.plot(history.history['loss'], label='Pierdere antrenare')

    plt.plot(history.history['val_loss'], label='Pierdere validare')

    plt.xlabel('Epoca')
    plt.ylabel('Pierdere')
    plt.title('Evoluția pierderii în timpul antrenării modelului')
    plt.legend()
    plt.grid(True)
    plt.show()

def augment_data(wav_data):
    noise = tf.random.normal(shape=tf.shape(wav_data), mean=0.0, stddev=0.005, dtype=tf.float32)
    wav_data = wav_data + noise
    return wav_data


# Setarea path-urilor pentru date
esc50_csv = 'set_de_date/ESC-50-master/meta/esc50.csv'
base_data_path = 'set_de_date/ESC-50-master/audio/'

pd_data = pd.read_csv(esc50_csv)
print(pd_data.head())

# Filtrarea datelor
unique_classes = pd_data['category'].unique()
map_class_to_id = {name: idx for idx, name in enumerate(unique_classes)}

class_id = pd_data['category'].apply(lambda name: map_class_to_id[name])
pd_data = pd_data.assign(target=class_id)

full_path = pd_data['filename'].apply(lambda row: os.path.join(base_data_path, row))
pd_data = pd_data.assign(filename=full_path)

print(pd_data.head(10))

# Incarcam fisierele audio si obtinem embedding-urile
filenames = pd_data['filename']
targets = pd_data['target']
folds = pd_data['fold']

main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))
print(main_ds.element_spec)

main_ds = main_ds.map(load_wav_for_map)
print(main_ds.element_spec)

# Augmentarea datelor
main_ds = main_ds.map(lambda wav, label, fold: (augment_data(wav), label, fold))
#print(main_ds.element_spec)

# Extractie embedding
main_ds = main_ds.map(extract_embedding).unbatch()
print(main_ds.element_spec)

# Impartirea setului de date
cached_ds = main_ds.cache()
train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 4)
val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)
test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 5)

remove_fold_column = lambda embedding, label, fold: (embedding, label)

train_ds = train_ds.map(remove_fold_column)
val_ds = val_ds.map(remove_fold_column)
test_ds = test_ds.map(remove_fold_column)

train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

# Crearea modelului
my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),                  # , kernel_regularizer=tf.keras.regularizers.l2(0.001)
    tf.keras.layers.Dense(len(unique_classes))
], name='my_model')


    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(512, activation='softmax'),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.BatchNormalization(),


my_model.summary()

# Compilarea modelului si inceperea antrenarii
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)   # 0.0005, 0.001
my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer=optimizer,
                 metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.0001,
                                                 patience=10)

print("Rata de invatare : ", optimizer.learning_rate, "\n")
history = my_model.fit(train_ds,
                       epochs=100,
                       validation_data=val_ds,
                       # callbacks=[reduce_lr]
                       )

loss, accuracy = my_model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

plot_training_history(history)
plot_loss_history(history)

"""
# Selectarea unui fisier pentru testare
audio_file_path = input("Calea fisierului audio : ")
testing_wav_data = load_wav_16k_mono(audio_file_path)

# Play the audio file.
display.Audio(testing_wav_data, rate=16000)

# Testarea modelului
scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
result = my_model(embeddings).numpy()

inferred_class = unique_classes[result.mean(axis=0).argmax()]
print(f'The main sound is: {inferred_class}') """


def select_audio_file():
    root = tk.Tk()
    # root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("WAV files", "*.wav")]
    )
    root.mainloop()

    return file_path


while True:
    audio_file_path = select_audio_file()

    if not audio_file_path:
        break

    testing_wav_data = load_wav_16k_mono(audio_file_path)

    display.display(display.Audio(testing_wav_data, rate=16000))

    # Testare model
    scores, embeddings, spectrogram = yamnet_model(testing_wav_data)
    result = my_model(embeddings).numpy()

    inferred_class = unique_classes[result.mean(axis=0).argmax()]
    print(f'The main sound is: {inferred_class}')


saved_model_path = './all_classes_yamnet'

input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                            trainable=False, name='yamnet')
_, embeddings_output, _ = embedding_extraction_layer(input_segment)
serving_outputs = my_model(embeddings_output)
serving_outputs = modelReteaNeuronala.ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
serving_model = tf.keras.Model(input_segment, serving_outputs)
serving_model.save(saved_model_path, include_optimizer=False)

tf.keras.utils.plot_model(serving_model)

# Reverificarea modelului
reloaded_model = tf.saved_model.load(saved_model_path)

reloaded_results = reloaded_model(testing_wav_data)
inferred_class = unique_classes[tf.math.argmax(reloaded_results)]
print(f'The main sound is: {inferred_class}')
