import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

file = pd.read_csv('packets.csv')
print(file.head())

from sklearn.preprocessing import MinMaxScaler

data = file[['Tamanho do Pacote (bytes)', 'Porta de Origem', 'Porta de Destino']]

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
print("----------------------------")
print(data_normalized[:5])

input_dim = data_normalized.shape[1]
encoding_dim = 2

input_layer= tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')
print("----------------------------")
print(autoencoder.summary())

history = autoencoder.fit(data_normalized, data_normalized, epochs=150, shuffle=True, validation_split=0.2)

autoencoder.save('autoencoder.keras')