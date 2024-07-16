import tensorflow as tf
import pandas as pd
import numpy as np

file = pd.read_csv('packets.csv')
print(file.head())
 
from sklearn.preprocessing import MinMaxScaler

data = file[['Tamanho do Pacote (bytes)', 'Porta de Origem', 'Porta de Destino']]

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
print("----------------------------")
print(data_normalized[:5])

autoencoder = tf.keras.models.load_model('autoencoder.keras')

#model.summary()

reconstructed_data = autoencoder.predict(data_normalized)

reconstruction_error = np.mean(np.power(data_normalized - reconstructed_data, 2), axis=1)

print(reconstruction_error[:5])

threshold = 0.15

anomalies = np.sum(reconstruction_error > threshold)
non_anomalies = np.sum(reconstruction_error < threshold)

print(f"anomalias: {anomalies}")
print(f"não_anomalias: {non_anomalies}")