import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

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

joblib.dump(scaler, 'scaler.save')

history = autoencoder.fit(data_normalized, data_normalized, epochs=150, shuffle=True, validation_split=0.2)

reconstructed_data = autoencoder.predict(data_normalized)

reconstruction_error = np.mean(np.power(data_normalized - reconstructed_data, 2), axis=1)

print(reconstruction_error[:5])

threshold = 0.033

anomalies = np.sum(reconstruction_error > threshold)
non_anomalies = np.sum(reconstruction_error < threshold)

print(f"anomalias: {anomalies}")
print(f"não_anomalias: {non_anomalies}")

from scapy.all import sniff, TCP, UDP
import numpy as np

def normalize_packet(packet):
    packet_size = packet[0]
    src_port = packet[1]
    dst_port = packet[2]

    normalized_packet = scaler.transform([[packet_size, src_port, dst_port]])
    return normalized_packet

def real_time_analysis(packet):
    packet_size = len(packet)
    src_port = None
    dst_port = None

    if packet.haslayer(TCP):
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif packet.haslayer(UDP):
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    if packet_size < 1200:
        normalized_packet = normalize_packet((packet_size, src_port, dst_port))

        reconstructed_packet = autoencoder.predict(normalized_packet)

        reconstruction_error = np.mean(np.power(normalized_packet - reconstructed_packet, 2))

        threshold = 0.11
        if reconstruction_error > threshold:
            print(f"Anomalia detectada: {packet_size} bytes, srcport={src_port}, dstport={dst_port}, erro de reconstrução={reconstruction_error:.4f}")
        else:
            print(f"Pacote normal: {packet_size} bytes, srcport={src_port}, dstport={dst_port}, erro de reconstrução={reconstruction_error:.4f}")

interface = 'enX0'
sniff(iface=interface, prn=real_time_analysis, store=0)