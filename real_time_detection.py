from scapy.all import sniff, TCP, UDP
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

autoencoder = tf.keras.models.load_model('autoencoder.keras')
scaler = joblib.load('scaler.save')

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