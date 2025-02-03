import numpy as np
import matplotlib.pyplot as plt

print("LINE ENCODING TECHNIQUES")
input_data = input("INPUT DATA : ")
input_data = [int(bit) for bit in input_data]

clock_frequency = 1  
clock_period = 1 / clock_frequency
t = np.arange(0, len(input_data), 0.01)  
clock_signal = np.where((t % clock_period) < (clock_period / 2), 1, -1)

def manchester_encoding(clock_signal, input_data):
    encoded_signal = np.zeros_like(clock_signal)
    for i in range(len(input_data)):
        if input_data[i] == 1:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 0.5) * len(clock_signal) / len(input_data))] = -1
            encoded_signal[int((i + 0.5) * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = 1
        else:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 0.5) * len(clock_signal) / len(input_data))] = 1
            encoded_signal[int((i + 0.5) * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = -1
    return encoded_signal

def differential_manchester_encoding(clock_signal, input_data):
    encoded_signal = np.zeros(int(len(clock_signal)))
    b = np.zeros(int(len(clock_signal) / len(input_data)))
    b[0:int(0.5 * len(b))] = -1
    b[int(0.5 * len(b)):int(len(b))] = 1
    encoded_signal[0:int(len(b))] = b
    for i in range(1, len(input_data)):
        if input_data[i] == 1:
            b = b * (-1)
        encoded_signal[i*int(len(b)):(i+1)*int(len(b))] = b
    return encoded_signal

def ami_encoding(clock_signal, input_data):
    encoded_signal = np.zeros_like(clock_signal)
    b=(-1)
    for i in range(len(input_data)):
        if input_data[i] == 1:
            b=b*(-1)
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = b
        else:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = 0
    return encoded_signal

def psedoternary(clock_signal, input_data):
    encoded_signal = np.zeros_like(clock_signal)
    b=(-1)
    for i in range(len(input_data)):
        if input_data[i] == 0:
            b=b*(-1)
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = b
        else:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = 0
    return encoded_signal

def nrz_encoding(clock_signal, input_data):
    encoded_signal = np.zeros_like(clock_signal)
    for i in range(len(input_data)):
        if input_data[i] == 1:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = 1
        else:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = 0
    return encoded_signal

def nrz_l_encoding(clock_signal, input_data):
    encoded_signal = np.zeros_like(clock_signal)
    for i in range(len(input_data)):
        if input_data[i] == 1:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = -1
        else:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = 1
    return encoded_signal

def nrz_i_encoding(clock_signal, input_data):
    encoded_signal = np.zeros_like(clock_signal)
    b = (-1)
    for i in range(len(input_data)):
        if input_data[i] == 1:
            b=b*(-1)
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = b
        else:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = b
    return encoded_signal

def rz_encoding(clock_signal, input_data):
    encoded_signal = np.zeros_like(clock_signal)
    for i in range(len(input_data)):
        if input_data[i] == 1:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 0.5) * len(clock_signal) / len(input_data))] = 1
            encoded_signal[int((i + 0.5) * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = 0
        else:
            encoded_signal[int(i * len(clock_signal) / len(input_data)):int((i + 0.5) * len(clock_signal) / len(input_data))] = -1
            encoded_signal[int((i + 0.5) * len(clock_signal) / len(input_data)):int((i + 1) * len(clock_signal) / len(input_data))] = 0
    return encoded_signal

nrz_encoded_signal = nrz_encoding(clock_signal, input_data)
nrz_l_encoded_signal = nrz_l_encoding(clock_signal, input_data)
nrz_i_encoded_signal = nrz_i_encoding(clock_signal, input_data)
rz_encoded_signal = rz_encoding(clock_signal, input_data)
manchester_encoded_signal = manchester_encoding(clock_signal, input_data)
differential_manchester_encoded_signal = differential_manchester_encoding(clock_signal, input_data)
ami_encoded_signal = ami_encoding(clock_signal, input_data)
pster_encoded_signal = psedoternary(clock_signal, input_data)

plt.figure(figsize=(12, 8))

plt.subplot(6, 1, 1)
plt.plot(t, clock_signal, label='Clock Signal')
plt.legend()

input_data.insert(0,input_data[0])
plt.subplot(6, 1, 2)
plt.step(np.arange(len(input_data)), input_data, label='Input Data')
plt.legend()

plt.subplot(6, 1, 3)
plt.plot(t, nrz_encoded_signal, label='NRZ Encoded Signal')
plt.legend()

plt.subplot(6, 1, 4)
plt.plot(t, nrz_l_encoded_signal, label='NRZ-L Encoded Signal')
plt.legend()

plt.subplot(6, 1, 5)
plt.plot(t, nrz_i_encoded_signal, label='NRZ-I Encoded Signal')
plt.legend()

plt.subplot(6, 1, 6)
plt.plot(t, rz_encoded_signal, label='RZ Encoded Signal')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(6, 1, 1)
plt.plot(t, clock_signal, label='Clock Signal')
plt.legend()

plt.subplot(6, 1, 2)
plt.step(np.arange(len(input_data)), input_data, label='Input Data')
plt.legend()

plt.subplot(6, 1, 3)
plt.plot(t, manchester_encoded_signal, label='Manchester Encoded Signal')
plt.legend()

plt.subplot(6, 1, 4)
plt.plot(t, differential_manchester_encoded_signal, label='Differential Manchester Encoded Signal')
plt.legend()

plt.subplot(6, 1, 5)
plt.plot(t, ami_encoded_signal, label='AMI Encoded Signal')
plt.legend()

plt.subplot(6, 1, 6)
plt.plot(t, pster_encoded_signal, label='Pseudoternary Encoded Signal')
plt.legend()

plt.tight_layout()
plt.show()





