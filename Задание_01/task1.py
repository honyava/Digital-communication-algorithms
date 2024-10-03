import numpy as np
import matplotlib.pyplot as plt

def psk_modulation(b, order, fcarrier, fs, fsym):
    t = np.linspace(0, 1/fsym, int(fs/fsym), endpoint=False)  # Временная шкала для одного символа
    signal = np.array([])

    if order == 2:  # BPSK
        for bit in b:
            phase = 0 if bit == 0 else np.pi
            carrier = np.cos(2 * np.pi * fcarrier * t + phase)
            signal = np.concatenate((signal, carrier))

    elif order == 4:  # QPSK
        for i in range(0, len(b), 2):
            bit_pair = b[i:i+2]
            if bit_pair == [0, 0]:
                phase = 0
            elif bit_pair == [0, 1]:
                phase = np.pi / 2
            elif bit_pair == [1, 0]:
                phase = -np.pi / 2
            elif bit_pair == [1, 1]:
                phase = np.pi
            carrier = np.cos(2 * np.pi * fcarrier * t + phase)
            signal = np.concatenate((signal, carrier))

    elif order == 8:  # 8PSK
        for i in range(0, len(b), 3):
            bit_triplet = b[i:i+3]
            if bit_triplet == [0, 0, 0]:
                phase = 0
            elif bit_triplet == [0, 0, 1]:
                phase = np.pi / 4
            elif bit_triplet == [0, 1, 0]:
                phase = np.pi / 2
            elif bit_triplet == [0, 1, 1]:
                phase = 3 * np.pi / 4
            elif bit_triplet == [1, 0, 0]:
                phase = np.pi
            elif bit_triplet == [1, 0, 1]:
                phase = 5 * np.pi / 4
            elif bit_triplet == [1, 1, 0]:
                phase = 3 * np.pi / 2
            elif bit_triplet == [1, 1, 1]:
                phase = 7 * np.pi / 4
            carrier = np.cos(2 * np.pi * fcarrier * t + phase)
            signal = np.concatenate((signal, carrier))

    return signal

# Функция для построения спектра
def plot_spectrum(fsym, fcarrier=15, fs=30, num_symbols=100):
    orders = [2, 4, 8]  # BPSK, QPSK, 8PSK
    plt.figure(figsize=(15, 6))

    for i, order in enumerate(orders):
        bits_per_symbol = int(np.log2(order))
        num_bits = num_symbols * bits_per_symbol
        bitstream = np.random.randint(0, 2, num_bits).tolist()

        # Модуляция
        modulated_signal = psk_modulation(bitstream, order, fcarrier, fs, fsym)

        # Построение спектра сигнала
        frequencies = np.fft.fftfreq(len(modulated_signal), 1/fs)
        spectrum = np.fft.fft(modulated_signal)

        plt.subplot(1, 3, i+1)
        plt.plot(frequencies[:len(frequencies)//2], np.abs(spectrum)[:len(frequencies)//2])
        plt.title(f"{order}-PSK Спектр сигнала")
        plt.xlabel("Частота, Гц")
        plt.ylabel("Амплитуда")
        plt.grid()

    plt.tight_layout()
    plt.show()

# Построение спектров для fsym = 1
plot_spectrum(fsym=1)

# Построение спектров для fsym = 2
plot_spectrum(fsym=2)
