import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rcosfilter


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


def psk_demodulation(signal, order, fcarrier, fs, fsym):
    t = np.linspace(0, 1/fsym, int(fs/fsym), endpoint=False)  # Временная шкала для одного символа
    
    num_samples_per_symbol = len(t)
    num_symbols = len(signal) // num_samples_per_symbol
    demod_bits = []

    if order == 2:  # BPSK
        for i in range(num_symbols):
            symbol = signal[i * num_samples_per_symbol:(i + 1) * num_samples_per_symbol]
            reference = np.cos(2 * np.pi * fcarrier * t)
            demodulated = np.sum(symbol * reference)
            bit = 0 if demodulated > 0 else 1
            demod_bits.append(bit)   
    elif order == 4:  # QPSK
        for i in range(num_symbols):
            symbol = signal[i * num_samples_per_symbol:(i + 1) * num_samples_per_symbol]
            reference_I = np.cos(2 * np.pi * fcarrier * t)
            reference_Q = -np.sin(2 * np.pi * fcarrier * t)
            I_component = int(np.sum(symbol * reference_I))
            Q_component = int(np.sum(symbol * reference_Q))
            
            if I_component > 0 and Q_component >= 0:
                demod_bits += [0, 0]  # 00
            elif I_component <= 0 and Q_component > 0:
                demod_bits += [0, 1]  # 01
            elif I_component < 0 and Q_component <= 0:
                demod_bits += [1, 1]  # 11
            elif I_component >= 0 and Q_component < 0:
                demod_bits += [1, 0]  # 10

    elif order == 8:  # 8PSK
        for i in range(num_symbols):
            symbol = signal[i * num_samples_per_symbol:(i + 1) * num_samples_per_symbol]
            reference_cos = np.cos(2 * np.pi * fcarrier * t)
            reference_sin = -np.sin(2 * np.pi * fcarrier * t)

            I_component = np.sum(symbol * reference_cos)
            Q_component = np.sum(symbol * reference_sin)

            phase = np.arctan2(Q_component, I_component)  # Определение фазы

            if -np.pi/8 <= phase < np.pi/8:
                demod_bits += [0, 0, 0]  # 000
            elif np.pi/8 <= phase < 3*np.pi/8:
                demod_bits += [0, 0, 1]  # 001
            elif 3*np.pi/8 <= phase < 5*np.pi/8:
                demod_bits += [0, 1, 0]  # 010
            elif 5*np.pi/8 <= phase < 7*np.pi/8:
                demod_bits += [0, 1, 1]  # 011
            elif phase < -7*np.pi/8 or phase >= 7*np.pi/8:
                demod_bits += [1, 0, 0]  # 100
            elif -7*np.pi/8 <= phase < -5*np.pi/8:
                demod_bits += [1, 0, 1]  # 101
            elif -5*np.pi/8 <= phase < -3*np.pi/8:
                demod_bits += [1, 1, 0]  # 110
            elif -3*np.pi/8 <= phase < -np.pi/8:
                demod_bits += [1, 1, 1]  # 111

    return demod_bits

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
# plot_spectrum(fsym=1)

# Построение спектров для fsym = 2
# plot_spectrum(fsym=2)

################## DEMODULATION ####################### 


# Пример использования демодулятора
fcarrier = 32  # Частота несущей
fs = 1000  # Частота дискретизации
fsym = 8  # Частота символов
order = 4  # Порядок модуляции 
num_symbols = 10  # Количество символов

bits_per_symbol = int(np.log2(order))
num_bits = num_symbols * bits_per_symbol
bitstream = np.random.randint(0, 2, num_bits).tolist()

# Модуляция
modulated_signal = psk_modulation(bitstream, order, fcarrier, fs, fsym)

# Демодуляция
demodulated_bits = psk_demodulation(modulated_signal, order, fcarrier, fs, fsym)

# Сравнение исходной и демодулированной последовательности
print("Исходные биты:        ", bitstream)
print("Демодулированные биты:", demodulated_bits)
print("Совпадают ли биты: ", bitstream == demodulated_bits)
