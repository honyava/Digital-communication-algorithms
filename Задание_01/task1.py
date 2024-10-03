import numpy as np
import matplotlib.pyplot as plt

def psk_modulation(b, order, fcarrier, fs, fsym):

    t = np.linspace(0, 1/fsym, int(fs/fsym), endpoint=False)  # Временная шкала для одного символа
    signal = np.array([])

    bit_labels = []  # Для хранения меток битов, которые будем выводить на график

    if order == 2:
        for bit in b:
            if bit == 0:
                phase = 0
                label = '0'
            else:
                phase = np.pi
                label = '1'
            carrier = np.cos(2 * np.pi * fcarrier * t + phase)
            signal = np.concatenate((signal, carrier))
            bit_labels.append(label)

    elif order == 4:
        for i in range(0, len(b), 2):
            bit_pair = b[i:i+2]
            if bit_pair == [0, 0]:
                phase = 0
                label = '00'
            elif bit_pair == [0, 1]:
                phase = np.pi / 2
                label = '01'
            elif bit_pair == [1, 0]:
                phase = -np.pi / 2
                label = '10'
            elif bit_pair == [1, 1]:
                phase = np.pi
                label = '11'
            carrier = np.cos(2 * np.pi * fcarrier * t + phase)
            signal = np.concatenate((signal, carrier))
            bit_labels.append(label)

    elif order == 8:
        for i in range(0, len(b), 3):
            bit_triplet = b[i:i+3]
            if bit_triplet == [0, 0, 0]:
                phase = 0
                label = '000'
            elif bit_triplet == [0, 0, 1]:
                phase = np.pi / 4
                label = '001'
            elif bit_triplet == [0, 1, 0]:
                phase = np.pi / 2
                label = '010'
            elif bit_triplet == [0, 1, 1]:
                phase = 3 * np.pi / 4
                label = '011'
            elif bit_triplet == [1, 0, 0]:
                phase = np.pi
                label = '100'
            elif bit_triplet == [1, 0, 1]:
                phase = 5 * np.pi / 4
                label = '101'
            elif bit_triplet == [1, 1, 0]:
                phase = 3 * np.pi / 2
                label = '110'
            elif bit_triplet == [1, 1, 1]:
                phase = 7 * np.pi / 4
                label = '111'
            carrier = np.cos(2 * np.pi * fcarrier * t + phase)
            signal = np.concatenate((signal, carrier))
            bit_labels.append(label)

    return signal, bit_labels

# Генерация случайной последовательности бит
order = 4  # Для 8PSK
num_symbols = 10  # 10 символов
bits_per_symbol = int(np.log2(order))
num_bits = num_symbols * bits_per_symbol

# Случайная битовая последовательность
bitstream = np.random.randint(0, 2, num_bits).tolist()

# Параметры модуляции
fcarrier = 5  # Частота несущей, Гц
fs = 10e3  # Частота дискретизации, Гц
fsym = 1  # Частота символов, символов в секунду

# Модуляция
modulated_signal, bit_labels = psk_modulation(bitstream, order, fcarrier, fs, fsym)

# Вычисление огибающей сигнала
envelope = np.abs(modulated_signal)

# Построение графика модулированного сигнала
plt.figure(figsize=(10, 6))
time = np.linspace(0, len(modulated_signal)/fs, len(modulated_signal))
plt.plot(time[:int(fs/fsym * num_symbols)], modulated_signal[:int(fs/fsym * num_symbols)], label='Модулированный сигнал')
#plt.plot(time[:int(fs/fsym * num_symbols)], envelope[:int(fs/fsym * num_symbols)], color='orange', linestyle='--', label='Огибающая')

# Добавление аннотаций с битовыми значениями
symbol_duration = int(fs / fsym)
for i, label in enumerate(bit_labels[:num_symbols]):
    t_symbol = time[i * symbol_duration] + (symbol_duration / (2 * fs))  # Середина символа
    plt.annotate(label, xy=(t_symbol, envelope[i * symbol_duration]), xytext=(t_symbol, envelope[i * symbol_duration] + 0.1),
                 arrowprops=dict(arrowstyle="->", color='red'), ha='center')

plt.title(f"{order}-PSK Модулированный сигнал с огибающей и битовыми значениями")
plt.xlabel("Время, сек")
plt.ylabel("Амплитуда")
plt.grid()
plt.legend()
plt.show()

# Построение спектра сигнала
plt.figure(figsize=(10, 4))
frequencies = np.fft.fftfreq(len(modulated_signal), 1/fs)
spectrum = np.fft.fft(modulated_signal)
plt.plot(frequencies[:len(frequencies)//2], np.abs(spectrum)[:len(frequencies)//2])
plt.title("Спектр модулированного сигнала")
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.grid()
plt.show()
