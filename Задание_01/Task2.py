import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rcosfilter

# Модуляция с формирующим фильтром RRC
def psk_modulation_filtered(b, order, fcarrier, fs, fsym, rolloff):
    Ts = 1 / fsym  # Длительность символа
    num_samples_per_symbol = int(fs / fsym)  # Количество отсчетов на символ
    N = num_samples_per_symbol * 100  # Длина фильтра (умножаем для получения более точной формы)
    
    # Создаем формирующий фильтр с поднятым косинусом
    filter_coeffs, _ = rcosfilter(N, rolloff, Ts, fs)
    t_symbol = np.linspace(0, Ts, num_samples_per_symbol, endpoint=False)
    
    signal = np.array([])

    if order == 2:  # BPSK
        for bit in b:
            phase = 0 if bit == 0 else np.pi
            carrier = np.cos(2 * np.pi * fcarrier * t_symbol + phase)
            modulated_symbol = carrier
            signal = np.concatenate((signal, modulated_symbol))

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
            carrier = np.cos(2 * np.pi * fcarrier * t_symbol + phase)
            modulated_symbol = carrier
            signal = np.concatenate((signal, modulated_symbol))

    # Применение формирующего фильтра к модульному сигналу
    filtered_signal = np.convolve(signal, filter_coeffs, mode='same')
    
    return signal, filtered_signal

# Параметры сигнала
fcarrier = 32  # Частота несущей
fs = 512  # Частота дискретизации
fsym = 8  # Частота символов
rolloff = 1  # Коэффициент скругления
order = 4  # QPSK
num_symbols = 5000  # Количество символов

# Создание случайной битовой последовательности
bits_per_symbol = int(np.log2(order))
num_bits = num_symbols * bits_per_symbol
bitstream = np.random.randint(0, 2, num_bits).tolist()

# Получение сигнала с формирующим фильтром и без
signal_unfiltered, signal_filtered = psk_modulation_filtered(bitstream, order, fcarrier, fs, fsym, rolloff)

# Построение спектров сигналов
plt.figure(figsize=(12, 6))

# Спектр сигнала без фильтрации
frequencies_unfiltered = np.fft.fftfreq(len(signal_unfiltered), 1/fs)
spectrum_unfiltered = np.fft.fft(signal_unfiltered)
plt.subplot(1, 2, 1)
plt.plot(frequencies_unfiltered[:len(frequencies_unfiltered)//2], np.abs(spectrum_unfiltered)[:len(frequencies_unfiltered)//2])
plt.title("Спектр сигнала без фильтрации")
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда")
plt.grid()

# Спектр сигнала с фильтрацией
frequencies_filtered = np.fft.fftfreq(len(signal_filtered), 1/fs)
spectrum_filtered = np.fft.fft(signal_filtered)
plt.subplot(1, 2, 2)
plt.plot(frequencies_filtered[:len(frequencies_filtered)//2], np.abs(spectrum_filtered)[:len(frequencies_filtered)//2])
plt.title("Спектр сигнала с фильтрацией")
plt.xlabel("Частота (Гц)")
plt.ylabel("Амплитуда")
plt.grid()

plt.tight_layout()
plt.show()
