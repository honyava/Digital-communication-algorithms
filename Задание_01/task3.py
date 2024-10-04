import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
scramb = np.load('scramb.npy')
data_bpsk = np.load('data_bpsk.npy')
data_qpsk = np.load('data_qpsk.npy')

# Извлечение символов с 24-го отсчета и далее через 4, всего 100 символов
start_index = 24
step = 4
num_symbols = 100

bpsk_symbols = data_bpsk[start_index:start_index + num_symbols * step:step]
qpsk_symbols = data_qpsk[start_index:start_index + num_symbols * step:step]

# Снятие скремблирующей последовательности
scramb_phase = np.exp(-1j * 2 * np.pi * scramb / 8)

# Применяем скремблирующую последовательность к сигналам
bpsk_descrambled = bpsk_symbols * scramb_phase[:len(bpsk_symbols)]
qpsk_descrambled = qpsk_symbols * scramb_phase[:len(qpsk_symbols)]

# Построение созвездий
plt.figure(figsize=(12, 6))

# Созвездие BPSK до и после снятия скремблирования
plt.subplot(2, 2, 1)
plt.scatter(bpsk_symbols.real, bpsk_symbols.imag, color='blue')
plt.title("Созвездие BPSK с шумом")
plt.xlabel("Вещественная часть")
plt.ylabel("Мнимая часть")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.scatter(bpsk_descrambled.real, bpsk_descrambled.imag, color='green')
plt.title("Созвездие BPSK после снятия скремблирования")
plt.xlabel("Вещественная часть")
plt.ylabel("Мнимая часть")
plt.grid(True)

# Созвездие QPSK до и после снятия скремблирования
plt.subplot(2, 2, 3)
plt.scatter(qpsk_symbols.real, qpsk_symbols.imag, color='red')
plt.title("Созвездие QPSK с шумом")
plt.xlabel("Вещественная часть")
plt.ylabel("Мнимая часть")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.scatter(qpsk_descrambled.real, qpsk_descrambled.imag, color='purple')
plt.title("Созвездие QPSK после снятия скремблирования")
plt.xlabel("Вещественная часть")
plt.ylabel("Мнимая часть")
plt.grid(True)

plt.tight_layout()
plt.show()
