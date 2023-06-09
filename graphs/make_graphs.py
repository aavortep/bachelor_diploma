import matplotlib.pyplot as plt
import numpy as np

index = np.arange(3)
bw = 0.3
plt.bar(index, [(84.45 + 85.15)/2, (82.58 + 87.3 + 93)/3, 70.04], bw, label='разработанный метод')
plt.bar(index + bw, [(67.69 + 87.2)/2, (81.04 + 86.71 + 78)/3, 64.14], bw, label='librosa', color='white', edgecolor='red', linewidth=2, linestyle='--')
plt.xlabel("Жанры")
plt.ylabel("Точность определения переменного темпа (%)")
plt.xticks(index+0.5*bw, ["поп", "рок", "соул"])
plt.legend()
plt.show()

index = np.arange(4)
bw = 0.4
plt.bar(index, [91.51, 72.48, 85.3, 93], bw, label='разработанный метод')
plt.bar(index + bw, [90.18, 90.89, 90.91, 99], bw, label='librosa', color='white', edgecolor='red', linewidth=2, linestyle='--')
plt.xlabel("Жанры")
plt.ylabel("Точность определения постоянного темпа (%)")
plt.xticks(index+0.5*bw, ["фанк", "джаз", "поп", "рок"])
plt.legend()
plt.show()
