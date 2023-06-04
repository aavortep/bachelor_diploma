import matplotlib.pyplot as plt
import numpy as np

index = np.arange(4)
plt.bar(index, [100, 100, 75, 75])
plt.xlabel("Жанры")
plt.ylabel("Точность определения размера (%)")
plt.xticks(index, ["поп", "рок", "фанк", "джаз"])
plt.show()
