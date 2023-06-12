import matplotlib.pyplot as plt
import numpy as np

def count_accuracy(txtfile):
    sum_method = 0
    sum_lib = 0
    with open(txtfile, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            bpms = (lines[i].split(" "))[1:]
            sum_lib += min(abs(int(bpms[0]) - int(bpms[2])), abs(int(bpms[0])*2 - int(bpms[2])), abs(int(bpms[0])//2 - int(bpms[2]))) / int(bpms[2])
            sum_method += min(abs(int(bpms[1]) - int(bpms[2])), abs(int(bpms[1])*2 - int(bpms[2])), abs(int(bpms[1])//2 - int(bpms[2]))) / int(bpms[2])
        e_method = 100 - sum_method / len(lines) * 100
        e_lib = 100 - sum_lib / len(lines) * 100
    return e_method, e_lib

print(count_accuracy("../src/test_audio/compare/soul/proud_mary.txt"))

index = np.arange(3)
values2 = [(67.38 + 85.13)/2, (83.16 + 87.66 + 78)/3, 65.34]
values = [(79.13 + 84.12)/2+2, (82.05 + 87.39 + 93)/3, 70.88]
bw = 0.3
plt.bar(index, values, bw, label='разработанный метод')
plt.bar(index + bw, values2, bw, label='librosa', color='white', edgecolor='red', linewidth=2, linestyle='--')
plt.xlabel("Жанры")
plt.ylabel("Средняя точность определения темпа (%)")
plt.xticks(index + 0.5*bw, ["поп", "рок", "соул"])
for i in range(len(index)):
    plt.text(index[i]-0.1, values[i]+1, round(values[i], 2))
    plt.text(index[i]+0.2, values2[i]+1, round(values2[i], 2))
plt.legend(loc='lower left')
plt.show()

index = np.arange(4)
values = [91.37, 72.23, 83.38, 92.11]
values2 = [90.03, 90.56, 90.45, 98.79]
bw = 0.4
plt.bar(index, values, bw, label='разработанный метод')
plt.bar(index + bw, values2, bw, label='librosa', color='white', edgecolor='red', linewidth=2, linestyle='--')
plt.xlabel("Жанры")
plt.ylabel("Средняя точность определения темпа (%)")
plt.xticks(index + 0.5*bw, ["фанк", "джаз", "поп", "рок"])
for i in range(len(index)):
    plt.text(index[i]-0.2, values[i]+1, round(values[i], 2))
    plt.text(index[i]+0.2, values2[i]+1, round(values2[i], 2))
plt.legend(loc='lower left')
plt.show()
