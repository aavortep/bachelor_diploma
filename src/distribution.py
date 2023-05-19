import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def show_distributions(bpm_data, rhythm_data):
    bpm_list = list(map(round, bpm_data.array))
    bpm_dict = dict(Counter(bpm_list))
    bpms = bpm_dict.keys()
    bpm_counts = bpm_dict.values()
    plt.bar(bpms, bpm_counts)
    plt.xlabel("BPM")
    plt.ylabel("Кол-во")
    plt.show()

    rhythm_list = list(rhythm_data.array)
    rhythm_dict = dict(Counter(rhythm_list))
    rhythms = rhythm_dict.keys()
    rhythm_counts = rhythm_dict.values()
    plt.bar(rhythms, rhythm_counts)
    plt.xlabel("Размеры")
    plt.ylabel("Кол-во")
    plt.show()


if __name__ == "__main__":
    spotify_data = pd.read_csv('tempo_dataset.csv')
    bpm_data = spotify_data['tempo']
    rhythm_data = spotify_data['time_signature']
    show_distributions(bpm_data, rhythm_data)
