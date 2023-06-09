from split_audio import split_mp3
from clean_dirs import clean_dir, remove_dir
from model import bpm_model
from estimation import get_tempo_bounds, genres_to_ints
import pandas as pd
import librosa
from scipy import stats
import numpy as np
import os


def compare_methods(root_dir_path: str):
    spotify_data = pd.read_csv('tempo_dataset.csv')
    bpm_dataset = spotify_data['tempo']
    genre_dataset = spotify_data['track_genre']
    step = 5000
    for root, dirs, files in os.walk(root_dir_path):
        if len(files) == 0:
            continue
        genre = (root.split("\\"))[-1]
        print(genre)
        min_bpm, max_bpm = get_tempo_bounds(bpm_dataset, genre_dataset, genre)
        genres_ints = genres_to_ints(genre_dataset)

        trace = bpm_model(min_bpm, max_bpm, bpm_dataset, genre_dataset, genres_ints)

        for filename in files:
            audio_path = os.path.join(root, filename)
            split_mp3(audio_path, step)
            bpms = []
            lib_bpms = []
            for audio in os.listdir("tmp/"):
                y, sr = librosa.load("tmp/" + audio)
                genre_int = np.where(genre_dataset.unique() == genre)[0][0]
                delta = 40
                prior_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
                lib_bpms.append(round(prior_bpm))
                tempos = np.arange(prior_bpm - delta, prior_bpm + delta, 0.5)

                genres_samples = trace['genre_coef']
                genre_coef_samples = [genres_samples[i][genre_int] for i in range(len(genres_samples))]
                # оценка функции плотности распределения genre_coef_samples
                density = stats.gaussian_kde(genre_coef_samples)
                # наибольшая плотность - наиболее вероятный коэффициент
                coef_estimate = genre_coef_samples[density(genre_coef_samples).argmax()]
                sigma = (max_bpm - min_bpm) / 12.0

                tempo_samples = trace['tempo']
                density = stats.gaussian_kde(tempo_samples)
                bpms.append(round(tempos[density(tempos).argmax()] + coef_estimate * sigma))

            clean_dir("tmp/")
            remove_dir("tmp/")
            name_only = (filename.split("."))[0]
            res_file = open(root + "/" + name_only + ".txt", "w")
            for i in range(len(bpms)):
                res_file.write(str(i * 5) + " " + str(lib_bpms[i]) + " " + str(bpms[i]) + "\n")
            res_file.close()


if __name__ == "__main__":
    compare_methods("test_audio/genres")
