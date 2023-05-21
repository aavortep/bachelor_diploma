from split_audio import split_mp3
from clean_dirs import clean_dir, remove_dir
from model import bpm_model, rhythm_model
import pandas as pd
import librosa
from scipy import stats
import numpy as np
import os


def genres_to_ints(genre_dataset):
    all_genres = genre_dataset.unique()
    genres_dict = {}
    for i, genre in enumerate(all_genres):
        genres_dict[genre] = i
    genres_ints = np.array([0] * len(genre_dataset))
    for i, genre in enumerate(genre_dataset.values):
        genres_ints[i] = genres_dict[genre]
    return genres_ints


def get_tempo_bounds(bpm_dataset, genre_dataset, track_genre: str):
    rows_num = [i for i in range(len(genre_dataset)) if genre_dataset.values[i] == track_genre]
    tempos = [bpm_dataset.values[i] for i in rows_num]
    max_bpm = max(tempos)
    min_bpm = min(tempos)
    return min_bpm, max_bpm


def estimate_bpm(audio_path: str, bpm_dataset, genre_dataset, track_genre: str) -> dict:
    step = 2000
    # диапазон bpm для аудиофайла
    min_bpm, max_bpm = get_tempo_bounds(bpm_dataset, genre_dataset, track_genre)
    # обозначение жанров числами
    genres_ints = genres_to_ints(genre_dataset)

    # байесовское моделирование для темпа
    trace = bpm_model(min_bpm, max_bpm, bpm_dataset, genre_dataset, genres_ints)

    split_mp3(audio_path, step)
    bpms = []
    times = []
    t = 0
    for audio in os.listdir("tmp/"):
        y, sr = librosa.load("tmp/" + audio)
        spectr = librosa.feature.melspectrogram(y=y, sr=sr)
        autocorr = librosa.autocorrelate(spectr)
        tempos = librosa.tempo_frequencies(len(autocorr))
        genre_int = np.where(genre_dataset.unique() == track_genre)[0][0]

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
        times.append(t)

        t += (step / 1000)

    inds_to_delete = []
    for i in range(1, len(bpms)):
        if bpms[i] == bpms[i - 1]:
            inds_to_delete.append(i)
    for i in range(len(inds_to_delete)-1, -1, -1):
        bpms.pop(inds_to_delete[i])
        times.pop(inds_to_delete[i])
    res_tempos = {times[i]: bpms[i] for i in range(len(bpms))}

    clean_dir("tmp/")
    remove_dir("tmp/")
    return res_tempos


def calc_measure(downbeats: list) -> int:
    downbeat_inds = [i for i, beat in enumerate(downbeats) if beat == 1]
    difs_sum = 0
    for i in range(1, len(downbeat_inds)):
        difs_sum += (downbeat_inds[i] - downbeat_inds[i - 1])
    avg_measure = difs_sum / (len(downbeat_inds) - 1)
    return round(avg_measure)


def estimate_rhythm(audio_path: str) -> dict:
    step = 10000  # длина частей аудио в мс
    split_mp3(audio_path, step)
    measures = []
    times = []
    t = 0
    for audio in os.listdir("tmp/"):
        y, sr = librosa.load("tmp/" + audio)

        # сила "нажатия" на ноты
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # пиковые силы "нажатия" на ноты
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3,
                                       post_avg=5, delta=0.5, wait=10)
        peaks_time = librosa.frames_to_time(peaks, sr=sr)

        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beats_time = librosa.frames_to_time(beats, sr=sr)
        # предположительные начала тактов
        downbeat_times = {}
        for i, beat in enumerate(beats_time):
            if beat in peaks_time:
                downbeat_times[i] = beat
        downbeats = [0 for i in range(len(beats))]
        for beat in downbeat_times.keys():
            downbeats[beat] = 1

        # байесовское моделирование для ритма
        trace = rhythm_model(beats, downbeats)

        downbeat_prob_samples = trace['downbeat_prob']
        density = stats.gaussian_kde(downbeat_prob_samples)
        prob_estimate = downbeat_prob_samples[density(downbeat_prob_samples).argmax()]
        new_downbeats = stats.bernoulli.rvs(prob_estimate, size=len(beats))

        measures.append(calc_measure(list(new_downbeats)))
        times.append(t)

        t += (step / 1000)

    inds_to_delete = []
    for i in range(1, len(measures)):
        if measures[i] == measures[i - 1]:
            inds_to_delete.append(i)
    for i in range(len(inds_to_delete) - 1, -1, -1):
        measures.pop(inds_to_delete[i])
        times.pop(inds_to_delete[i])
    res_measures = {times[i]: measures[i] for i in range(len(measures))}

    clean_dir("tmp/")
    remove_dir("tmp/")

    return res_measures


if __name__ == "__main__":
    # загрузка данных
    spotify_data = pd.read_csv('tempo_dataset.csv')
    music = 'test_audio/chop.mp3'
    genre = 'metal'

    print('Estimated tempo: ', estimate_bpm(music, spotify_data['tempo'], spotify_data['track_genre'], genre))
    #print('Estimated time signature: ', estimate_rhythm(music))