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


def estimate_bpm(audio_path: str, bpm_dataset, genre_dataset, track_genre: str, step: int, progress) -> dict:
    # диапазон bpm для аудиофайла
    min_bpm, max_bpm = get_tempo_bounds(bpm_dataset, genre_dataset, track_genre)
    # обозначение жанров числами
    genres_ints = genres_to_ints(genre_dataset)

    # байесовское моделирование для темпа
    trace = bpm_model(min_bpm, max_bpm, bpm_dataset, genre_dataset, genres_ints, progress)

    split_mp3(audio_path, step)
    bpms = []
    times = []
    t = 0
    for audio in os.listdir("tmp/"):
        y, sr = librosa.load("tmp/" + audio)
        genre_int = np.where(genre_dataset.unique() == track_genre)[0][0]
        delta = 40
        prior_bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
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
    if len(downbeat_inds) <= 1:
        return 0
    difs_sum = 0
    for i in range(1, len(downbeat_inds)):
        difs_sum += (downbeat_inds[i] - downbeat_inds[i - 1])
    avg_measure = difs_sum / (len(downbeat_inds) - 1)
    return round(avg_measure)


def get_measure_range(audio_path: str, tail: list):
    y, sr = librosa.load(audio_path)
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
    downbeats = tail + downbeats
    prior_measure = calc_measure(downbeats)
    if prior_measure == 0:
        return -1, downbeats
    if prior_measure > 2:
        return 0, np.arange(prior_measure - 2, prior_measure + 3, 1)
    elif prior_measure == 2:
        return 0, np.arange(prior_measure - 1, prior_measure + 3, 1)
    else:
        return 0, np.arange(prior_measure, prior_measure + 3, 1)


def estimate_rhythm(audio_path: str, rhythm_dataset, step: int, progress) -> dict:
    # байесовское моделирование для ритма
    _, prior_range = get_measure_range(audio_path, [])
    trace = rhythm_model(prior_range[0], prior_range[-1], rhythm_dataset, progress)

    split_mp3(audio_path, step)
    measures = []
    times = []
    t = 0
    tail = []
    for audio in os.listdir("tmp/"):
        er, measure_range = get_measure_range("tmp/" + audio, tail)
        if er == -1:  # если в отрывке меньше двух сильных ударов, то объединяем со следующим отрывком
            tail = measure_range
            continue
        else:
            tail = []

        measure_samples = trace['measure']
        density = stats.gaussian_kde(measure_samples)
        measure_estimate = measure_range[density(measure_range).argmax()]

        measures.append(measure_estimate)
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
    music = 'test_audio/muzak.mp3'
    genre = 'alt-rock'

    #print('Estimated tempo: ', estimate_bpm(music, spotify_data['tempo'], spotify_data['track_genre'],
    #                                        genre, 5000))
    print('Estimated time signature: ', estimate_rhythm(music, spotify_data['time_signature'], 5000))
