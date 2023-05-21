import pymc3 as pm
import pandas as pd
import arviz as az
import librosa
from scipy import stats
import numpy as np


def calc_measure(downbeats: list) -> int:
    downbeat_inds = [i for i, beat in enumerate(downbeats) if beat == 1]
    difs_sum = 0
    for i in range(1, len(downbeat_inds)):
        difs_sum += (downbeat_inds[i] - downbeat_inds[i - 1])
    avg_measure = difs_sum / (len(downbeat_inds) - 1)
    return round(avg_measure)


def estimate_rhythm(audio_path: str) -> int:
    y, sr = librosa.load(audio_path)

    # сила "нажатия" на ноты
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # пиковые силы "нажатия" на ноты
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3,
                                   post_avg=5, delta=0.5, wait=10)
    peaks_time = librosa.frames_to_time(peaks, sr=sr)

    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # beat_strengths = onset_env[beats]
    beats_time = librosa.frames_to_time(beats, sr=sr)
    # предположительные начала тактов
    downbeat_times = {}
    for i, beat in enumerate(beats_time):
        if beat in peaks_time:
            downbeat_times[i] = beat
    # print(downbeats)
    downbeats = [0 for i in range(len(beats))]
    for beat in downbeat_times.keys():
        downbeats[beat] = 1

    with pm.Model() as model:
        downbeat_prob = pm.Beta('downbeat_prob', alpha=2, beta=2)
        # вероятностная модель для генерации начала такта
        downbeats_obs = pm.Bernoulli('downbeats_obs', p=downbeat_prob, shape=len(beats), observed=downbeats)
        trace = pm.sample(1000, tune=1000, chains=2)

    az.plot_posterior(trace, hdi_prob=0.99, show=True)

    downbeat_prob_samples = trace['downbeat_prob']
    density = stats.gaussian_kde(downbeat_prob_samples)
    prob_estimate = downbeat_prob_samples[density(downbeat_prob_samples).argmax()]
    new_downbeats = stats.bernoulli.rvs(prob_estimate, size=len(beats))

    avg_measure = calc_measure(list(new_downbeats))
    return avg_measure


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


def estimate_bpm(audio_path: str, bpm_dataset, genre_dataset, track_genre: str) -> float:
    y, sr = librosa.load(audio_path)
    spectr = librosa.feature.melspectrogram(y=y, sr=sr)
    autocorr = librosa.autocorrelate(spectr)
    min_bpm, max_bpm = get_tempo_bounds(bpm_dataset, genre_dataset, track_genre)
    # диапазон bpm для конкретного аудиофайла
    tempos = librosa.tempo_frequencies(len(autocorr))
    # обозначение жанров числами
    genres_ints = genres_to_ints(genre_dataset)
    genre_int = np.where(genre_dataset.unique() == track_genre)[0][0]

    with pm.Model() as model:
        # hyperpriors (lvl 1)
        tempo = pm.Uniform('tempo', lower=min_bpm, upper=max_bpm)
        mu = (min_bpm + max_bpm) / 2.0
        sigma = (max_bpm - min_bpm) / 12.0
        genre_coef = pm.Normal('genre_coef', mu=0, sd=1, shape=len(genre_dataset.unique()))

        # prior (lvl 2)
        bpm_est = mu + genre_coef[genres_ints] * sigma

        # likelihood (lvl 3)
        bpm_obs = pm.Normal('bpm_obs', mu=bpm_est, sd=sigma, observed=bpm_dataset)
        # bpm_obs = pm.Normal('bpm_obs', mu=mu, sd=sigma, observed=spotify_data['tempo'])

        # get the samples
        trace = pm.sample(1000, tune=500, chains=2, cores=1)

    # Визуализация результатов
    # az.plot_posterior(trace, hdi_prob=0.99, show=True)

    genres_samples = trace['genre_coef']
    genre_coef_samples = [genres_samples[i][genre_int] for i in range(len(genres_samples))]
    # оценка функции плотности распределения genre_coef_samples
    density = stats.gaussian_kde(genre_coef_samples)
    # наибольшая плотность - наиболее вероятный коэффициент
    coef_estimate = genre_coef_samples[density(genre_coef_samples).argmax()]
    print(coef_estimate)
    sigma = (max_bpm - min_bpm) / 12.0

    tempo_samples = trace['tempo']
    density = stats.gaussian_kde(tempo_samples)
    bpm_estimate = tempos[density(tempos).argmax()] + coef_estimate * sigma

    return float(bpm_estimate)


if __name__ == "__main__":
    # загрузка данных
    spotify_data = pd.read_csv('tempo_dataset.csv')
    music = 'test_audio/vo_sne_guitars.mp3'
    genre = 'alt-rock'
    # print(get_tempo_bounds(spotify_data['tempo'], spotify_data['track_genre'], genre))

    print('Estimated tempo: ', estimate_bpm(music, spotify_data['tempo'], spotify_data['track_genre'], genre))
    # print('Estimated time signature: ', estimate_rhythm(music))
