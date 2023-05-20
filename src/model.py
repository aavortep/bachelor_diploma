import pymc3 as pm
import pandas as pd
import arviz as az
import librosa
from scipy import stats


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
    # начала тактов
    downbeats = {}
    for i, beat in enumerate(beats_time):
        if beat in peaks_time:
            downbeats[i] = beat
    # print(downbeats)

    downbeats_nums = sorted(list(downbeats.keys()))
    difs_sum = 0
    for i in range(1, len(downbeats_nums)):
        difs_sum += downbeats_nums[i] - downbeats_nums[i - 1]

    # среднее число ударов в такте (средний размер)
    avg_measure = difs_sum / (len(downbeats_nums) - 1)

    return round(avg_measure)+1


def estimate_bpm(audio_path: str, bpm_dataset) -> float:
    y, sr = librosa.load(audio_path)
    spectr = librosa.feature.melspectrogram(y=y, sr=sr)
    autocorr = librosa.autocorrelate(spectr)
    min_bpm = 40
    max_bpm = 260
    # диапазон bpm для конкретного аудиофайла
    tempos = librosa.tempo_frequencies(len(autocorr))

    with pm.Model() as model:
        # define the prior
        tempo = pm.Uniform('tempo', lower=min_bpm, upper=max_bpm)
        # mu = pm.Normal('mu', mu=120, sd=10)
        # sigma = pm.HalfNormal('sigma', sd=10)

        # define the likelihood
        bpm_obs = pm.Normal('bpm_obs', mu=(min_bpm + max_bpm) / 2.0, sd=(max_bpm - min_bpm) / 12.0,
                            observed=bpm_dataset)
        # bpm_obs = pm.Normal('bpm_obs', mu=mu, sd=sigma, observed=spotify_data['tempo'])

        # get the samples
        trace = pm.sample(1000, tune=1000, chains=2)

    # Визуализация результатов
    az.plot_posterior(trace, hdi_prob=0.99, show=True)

    tempo_samples = trace['tempo']
    # оценка функции плотности распределения tempo_samples
    density = stats.gaussian_kde(tempo_samples)
    # наибольшая плотность - наиболее вероятный темп
    bpm_estimate = tempos[density(tempos).argmax()]

    return float(bpm_estimate)


if __name__ == "__main__":
    # загрузка данных
    spotify_data = pd.read_csv('tempo_dataset.csv')
    music = 'test_audio/vo_sne_guitars.mp3'

    print('Estimated tempo: ', estimate_bpm(music, spotify_data['tempo']))
    print('Estimated time signature: ', estimate_rhythm(music))
