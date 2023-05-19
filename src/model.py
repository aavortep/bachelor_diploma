import pymc3 as pm
import pandas as pd
import arviz as az
import librosa
from scipy import stats


def estimate_bpm(audio_path: str, bpm_dataset):
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

    return bpm_estimate


if __name__ == "__main__":
    # загрузка данных
    spotify_data = pd.read_csv('tempo_dataset.csv')

    print('Estimated tempo: ', estimate_bpm('test_audio/master.mp3', spotify_data['tempo']))
