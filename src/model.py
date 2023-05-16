import pymc3 as pm
import pandas as pd
import arviz as az

# загрузка данных
spotify_data = pd.read_csv('tempo_dataset.csv')

if __name__ == "__main__":
    with pm.Model() as model:
        # define the prior
        mu_bpm = pm.Normal('mu_bpm', mu=120, sd=10)
        sigma_bpm = pm.HalfNormal('sigma_bpm', sd=10)

        # define the likelihood
        bpm_obs = pm.Normal('bpm_obs', mu=mu_bpm, sd=sigma_bpm, observed=spotify_data['tempo'])

        # get the samples
        trace = pm.sample(return_inferencedata=True)

    # Визуализация результатов
    az.plot_posterior(trace, hdi_prob=0.99, show=True)
