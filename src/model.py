import pymc3 as pm


def rhythm_model(beats, downbeats):
    with pm.Model() as model:
        downbeat_prob = pm.Beta('downbeat_prob', alpha=2, beta=2)
        # вероятностная модель для генерации начала такта
        downbeats_obs = pm.Bernoulli('downbeats_obs', p=downbeat_prob, shape=len(beats), observed=downbeats)
        trace = pm.sample(1000, tune=1000, chains=2)

    return trace


def bpm_model(min_bpm: int, max_bpm: int, bpm_dataset, genre_dataset, genres_ints):
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

    return trace
