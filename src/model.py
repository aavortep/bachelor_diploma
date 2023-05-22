import pymc3 as pm


def calc_measure(downbeats: list) -> int:
    print(downbeats)
    downbeat_inds = [i for i, beat in enumerate(downbeats) if beat == 1]
    difs_sum = 0
    for i in range(1, len(downbeat_inds)):
        difs_sum += (downbeat_inds[i] - downbeat_inds[i - 1])
    avg_measure = difs_sum / (len(downbeat_inds) - 1)
    return round(avg_measure)


def rhythm_model(measure_min, measure_max, rhythm_dataset):
    with pm.Model() as model:
        # prior
        measure = pm.Uniform('measure', lower=measure_min, upper=measure_max)
        mu = (measure_min + measure_max) / 2.0
        sigma = (measure_max - measure_min) / 12.0

        # likelihood
        measure_obs = pm.Normal('measure_obs', mu=mu, sd=sigma, observed=rhythm_dataset)

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
