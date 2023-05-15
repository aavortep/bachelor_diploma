import pymc3 as pm
import numpy as np
import pandas as pd

spotify_data = pd.read_csv('tempo_dataset.csv')
bpm_data = spotify_data['tempo']
rhythm_data = spotify_data['time_signature']
print(bpm_data, rhythm_data)
