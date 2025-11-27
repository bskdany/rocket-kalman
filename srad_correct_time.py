import pandas as pd
import glob
import os

srad_files = glob.glob('cuinspace_el_blasto/*srad*.csv')

for file in srad_files:
    df = pd.read_csv(file)
    df['mission_time'] = (df['mission_time'] - 5283.71).round(2)
    base = os.path.splitext(file)[0]
    ext = os.path.splitext(file)[1]
    new_file = f'{base}_corrected{ext}'
    df.to_csv(new_file, index=False)

