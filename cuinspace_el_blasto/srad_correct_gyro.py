import pandas as pd

# Read the gyro file
df = pd.read_csv('gyro_srad.csv')

# Scale x, y, z, and magnitude columns by 100
df['x'] = df['x'] * 100
df['y'] = df['y'] * 100
df['z'] = df['z'] * 100
df['magnitude'] = df['magnitude'] * 100

# Write the corrected data back
df.to_csv('gyro_srad.csv', index=False)

