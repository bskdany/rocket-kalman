import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cuinspace_el_blasto_cots.csv')
altitude = df['alt'].values
time = range(len(altitude))

plt.scatter(time, altitude)
plt.xlabel('Time (seconds)')
plt.ylabel('Altitude')
plt.title('Altitude over Time')
plt.show()

