import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def load_srad_alt():
    df = pd.read_csv('cuinspace_el_blasto/altitude_srad.csv')
    altitude = df['metres'].values
    time = df['mission_time'].values
    return time, altitude

def load_srad_pressure():
    df = pd.read_csv('cuinspace_el_blasto/pressure_srad.csv')
    pressure = df['pascals'].values
    time = df['mission_time'].values
    return time, pressure

def load_easymini():
    df = pd.read_csv('cuinspace_el_blasto/easymini.csv')
    pressure = df['pressure'].values
    altitude = df['altitude'].values
    time = range(len(altitude))
    return time, altitude

def plot_variance(time, data, window_size=50):
    df = pd.DataFrame({'time': time, 'data': data})
    std = df['data'].rolling(window=window_size).std()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['time'], df['data'], std, s=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Altitude')
    ax.set_zlabel('Standard Deviation')
    ax.set_title('Time, Altitude, and Standard Deviation')
    plt.show()


# altimiter datasheet: https://www.mouser.ca/datasheet/3/2640/1/ENG_DS_MS5607-02BA03_B5.pdf
# formulas and notation from: https://kalmanfilter.net/multiSummary.html

# x_k = state estimate at time k
# x_k_pred = predicted state estimate at time k
# k_k_minus_1 = state estimate at time k-1
# z_k = measurement at time k
# F = state transition matrix
# H = measurement matrix, maps the state to the measurement
# P_k = covariance matrix of the state estimate at time k
# Q = process noise covariance matrix
# R = measurement noise covariance matrix
# K_k = Kalman gain at time k

pressure_resolution = 2.4  # Pa (at OSR 4096)

F = np.array([[1.0]]) # state transition matrix, no change in pressure over time so stays the same
H = np.array([[1.0]]) # state to measurement matrix, pressure is measured directly
Q = np.array([[0.01]])  # no drift expected, very small value for numerical stability
R = np.array([[pressure_resolution**2]])  # measurement noise covariance matrix, pressure resolution from the datasheet (pascals^2)
I = np.array([[1.0]])

def kalman_update(z_k, x_k_minus_1, P_k_minus_1):
    # predict 
    x_pred = F @ x_k_minus_1
    P_pred = F @ P_k_minus_1 @ F.T + Q
    
    # update
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_k = x_pred + K @ (z_k - H @ x_pred)
    P_k = (I - K @ H) @ P_pred
    
    return x_k, P_k

x = np.array([[94810]]) # initial sample based on 10 measurements
P = np.array([[pressure_resolution**2 / 10]])  # covariance matrix of the initial sample, pressure resolution from the datasheet (pascals^2) / 10

time, pressure = load_srad_pressure()
easymini_time, easymini_pressure, easymini_altitude = load_easymini()

corrected_pressure = []

for i in range(len(time)):
    z = np.array([[pressure[i]]])
    x, P = kalman_update(z, x, P)
    corrected_pressure.append(x[0,0])

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(time, pressure, s=1, label='Pressure')
ax1.scatter(time, corrected_pressure, s=1, label='Corrected Pressure')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Pressure (Pa)')
ax1.set_title('Pressure over Time')
ax1.legend()

plt.tight_layout()
plt.show()