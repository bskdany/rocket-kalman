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
    time = df['time'].values
    return time, pressure, altitude

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

H = np.array([[1.0, 0.0, 0.0]])

Q = np.array([[1.0, 0.0, 0.0],
              [0.0, 10.0, 0.0],
              [0.0, 0.0, 10.0]])

R = np.array([[pressure_resolution**2]])
I = np.eye(3)

def kalman_predict(x_k_minus_1, P_k_minus_1, dt_step):
    # State: [pressure, pressure_velocity, pressure_acceleration]
    # pressure_new = pressure_old + velocity_old * dt + 0.5 * acceleration_old * dt^2
    # velocity_new = velocity_old + acceleration_old * dt
    # acceleration_new = acceleration_old
    F_dt = np.array([[1.0, dt_step, 0.5 * dt_step**2],
                     [0.0, 1.0, dt_step],
                     [0.0, 0.0, 1.0]])
    x_pred = F_dt @ x_k_minus_1
    P_pred = F_dt @ P_k_minus_1 @ F_dt.T + Q
    return x_pred, P_pred

def kalman_update(z_k, x_k_minus_1, P_k_minus_1, dt_step):
    F_dt = np.array([[1.0, dt_step, 0.5 * dt_step**2],
                     [0.0, 1.0, dt_step],
                     [0.0, 0.0, 1.0]])
    x_pred = F_dt @ x_k_minus_1
    P_pred = F_dt @ P_k_minus_1 @ F_dt.T + Q
    
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x_k = x_pred + K @ (z_k - H @ x_pred)
    P_k = (I - K @ H) @ P_pred
    return x_k, P_k

def find_min_update_interval(time, pressure, max_error, max_steps=200):
    # Find minimum steps between Kalman updates such that prediction error < max_error
    # Looks through max_steps possible intervals

    errors = {}
    
    for step_interval in range(1, max_steps + 1):
        x = np.array([[pressure[0]],
                      [0.0],
                      [0.0]])
        P = np.array([[pressure_resolution**2 / 10, 0.0, 0.0],
                      [0.0, 100.0, 0.0],
                      [0.0, 0.0, 50.0]])
        
        predicted_pressure = []
        max_abs_error = 0.0
        
        for i in range(len(time)):
            dt_step = 0.0 if i == 0 else time[i] - time[i-1]
            
            if i % step_interval == 0:
                z = np.array([[pressure[i]]])
                x, P = kalman_update(z, x, P, dt_step)
                predicted_pressure.append(x[0,0])
            else:
                x, P = kalman_predict(x, P, dt_step)
                predicted_pressure.append(x[0,0])
            
            error = abs(predicted_pressure[-1] - pressure[i])
            max_abs_error = max(max_abs_error, error)
        
        errors[step_interval] = max_abs_error
        
        if max_abs_error <= max_error:
            return step_interval, errors
    
    return errors
    

time, pressure, _= load_easymini()
dt = np.mean(np.diff(time)) if len(time) > 1 else 0.07

predicted_pressure = []
x = np.array([[pressure[0]],
              [0.0],  # pressure_velocity
              [0.0]])  # pressure_acceleration
P = np.array([[pressure_resolution**2 / 10, 0.0, 0.0],
              [0.0, 100.0, 0.0],
              [0.0, 0.0, 50.0]])

# for i in range(len(time)):
#     dt_step = 0.0 if i == 0 else time[i] - time[i-1]
    
#     if(i % 100 == 0):
#         z = np.array([[pressure[i]]])
#         x, P = kalman_update(z, x, P, dt_step)
#         predicted_pressure.append(x[0,0])
#     else:
#         x, P = kalman_predict(x, P, dt_step)
#         predicted_pressure.append(x[0,0])

errors_dict = find_min_update_interval(time, pressure, 100)

fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))

intervals = list(errors_dict.keys())
errors = [float(errors_dict[i]) for i in intervals]

ax1.plot(intervals, errors, 'o-', markersize=3, linewidth=1)
ax1.set_xlabel('Update Interval (steps)')
ax1.set_ylabel('Maximum Prediction Error (Pa)')
ax1.set_title('Prediction Error vs Update Interval')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()