import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(os.path.join(parent, "src"))
from simple_ml.linear_model import LinearRegression


def run_real_world_test():
    data_frame = pd.read_csv("FuelConsumption.csv")
    X = data_frame[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]].values
    y = data_frame["CO2EMISSIONS"].values
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std

    model = LinearRegression(learning_rate=0.01, n_iters=3000)
    model.fit(X_scaled, y)

    print("-" * 24)
    print(f"X {X}  weights {model.weights} bias {model.bias}")
    y_pred = model.predict(X_scaled)
    y_pred = y_pred.flatten()

    print(f"{'real (y)':<15} | {' (y_pred)':<15} | {' (Error)':<15}")
    print("-" * 50)

    for i in range(5):
        actual = y[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        print(f"{actual:<15.2f} | {predicted:<15.2f} | {error:<15.2f}")

    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    print(f"(RMSE): {rmse:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, color="BLUE", alpha=0.5, label="real data")
    plt.scatter(X[:, 0], y_pred, color="RED", alpha=0.5, label="predicted data")
    plt.xlabel("Engine size")
    plt.ylabel("co2 emission")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_real_world_test()

# Columns: [MODELYEAR, MAKE, MODEL, VEHICLECLASS, ENGINESIZE, CYLINDERS,
# TRANSMISSION, FUELTYPE, FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY,
# FUELCONSUMPTION_COMB, FUELCONSUMPTION_COMB_MPG, CO2EMISSIONS]
