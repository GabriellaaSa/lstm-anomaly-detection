import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
WINDOW = 30  # janela deslizante de 30 timesteps

def load_and_scale(path: str):
    df = pd.read_csv(path)
    data = df[FEATURES].values

    # Normaliza entre 0 e 1
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, "outputs/scaler.pkl")

    return data_scaled, df["Machine failure"].values

def create_windows(data: np.ndarray, window: int = WINDOW):
    X = []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
    return np.array(X)

if __name__ == "__main__":
    data_scaled, labels = load_and_scale("data/ai4i2020.csv")

    # Treina apenas com dados normais (sem falha)
    normal_idx = np.where(labels == 0)[0]
    normal_data = data_scaled[normal_idx]

    X_train = create_windows(normal_data)
    X_all   = create_windows(data_scaled)
    y_all   = labels[WINDOW:]

    np.save("data/X_train.npy", X_train)
    np.save("data/X_all.npy",   X_all)
    np.save("data/y_all.npy",   y_all)

    print(f"✅ X_train: {X_train.shape} | X_all: {X_all.shape}")