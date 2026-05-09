import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ── Carrega dados ─────────────────────────────────────────────
X_train = np.load("data/X_train.npy")
print(f"Shape treino: {X_train.shape}")  # (amostras, 30, 5)

WINDOW   = X_train.shape[1]
N_FEATS  = X_train.shape[2]

# ── Arquitetura LSTM Autoencoder ──────────────────────────────
inputs  = Input(shape=(WINDOW, N_FEATS))
# Encoder
x = LSTM(64, activation="tanh", return_sequences=False)(inputs)
# Bottleneck
x = RepeatVector(WINDOW)(x)
# Decoder
x = LSTM(64, activation="tanh", return_sequences=True)(x)
outputs = TimeDistributed(Dense(N_FEATS))(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")
model.summary()

# ── Treinamento ───────────────────────────────────────────────
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[es],
    shuffle=True
)

# ── Curva de loss ─────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"],     label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("LSTM Autoencoder — Curva de Loss")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/loss_curve.png", dpi=150)
plt.show()

# ── Salva modelo ──────────────────────────────────────────────
model.save("outputs/lstm_autoencoder.keras")
print("✅ Modelo salvo em outputs/lstm_autoencoder.keras")