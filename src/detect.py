import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, roc_auc_score

# ── Carrega modelo e dados ────────────────────────────────────
model  = load_model("outputs/lstm_autoencoder.keras")
X_all  = np.load("data/X_all.npy")
y_true = np.load("data/y_all.npy")

# ── Erro de reconstrução ──────────────────────────────────────
X_pred = model.predict(X_all, verbose=0)
mse    = np.mean(np.power(X_all - X_pred, 2), axis=(1, 2))

# ── Threshold automático (percentil 95 dos erros) ────────────
threshold = np.percentile(mse, 95)
print(f"Threshold: {threshold:.6f}")

y_pred = (mse > threshold).astype(int)

# ── Avaliação ─────────────────────────────────────────────────
print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalia"]))
print(f"ROC-AUC: {roc_auc_score(y_true, mse):.4f}")

# ── Gráfico 1: Erro de reconstrução ao longo do tempo ────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(mse, color="steelblue", linewidth=0.8, label="Erro de Reconstrução (MSE)")
axes[0].axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.5f})")
axes[0].fill_between(range(len(mse)), mse, threshold,
                     where=(mse > threshold), color="red", alpha=0.3, label="Anomalia detectada")
axes[0].set_ylabel("MSE")
axes[0].set_title("Erro de Reconstrução — LSTM Autoencoder")
axes[0].legend()

# ── Gráfico 2: Rótulo real vs detectado ──────────────────────
axes[1].plot(y_true, color="gray",   linewidth=0.6, label="Falha Real",      alpha=0.7)
axes[1].plot(y_pred, color="tomato", linewidth=0.6, label="Anomalia Predita", alpha=0.7)
axes[1].set_ylabel("0 = Normal | 1 = Anomalia")
axes[1].set_xlabel("Timestep")
axes[1].set_title("Comparação: Falha Real vs Anomalia Detectada")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/plots/anomaly_detection.png", dpi=150)
plt.show()

# ── Gráfico 3: Distribuição do erro por classe ────────────────
plt.figure(figsize=(8, 4))
plt.hist(mse[y_true == 0], bins=80, alpha=0.6, color="steelblue", label="Normal")
plt.hist(mse[y_true == 1], bins=80, alpha=0.6, color="tomato",    label="Falha real")
plt.axvline(threshold, color="black", linestyle="--", label="Threshold")
plt.xlabel("MSE")
plt.ylabel("Frequência")
plt.title("Distribuição do Erro de Reconstrução por Classe")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/error_distribution.png", dpi=150)
plt.show()