# LSTM Anomaly Detection — Industrial Mining Equipment

Projeto de detecção de anomalias em sensores industriais de mineração utilizando **LSTM Autoencoder** — uma abordagem de deep learning não supervisionada que aprende o comportamento normal da máquina e detecta automaticamente desvios.

## Como funciona
O modelo LSTM Autoencoder é treinado **apenas com dados normais**. Quando encontra uma sequência que não consegue reconstruir bem (erro alto), classifica como anomalia — sem precisar de rótulos de falha durante o treino.

## Dataset
- **UCI AI4I 2020 Predictive Maintenance Dataset**
- 10.000 registros de sensores industriais
- [Link para download](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## Pipeline
1. **Pré-processamento** — normalização MinMax + janelas deslizantes (30 timesteps)
2. **LSTM Autoencoder** — encoder comprime a sequência, decoder reconstrói
3. **Detecção** — erro de reconstrução (MSE) acima do threshold = anomalia
4. **Avaliação** — Classification Report + ROC-AUC + visualizações

## Estrutura
```
lstm-anomaly-detection/
├── data/               # Dataset e arrays processados (.npy)
├── src/
│   ├── preprocessing.py   # Normalização e janelas deslizantes
│   ├── model.py           # Arquitetura e treinamento da LSTM
│   └── detect.py          # Detecção de anomalias e avaliação
├── outputs/
│   ├── plots/             # Gráficos gerados
│   ├── scaler.pkl         # Scaler salvo
│   └── lstm_autoencoder.keras  # Modelo treinado
├── requirements.txt
└── README.md
```

##  Como rodar
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src/preprocessing.py
python src/model.py
python src/detect.py
```

## Resultados
- **Accuracy:** 0.92
- **ROC-AUC:** 0.53
- Modelo detecta bem o comportamento normal (precision 0.97)
- Dataset sem ordem temporal real limita a performance na classe de anomalia

## Tecnologias
Python · TensorFlow/Keras · Pandas · Scikit-learn · Matplotlib · Seaborn
```

Depois roda:

```bash
git add README.md
git commit -m "docs: adiciona README"
git push
```

Projeto completo no GitHub! 