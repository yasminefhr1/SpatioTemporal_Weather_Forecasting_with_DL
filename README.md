# 🌦️ Spatio-Temporal Weather Forecasting with Deep Learning

<p align="center">
  <img width="1081" height="462" alt="image" src="https://github.com/user-attachments/assets/121f5509-7721-4580-9d03-7890be8a888c" />
</p>

## 🚀 Overview

This project focuses on **multivariate weather forecasting** using deep learning models on long-term historical data from **Météo-France**.

The goal is to predict **monthly temperature (TM)** over a **12-month horizon**, leveraging both:

- temporal dynamics,
- spatial information,
- and multivariate meteorological features.

The project is designed as a **reproducible experimental framework**, combining machine learning rigor with engineering practices.

---

## 🎯 Objectives

- Forecast temperature over long horizons (up to 12 months)
- Exploit multivariate dependencies between weather variables
- Evaluate models under **realistic constraints**:
  - temporal split
  - spatial generalization (unseen regions)
- Compare multiple deep learning architectures

---

## 📊 Dataset

- **Source**: Météo-France (via data.gouv.fr)
- **Granularity**: Monthly
- **Format**: `.csv.gz`
- **Time span**: ~1950 – 2025

### Features
- Temperature: `TM`, `TN`, `TX`
- Precipitation: `RR`
- Wind: `FFM`
- Sunshine: `INST`
- Spatial: `LAT`, `LON`, `ALTI`
- Seasonal encoding: `month_sin`, `month_cos`

---

## 🧠 Models

### Deep Learning Models
- **CNN 1D** — simple baseline
- **TCN** — temporal convolutional model
- **Seq2Seq + Attention** — encoder-decoder model
- **PatchTST** — transformer-based architecture

### Baselines
- Persistence (last value)
- Seasonal naive

---

## 🧪 Experimental Setup
Split protocol (spatial + temporal)

<table>
  <tr>
    <td style="width: 58%; vertical-align: top;">
      <img src="https://github.com/user-attachments/assets/dc4e9131-84b2-4811-9dad-7c2405032e5a"
           width="720"
           alt="Carte des départements de France" />
    </td>
    <td style="vertical-align: top; padding-left: 16px;">

### Carte de référence
> Les départements sont choisis proches géographiquement pour garder une cohérence climatique par exemple :(Rhône-Alpes / zone alpine).

### Split spatial (départements proches)
On entraîne sur une zone et on teste sur un département voisin **jamais vu** (généralisation spatiale).

- **Train** : `01, 38, 73`  
  (Ain, Isère, Savoie — zone proche / influence alpine)
- **Validation** : `74`  
  (Haute-Savoie — proche des départements train, utilisé pour réglages)
- **Test (unseen)** : `69`  
  (Rhône — voisin de 01 / proche de 38, jamais utilisé pendant entraînement)

### Split temporel strict
- **Train** : dates ≤ `train_end`  
- **Validation** : `train_end` < dates ≤ `val_end`  
- **Test** : dates > `val_end`

    </td>
  </tr>
</table>
---

## 📁 Project Structure

```
├── src/
│   ├── train.py
│   ├── data_processing.py
│   ├── models.py
│   ├── baselines_cnn.py
│   ├── baseline_tcn1d.py
│   ├── evaluate.py
│   ├── dashboard.py
│   └── benchmark_latency.py
├── tests/
├── scripts/
├── results/
├── data/
└── README.md
```

---
## 🏗️ Project Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/6ad50e54-3f66-4c53-bf7f-5c5b796e7afd" />

---

## ⚙️ Installation

```bash
git clone <YOUR_REPO_URL>
cd <PROJECT_NAME>

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\Activate.ps1  # Windows

pip install -r requirements.txt
```

---

## 📥 Data Download

```bash
python scripts/download_data.py --out_dir data/raw
```

---

## 🏋️ Training

Example (PatchTST):

```bash
python -m src.train --model patchtst --data_dir data/raw --train_depts 01,38,73 --val_depts 74 --test_depts 69 --train_end 2018-12-01 --val_end 2020-12-01 --history_len 24 --horizon 12 --epochs 50 --batch_size 64
```

---

## 📈 Evaluation

```bash
python -m src.evaluate --compare_all
```

Results are saved in:

```
results/comparisons/
```

---

## ⚡ Inference Latency

```bash
python -m src.benchmark_latency
```

---

## 🐳 Docker

```bash
docker build -t weather-forecast .
docker run --rm weather-forecast
```

---

## 🧩 Dashboard

```bash
python -m src.dashboard
```

<img width="1856" height="830" alt="image" src="https://github.com/user-attachments/assets/9a400450-eaee-477a-b775-85defde7a3e7" />
<img width="1866" height="707" alt="image" src="https://github.com/user-attachments/assets/902f6c02-2591-4f52-b605-cdf52e851d46" />

---


---

## 📉 Example Results

The table below summarizes representative runs obtained on the strict spatial-temporal split.

| Model     | MAE   | RMSE  | sMAPE (%) | MAPE (%) |
|----------|------:|------:|----------:|---------:|
| Seq2Seq  | 0.335 | 0.405 | 57.44     | 79.79    |
| PatchTST | 0.354 | 0.427 | 60.40     | 100.44   |
| TCN      | 0.365 | 0.426 | 56.17     | 96.01    |
| CNN      | 0.676 | 0.815 | 97.86     | 171.85   |

### Baseline reference
- **Persistence baseline**: MAE = 1.099, RMSE = 1.354
- **Seasonal naive baseline**: MAE = 0.265, RMSE = 0.342

### Quick interpretation
- **Seq2Seq** achieved the best performance among the deep learning models on this split.
- **PatchTST** and **TCN** also produced competitive results and clearly outperformed the persistence baseline.
- **CNN** remained weaker than the other deep architectures.
- The **seasonal naive baseline** is particularly strong on this setup, which suggests that monthly seasonality is a major driver of the signal and that outperforming it is challenging.

> Note: several PatchTST runs were executed; the table reports a representative stable run rather than the worst outlier.

---

## 🔍 Key Insights

- Performance decreases with longer forecast horizons
- Models tend to smooth extreme variations
- Spatial generalization remains challenging
- Transformer-based models show strong long-term capabilities

---

## 📌 Features

- Reproducible training pipeline
- Multiple deep learning models
- Spatial-temporal split strategy
- Automated evaluation
- Latency benchmarking
- Docker support
- Basic testing suite

---

## 📚 References

- PatchTST (Nie et al., 2023)
- Autoformer (Wu et al., 2021)
- TCN (Bai et al., 2018)
