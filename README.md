<!-- PROJECT HEADER -->
<p align="center">
  <img src="https://img.shields.io/github/stars/Bilal-ahmad8/Advanced-Monitoring-System?style=flat-square" alt="GitHub Stars"/>
  <img src="https://img.shields.io/github/license/Bilal-ahmad8/Advanced-Monitoring-System?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/docker/pulls/library/python?style=flat-square" alt="Docker Pulls"/>
</p>

<h1 align="center">📊 Advanced Monitoring System with ML-based Anomaly Detection</h1>
<p align="center"><b>Real-time, ML-driven anomaly detection and monitoring system for server metrics, fully dockerized and production-ready.</b></p>

---

## 🚦 Quick Links

- [Features](#-features)
- [Project Structure](#-project-structure)
- [ML Model Overview](#-ml-model-overview)
- [Grafana Dashboard](#-grafana-dashboard)
- [Prometheus Setup](#-prometheus)
- [Docker Orchestration](#-docker-orchestration)
- [How to Run](#-how-to-run)
- [Enhancements](#-optional-enhancements)
- [System Architecture](#-system-architecture)
- [Contact](#-contact)

---

## 🚀 Features

- Real-time anomaly detection on time-series server metrics
- Custom ML inference container with buffered startup
- Automated data scraping, metric pushing, and dashboarding
- Grafana with threshold lines, update intervals, and provisioned dashboards
- Fully dockerized and scalable

---

## 🧱 Project Structure

```
Advanced-Monitoring-System/
├── infra/                        # Infrastructure & configs
│   ├── docker-compose.yml
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules/
│   │       └── alert_rules.yml (optional)
│   ├── grafana/
│   │   └── provisioning/
│   │       └── dashboards/
│   │           └── monitoring-dashboard.json
|   |
│   |
├── ml/
|   ├──other_modules
│   ├── src/
│   │   └── pipeline/
│   │       └── inference_pipeline.py
│   ├── artifact/
│   │   └── model.pt              # Trained model
│   └── Dockerfile                # For ML container
```

---

## 🧠 ML Model Overview

- **Model Type:** LSTM AutoEncoder
- **Purpose:** Detect anomalies in server metrics (e.g., CPU, Memory)
- **Inference Frequency:** Every 10 minutes
- **Buffer:** Pre-loaded with 19 rows to avoid cold start

---

## 📈 Grafana Dashboard

- Line plots of `anomaly_score` `CPU metrics` and bar plot for `is_anomaly`
- Threshold lines added for visibility
- Custom update interval: 30s
- Provisioned dashboard via JSON

---

## 🔄 Prometheus

- Retains data for 15 days
- Uses Pushgateway to receive `anomaly_score`

---

## 🐳 Docker Orchestration

All containers are started using `docker-compose.yml`:

- `prometheus`
- `pushgateway`
- `grafana`
- `ml_inference` (depends on Prometheus)

Proper file/folder permissions ensured via WSL metadata or UID `65534`.

---

## ✅ How to Run

```bash
# From project root:
cd infra

docker compose up --build
```

- Visit Prometheus: [http://localhost:9090](http://localhost:9090)
- Visit Grafana: [http://localhost:3000](http://localhost:3000)
  - Login: `admin` / `admin`
  - Imported dashboard should load automatically

---

## 🛠 Optional Enhancements

- Alertmanager integration for Slack/email alerts
- MLflow for model versioning
- Streamlit UI for anomaly exploration
- Cloud deployment using ECS/GKE

---

## 📐 System Architecture


```
┌────────────┐    Metrics    ┌──────────────┐    Scrape    ┌────────────┐
│  Exporter  ├──────────────▶│ ML Inference ├─────────────▶│ Pushgateway│
└────────────┘               └──────────────┘              └─────┬──────┘
                                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │ Prometheus   │
                                                          └─────┬────────┘
                                                                  │
                                                                  ▼
                                                           ┌────────────┐
                                                           │  Grafana   │
                                                           └────────────┘
```
**Flow:**  
Exporters collect metrics → ML Inference processes & pushes anomaly scores to Pushgateway → Prometheus scrapes from Pushgateway → Grafana visualizes in dashboards.  
_No dedicated database is used; all metrics flow through Prometheus._

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to [open an issue](https://github.com/Bilal-ahmad8/Advanced-Monitoring-System/issues) or submit a pull request.

---

## 📬 Contact

For questions, open an issue or reach out to [Bilal-ahmad8](https://github.com/Bilal-ahmad8).

---

> _Made by Bilal-ahmad8_

<h1 align="center">📊 Advanced Monitoring System with ML-based Anomaly Detection</h1>
<p align="center"><b>Real-time, ML-driven anomaly detection and monitoring system for server metrics, fully dockerized and production-ready.</b></p>

---

## 🚦 Quick Links

- [Features](#-features)
- [Project Structure](#-project-structure)
- [ML Model Overview](#-ml-model-overview)
- [Grafana Dashboard](#-grafana-dashboard)
- [Prometheus Setup](#-prometheus)
- [Docker Orchestration](#-docker-orchestration)
- [How to Run](#-how-to-run)
- [Enhancements](#-optional-enhancements)
- [System Architecture](#-system-architecture)
- [Contact](#-contact)

---

## 🚀 Features

- Real-time anomaly detection on time-series server metrics
- Custom ML inference container with buffered startup
- Automated data scraping, metric pushing, and dashboarding
- Grafana with threshold lines, update intervals, and provisioned dashboards
- Fully dockerized and scalable

---

## 🧱 Project Structure

```
Advanced-Monitoring-System/
├── infra/                        # Infrastructure & configs
│   ├── docker-compose.yml
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules/
│   │       └── alert_rules.yml (optional)
│   ├── grafana/
│   │   └── provisioning/
│   │       └── dashboards/
│   │           └── monitoring-dashboard.json
|   |
│   |
├── ml/
|   ├──other_modules
│   ├── src/
│   │   └── pipeline/
│   │       └── inference_pipeline.py
│   ├── artifact/
│   │   └── model.pt              # Trained model
│   └── Dockerfile                # For ML container
```

---

## 🧠 ML Model Overview

- **Model Type:** LSTM AutoEncoder
- **Purpose:** Detect anomalies in server metrics (e.g., CPU, Memory)
- **Inference Frequency:** Every 10 minutes
- **Buffer:** Pre-loaded with 19 rows to avoid cold start

---

## 📈 Grafana Dashboard

- Line plots of `anomaly_score` `CPU metrics` and bar plot for `is_anomaly`
- Threshold lines added for visibility
- Custom update interval: 30s
- Provisioned dashboard via JSON

---

## 🔄 Prometheus

- Retains data for 15 days
- Uses Pushgateway to receive `anomaly_score`

---

## 🐳 Docker Orchestration

All containers are started using `docker-compose.yml`:

- `prometheus`
- `pushgateway`
- `grafana`
- `ml_inference` (depends on Prometheus)

Proper file/folder permissions ensured via WSL metadata or UID `65534`.

---

## ✅ How to Run

```bash
# From project root:
cd infra

docker compose up --build
```

- Visit Prometheus: [http://localhost:9090](http://localhost:9090)
- Visit Grafana: [http://localhost:3000](http://localhost:3000)
  - Login: `admin` / `admin`
  - Imported dashboard should load automatically

---

## 🛠 Optional Enhancements

- Alertmanager integration for Slack/email alerts
- MLflow for model versioning
- Streamlit UI for anomaly exploration
- Cloud deployment using ECS/GKE

---

## 📐 System Architecture


```
┌────────────┐    Metrics    ┌──────────────┐    Scrape    ┌────────────┐
│  Exporter  ├──────────────▶│ ML Inference ├─────────────▶│ Pushgateway│
└────────────┘               └──────────────┘              └─────┬──────┘
                                                                  │
                                                                  ▼
                                                          ┌──────────────┐
                                                          │ Prometheus   │
                                                          └─────┬────────┘
                                                                  │
                                                                  ▼
                                                           ┌────────────┐
                                                           │  Grafana   │
                                                           └────────────┘
```
**Flow:**  
Exporters collect metrics → ML Inference processes & pushes anomaly scores to Pushgateway → Prometheus scrapes from Pushgateway → Grafana visualizes in dashboards.  
_No dedicated database is used; all metrics flow through Prometheus._

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to [open an issue](https://github.com/Bilal-ahmad8/Advanced-Monitoring-System/issues) or submit a pull request.

---

## 📬 Contact

For questions, open an issue or reach out to [Bilal-ahmad8](https://github.com/Bilal-ahmad8).

---

> _Made by Bilal-ahmad8_
