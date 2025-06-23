import requests
import pandas as pd
import time
from datetime import datetime, timedelta

PROMETHEUS_URL = "http://localhost:9090"  
STEP = "15s"  # Prometheus scrape interval 

# Metrics to collect
METRICS = {
    "cpu_usage": '100 - avg by (instance) (rate(windows_cpu_time_total{mode="idle"}[1m])) * 100',

    "memory_available": 'windows_os_physical_memory_free_bytes',

    "disk_read" : 'rate(windows_logical_disk_read_bytes_total[1m])',
    "disk_read_c": 'rate(windows_logical_disk_read_bytes_total{volume="C:"}[1m])',
    "disk_read_d": 'rate(windows_logical_disk_read_bytes_total{volume="D:"}[1m])',
    "disk_read_e": 'rate(windows_logical_disk_read_bytes_total{volume="E:"}[1m])',

    "disk_write": 'rate(windows_logical_disk_write_bytes_total[1m])',
    "disk_write_c": 'rate(windows_logical_disk_write_bytes_total{volume="C:"}[1m])',
    "disk_write_d": 'rate(windows_logical_disk_write_bytes_total{volume="D:"}[1m])',
    "disk_write_e": 'rate(windows_logical_disk_write_bytes_total{volume="E:"}[1m])',

    "network_receive": 'rate(windows_net_bytes_received_total{nic="TP-Link Wireless USB Adapter"}[1m])',
    "network_transmit": 'rate(windows_net_bytes_sent_total{nic="TP-Link Wireless USB Adapter"}[1m])'
}


def query_range(metric_name, query, start, end, step):
    url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {
        "query": query,
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
        "step": step
    }
    response = requests.get(url, params=params)
    result = response.json()

    if result['status'] != 'success':
        print(f"Prometheus query failed for {metric_name}: {result.get('error', 'Unknown error')}")
        return pd.DataFrame()
    values = result['data']['result']
    
    if not values:
        print(f"No data for metric: {metric_name}")
        return pd.DataFrame()
    
    series = values[0]["values"]
    timestamps, metrics = zip(*series)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit='s'),
        metric_name: list(map(float, metrics))
    })
    return df

def main():
    print("Fetching metrics from Prometheus...")
    
    end = datetime.now()
    start = datetime.strptime("2025-06-17 08:47:00", "%Y-%m-%d %H:%M:%S")

    combined_df = None

    for metric_name, query in METRICS.items():
        print(f"Querying {metric_name}...")
        df = query_range(metric_name, query, start, end, STEP)
        
        if df.empty:
            continue

        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on="timestamp", how="outer")

    if combined_df is not None:
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
        combined_df.to_csv("train_prometheus_metrics2.csv", index=False)
        print("Metrics exported to prometheus_metrics.csv")
    else:
        print("No metrics were collected.")

if __name__ == "__main__":
    main()
