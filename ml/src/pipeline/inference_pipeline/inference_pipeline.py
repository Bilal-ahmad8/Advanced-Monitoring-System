from src.components.inference.inference import Inferencing
import pandas as pd
import time
from collections import deque


SEQUENCE_LENGTH = 20


if __name__ == '__main__':

    obj = Inferencing()
    print("Starting real-time inference loop...")
    history = [] 
    seed_df = pd.read_csv("buffer_data.csv")
    seed_df['timestamp'] = pd.to_datetime(seed_df['timestamp'])
    history = seed_df.to_dict("records")

    while True:
        incoming_df = (obj.query_latest_metrics())
        print(incoming_df)
        if incoming_df is not None:
            incoming = incoming_df.to_dict("records")
        #     # Merging and droping duplicate timestamps
            combined = {entry["timestamp"]: entry for entry in history + incoming}
        #     # Sorting by timestamp and keep only latest SEQUENCE_LENGTH
            history = sorted(combined.values(), key=lambda x: x["timestamp"])[-SEQUENCE_LENGTH:]
            print(f'Current Data length: {len(history)}')

        if len(history) == SEQUENCE_LENGTH:
            data = pd.DataFrame(history)
            score = obj.inference(data=data)
            print(f'Anomaly score {score}')
            obj.push_score_to_gateway(score=score)
        
        time.sleep(30)


