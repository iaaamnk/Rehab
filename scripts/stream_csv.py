import asyncio
import websockets
import json
import pandas as pd
import time
import os

async def stream_data():
    uri = "ws://localhost:8000/ws/stream"
    
    # Load the CSV we generated
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'simulated_poses.csv')
    df = pd.read_csv(csv_path)
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Streaming frames...")
            
            # Read first ~200 frames for a quick test (mix of healthy and compensatory)
            for idx, row in df.head(300).iterrows():
                
                # We need to construct the pose dict the API expects
                pose_dict = {}
                for col in df.columns:
                    if col.startswith('j'):
                        pose_dict[col] = float(row[col])
                        
                await websocket.send(json.dumps(pose_dict))
                
                # Get feedback (optional, we could just send)
                response = await websocket.recv()
                
                # Sleep to simulate ~30 FPS
                time.sleep(1/30.0)
                
            print("Finished streaming.")
    except Exception as e:
        print(f"Connection failed: {e}. Is the backend running?")

if __name__ == "__main__":
    asyncio.run(stream_data())
