print("--- 1. SCRIPT STARTING ---")
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

print("--- 2. LIBRARIES LOADED ---")
app = Flask(__name__)
CORS(app) # Allows your React frontend to connect

# --- DATA LOADING SECTION ---
# This part ensures we don't crash silently if the file is missing
file_name = 'league_shots.parquet'
print(f"--- 3. LOOKING FOR DATA IN: {os.getcwd()} ---")

if os.path.exists(file_name):
    print(f"--- 4. FOUND {file_name}. LOADING... ---")
    df = pd.read_parquet(file_name)
    print("--- 5. DATA LOADED SUCCESSFULLY ---")
else:
    print(f"--- 4. ERROR: {file_name} NOT FOUND! ---")
    print("Check if the file is in the same folder as this script.")
    exit()

# --- THE GEOMETRY ENGINE ---
def simulate_new_zones(row, new_arc_dist):
    # Handle NBA API decifeet units
    dist = row['SHOT_DISTANCE'] / 10 if row['SHOT_DISTANCE'] > 100 else row['SHOT_DISTANCE']
    x, y = row['LOC_X'], row['LOC_Y']
    x_abs = abs(x)
    
    # 1. Static Zones
    if dist < 8: return 'Paint'
    if dist < 16: return 'Short Mid-Range'

    # 2. Dynamic Corner Logic
    new_corner_dist = new_arc_dist - 1.75
    is_corner_area = y < 92.5 # Traditional 'break' height
    
    if is_corner_area:
        # Corner 3s move out as the slider moves
        if x_abs >= (new_corner_dist * 10) and x_abs <= 250:
            return 'Corner 3 (Sim)'
    else:
        # Above the break shots
        if dist >= new_arc_dist:
            angle = np.arctan2(y, x) * 180 / np.pi
            if (22 < angle < 70) or (110 < angle < 158):
                return 'Wing 3 (Sim)'
            else:
                return 'Top of Key 3 (Sim)'
    
    return 'Long Mid-Range (Sim)'

# --- THE API ENDPOINT ---
# --- [REPLACE EVERYTHING FROM @app.route DOWNWARDS WITH THIS] ---

@app.route('/api/calculate', methods=['POST'])
def calculate():
    print("--- API REQUEST RECEIVED ---")
    try:
        data = request.json
        line_dist = data.get('threePtDistance', 23.75)
        
        # Process the simulation
        df_sim = df.copy()
        df_sim['SIM_ZONE'] = df_sim.apply(simulate_new_zones, args=(line_dist,), axis=1)
        
        # Calculate PPS: (3pts if zone is a 3, else 2pts) * Shot Made (1 or 0)
        df_sim['SIM_PTS'] = df_sim['SIM_ZONE'].apply(
            lambda x: 3 if '3 (Sim)' in x else 2
        ) * df_sim['SHOT_MADE_FLAG']

        # Group results for the frontend
        # FIXED TYPO HERE: Changed 'dfsim' to 'df_sim'
        stats = df_sim.groupby('SIM_ZONE').agg({
            'SIM_PTS': 'mean',
            'SHOT_MADE_FLAG': 'count'
        }).rename(columns={'SIM_PTS': 'pps', 'SHOT_MADE_FLAG': 'volume'}).to_dict(orient='index')
        
        print("--- CALCULATION COMPLETE. SENDING RESULTS ---")
        return jsonify(stats)
        
    except Exception as e:
        print(f"ERROR IN CALCULATION: {e}")
        return jsonify({"error": str(e)}), 500

# --- THESE LINES MUST BE FLUSH LEFT (NOT INDENTED) ---
print("--- 6. STARTING SERVER ON PORT 5000 ---")
if __name__ == '__main__':
    app.run(debug=True, port=5000)