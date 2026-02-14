import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- PART 1: THE GEOMETRY ENGINE (categorizing shot zones)
def simulate_new_zones(row, new_arc_dist):
    # Nba API sometimes use decifeet units for distance, so we convert to feet if needed
    dist = row['SHOT_DISTANCE'] / 10 if row['SHOT_DISTANCE'] > 100 else row['SHOT_DISTANCE']
    x, y = row['LOC_X'], row['LOC_Y']
    
    # 1. Constant Zones: These stay the same regardless of the 3pt line
    if dist < 8: return 'Paint'
    if dist < 16: return 'Short Mid-Range'

    # 2. Variable Geometry: Define the new line's shape
    # Corner 3s are traditionally ~1.75ft closer than the top of the arc
    new_corner_dist = new_arc_dist - 1.75
    is_corner_area = y < 92.5 # The height of the straight baseline lines
    
    # Check if the shot is behind our NEW simulated line
    is_three = (is_corner_area and dist >= new_corner_dist) or (not is_corner_area and dist >= new_arc_dist)

    if is_three:
        if is_corner_area: return 'Corner 3 (Sim)'
        
        # Slicing the 'Above the Break' area into Wings and Top using polar coordinates
        angle = np.arctan2(y, x) * 180 / np.pi
        if (22 < angle < 70) or (110 < angle < 158):
            return 'Wing 3 (Sim)'
        else:
            return 'Top of Key 3 (Sim)'
    
    # 3. Expansion: Anything between 16ft and the new line becomes Long Mid-Range
    return 'Long Mid-Range (Sim)'



# --- PART 2: THE DYNAMIC VISUALIZATION ENGINE ---
def draw_court(ax, line_dist, zone_stats):
    # Standard court pieces
    ax.add_patch(plt.Circle((0, 0), 7.5, color='black', fill=False)) # Rim
    ax.add_patch(plt.Rectangle((-80, -47.5), 160, 190, color='black', fill=False)) # Paint
    
    # DYNAMIC 3-POINT LINE (Calculated in decifeet for the plot)
    # The 'width' and 'height' in patches.Arc are diameters (2 * radius)
    arc_diameter = line_dist * 10 * 2
    ax.add_patch(patches.Arc((0, 0), arc_diameter, arc_diameter, 
                             theta1=22, theta2=158, linewidth=3, color='blue'))
    
    # Dynamic Corner Lines (Ends at x = +/- 220)
  # 1. Calculate the corner distance and the intersection point
    # corn_x is the x-coordinate of the straight line
    corn_x = (line_dist - 1.75) * 10
    # r is the radius of the arc in decifeet
    r = line_dist * 10
    
    if corn_x <= 250:
        # Calculate where the straight line meets the arc: y = sqrt(r^2 - x^2)
        # This ensures the line ends exactly where the curve starts
        y_intersect = np.sqrt(r**2 - corn_x**2)
        
        # 2. Draw the Straight Lines
        # We start at the baseline (-47.5) and end exactly at y_intersect
        ax.add_artist(plt.Line2D([-corn_x, -corn_x], [-47.5, y_intersect], linewidth=3, color='blue'))
        ax.add_artist(plt.Line2D([corn_x, corn_x], [-47.5, y_intersect], linewidth=3, color='blue'))
        
        # 3. Calculate the Arc Angles (theta)
        # theta is the angle from the center (0,0) to the intersection point
        theta_angle = np.degrees(np.arcsin(y_intersect / r))
        
        arc_dia = r * 2
        ax.add_patch(patches.Arc((0, 0), arc_dia, arc_dia, 
                                 theta1=theta_angle, theta2=180-theta_angle, 
                                 linewidth=3, color='blue'))

    # ADDING PPS LABELS DIRECTLY ON THE MAP
    # We pull these from the 'zone_stats' we calculated earlier
    labels = {
        'Paint': (0, 30),
        'Short Mid-Range': (0, 120),
        'Long Mid-Range (Sim)': (0, 200),
        'Top of Key 3 (Sim)': (0, 280),
        'Wing 3 (Sim)': (150, 240),
        'Corner 3 (Sim)': (230, 40)
    }
    
    for zone, pos in labels.items():
        if zone in zone_stats:
            pps = zone_stats[zone]
            # Color coding the text: Red for low efficiency, Green for high
            color = 'green' if pps > 1.1 else ('orange' if pps > 0.9 else 'red')
            ax.text(pos[0], pos[1], f"{zone}\n{pps:.2f} PPS", 
                    ha='center', va='center', fontweight='bold', color=color,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlim(-250, 250)
    ax.set_ylim(-47.5, 422.5)
    return ax





# --- PART 3: THE STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="NBA Geometry Sim", layout="wide")
st.title("🏀 NBA Rule Change: The Geometry Simulator")

# 1. Sidebar for User Control
st.sidebar.header("Settings")
line_dist = st.sidebar.slider("Move 3-Point Line (ft)", 22.0, 32.0, 23.75)

# 2. Data Loading (Fast & Cached)
@st.cache_data
def load_data():
    # Make sure your parquet file is in the same folder as this script!
    return pd.read_parquet('league_shots.parquet') 

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Ensure 'league_shots.parquet' is in this folder.")
    st.stop()

# 3. Processing: Apply our logic based on the user's slider input
df['SIM_ZONE'] = df.apply(simulate_new_zones, args=(line_dist,), axis=1)
# Calculate Simulated Points Per Shot (PPS)
df['SIM_PTS'] = df['SIM_ZONE'].apply(lambda x: 3 if '3 (Sim)' in x else 2) * df['SHOT_MADE_FLAG']

# 4. Layout: Display Stats and Plot side-by-side
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Efficiency by Zone")
    
    # 1. Group and calculate mean
    zone_stats = df.groupby('SIM_ZONE')['SIM_PTS'].mean().sort_values(ascending=False)
    
    # 2. CONVERT TO DATAFRAME (This fixes the AttributeError)
    zone_df = zone_stats.to_frame(name='PPS')
    
    # 3. Apply style to the new DataFrame
    st.dataframe(zone_df.style.background_gradient(cmap='RdYlGn', vmin=0.7, vmax=1.3))

with col2:
    st.subheader("Spatial Efficiency Map")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # We pass the slider value and the stats directly to the drawing function
    # This ensures the map updates whenever the slider moves
    draw_court(ax, line_dist, zone_stats)
    
    # Remove the axis ticks for a cleaner 'app' look
    ax.set_xticks([])
    ax.set_yticks([])
    
    st.pyplot(fig)