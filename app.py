import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
from matplotlib.lines import Line2D

# ── PART 1: GEOMETRY ENGINE ───────────────────────────────────────────────────
def simulate_new_zones(row, new_arc_dist):
    dist = row['SHOT_DISTANCE'] / 10 if row['SHOT_DISTANCE'] > 100 else row['SHOT_DISTANCE']
    x, y = row['LOC_X'], row['LOC_Y']

    if dist < 8:  return 'Paint'
    if dist < 16: return 'Short Mid-Range'

    new_corner_dist = new_arc_dist - 1.75
    is_corner_area  = y < 92.5

    is_three = (is_corner_area and dist >= new_corner_dist) or \
               (not is_corner_area and dist >= new_arc_dist)

    if is_three:
        if is_corner_area:
            return 'Corner 3 (Sim)'
        angle = np.arctan2(y, x) * 180 / np.pi
        if (22 < angle < 70) or (110 < angle < 158):
            return 'Wing 3 (Sim)'
        return 'Top of Key 3 (Sim)'

    return 'Long Mid-Range (Sim)'


# ── PART 2: HELPER FUNCTIONS ──────────────────────────────────────────────────
def pps_to_hsl(pps, vmin=0.6, vmax=1.2):
    clamped = max(0.0, min(1.0, (pps - vmin) / (vmax - vmin)))
    hue = int(clamped * 120)
    return f"hsl({hue}, 75%, 42%)"


# ── PART 3: COURT ─────────────────────────────────────────────────────────────
ZONE_ORDER = [
    'Paint',
    'Short Mid-Range',
    'Long Mid-Range (Sim)',
    'Wing 3 (Sim)',
    'Corner 3 (Sim)',
    'Top of Key 3 (Sim)',
]

ZONE_DISPLAY = {
    'Paint':               'Paint',
    'Short Mid-Range':     'Short Mid-Range',
    'Long Mid-Range (Sim)':'Long Mid-Range',
    'Wing 3 (Sim)':        'Wing 3',
    'Corner 3 (Sim)':      'Corner 3',
    'Top of Key 3 (Sim)':  'Top of Key 3',
}


def draw_hexbin_chart(line_dist, df):
    fig, ax = plt.subplots(figsize=(8.5, 7.5), facecolor='#f5e6c8')
    ax.set_facecolor('#f5e6c8')

    # ── Paint fill ─────────────────────────────────────────────────────────
    ax.add_patch(Rectangle((-80, -47.5), 160, 190, facecolor='#e8c88a', alpha=0.25, zorder=0))

    # ── Hexbin of expected value per shot ──────────────────────────────────
    df_plot = df[df['SHOT_ATTEMPTED_FLAG'] == 1]
    hb = ax.hexbin(
        df_plot['LOC_X'], df_plot['LOC_Y'],
        C=df_plot['SIM_PTS'],
        reduce_C_function=np.mean,
        gridsize=25,
        cmap='RdYlGn', vmin=0.5, vmax=1.5,
        mincnt=5,
        alpha=0.92,
        extent=[-250, 250, -47.5, 422.5],
        edgecolors='white', linewidths=0.3,
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.01, aspect=20)
    cb.set_label('Expected pts / attempt', fontsize=9, rotation=270, labelpad=14, color='#1a1209')
    cb.ax.tick_params(labelsize=8, colors='#1a1209')

    # ── Helper: draw court line ────────────────────────────────────────────
    def cl(xs, ys, color='#1a1a1a', lw=1.5, **kw):
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round', **kw)

    # ── Static court markings ──────────────────────────────────────────────
    cl([-250, 250, 250, -250, -250], [-47.5, -47.5, 422.5, 422.5, -47.5], color='#1a1a1a')
    cl([-80, -80, 80, 80], [-47.5, 142.5, 142.5, -47.5])
    cl([-80, 80], [142.5, 142.5])
    ax.add_patch(Arc((0, 142.5), 120, 120, theta1=0, theta2=180, color='#1a1a1a', lw=1.5))
    ax.add_patch(Arc((0, 0), 80, 80, theta1=0, theta2=180, color='#1a1a1a', lw=1.5))
    ax.add_patch(Circle((0, 0), 7.5, fill=False, color='#1a1a1a', lw=1.5))
    cl([-30, 30], [-7.5, -7.5])

    # ── Helper: draw a 3-point line given distance ─────────────────────────
    def draw_3pt(dist, color, lw, ls='-'):
        rr = dist * 10
        cx = (dist - 1.75) * 10
        if cx <= 250:
            yi = np.sqrt(rr**2 - cx**2)
            ti = np.degrees(np.arctan2(yi, cx))
            cl([cx, cx], [-47.5, yi], color=color, lw=lw, ls=ls)
            cl([-cx, -cx], [-47.5, yi], color=color, lw=lw, ls=ls)
            ax.add_patch(Arc((0, 0), rr*2, rr*2,
                             theta1=ti, theta2=180-ti, color=color, lw=lw, ls=ls))
        else:
            ys = np.sqrt(rr**2 - 250**2)
            ts = np.degrees(np.arccos(250.0 / rr))
            cl([250, 250], [-47.5, ys], color=color, lw=lw, ls=ls)
            cl([-250, -250], [-47.5, ys], color=color, lw=lw, ls=ls)
            ax.add_patch(Arc((0, 0), rr*2, rr*2,
                             theta1=ts, theta2=180-ts, color=color, lw=lw, ls=ls))

    # Baseline 23.75 ft line (gray dashed reference)
    if line_dist != 23.75:
        draw_3pt(23.75, '#888888', lw=1.5, ls='--')

    # New 3-point line (solid blue)
    draw_3pt(line_dist, '#1a6bb5', lw=2.5)

    # ── Legend for the two lines ───────────────────────────────────────────
    handles = [Line2D([0], [0], color='#1a6bb5', lw=2.5, label=f'{line_dist} ft (new)')]
    if line_dist != 23.75:
        handles.append(Line2D([0], [0], color='#888888', lw=1.5, ls='--', label='23.75 ft (current)'))
    ax.legend(handles=handles, bbox_to_anchor=(0.99, 0.97), loc='upper right', fontsize=8, frameon=False, labelcolor='#1a1209')

    ax.set_xlim(-260, 260)
    ax.set_ylim(-60, 470)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Expected Value per Shot — {line_dist} ft arc', loc='left', fontsize=13, fontweight='bold', pad=10, color='#1a1209')
    fig.text(0.01, 0.01, 'Data: NBA.com  |  Sim by arc distance', fontsize=7, color='#8b7040')
    fig.tight_layout()
    return fig


# ── PART 4: STREAMLIT UI ──────────────────────────────────────────────────────
st.set_page_config(page_title="NBA Geometry Sim", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
* { font-family: 'Roboto', system-ui, sans-serif; }
.stApp, [data-testid="stAppViewContainer"], section[data-testid="stMain"] {
    background: #f0e6ce !important;
}
[data-testid="stMainBlockContainer"] { background: #f0e6ce !important; }
[data-testid="stSidebar"] { background: #e6d8bb !important; }
h2, h3 { font-size: 0.8rem !important; font-weight: 700 !important;
          letter-spacing: 1.5px; text-transform: uppercase; color: #5c4a1e !important; }
.block-container { padding-top: 1.2rem !important; padding-left: 2rem !important;
                   padding-right: 2rem !important; }
[data-testid="stSlider"] label { font-size: 0.75rem !important; font-weight: 700;
                                  letter-spacing: 0.8px; text-transform: uppercase;
                                  color: #111 !important; }
[data-testid="stSidebar"] * { color: #111; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="border-bottom:3px solid #8b6914;padding-bottom:14px;margin-bottom:2px;">
  <div style="font-size:0.65rem;font-weight:700;letter-spacing:2.5px;color:#a08040;
              text-transform:uppercase;margin-bottom:6px;">NBA Analytics</div>
  <div style="font-size:1.9rem;font-weight:900;color:#1a1209;line-height:1.1;
              letter-spacing:-0.5px;">The Geometry Simulator</div>
  <div style="font-size:0.83rem;color:#5c4a1e;margin-top:8px;font-weight:400;">
    How a longer 3-point line reshapes NBA shot quality — zone by zone
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
line_dist = st.sidebar.slider("3-Point Line Distance (ft)", 22.0, 32.0, 23.75, step=0.25)

corn_display = round(line_dist - 1.75, 2)
st.sidebar.markdown(f"""
<div style="text-align:center;padding:10px 0 4px 0;">
  <div style="font-size:2.4rem;font-weight:900;color:#111;line-height:1">{line_dist}</div>
  <div style="font-size:0.65rem;font-weight:700;letter-spacing:1.5px;color:#333;
              text-transform:uppercase;">feet from basket</div>
  <div style="font-size:0.75rem;color:#333;margin-top:4px;">
    Corner: {corn_display} ft
  </div>
</div>""", unsafe_allow_html=True)
if line_dist > 26.75:
    st.sidebar.warning("Corner 3 zone eliminated at this distance.")

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_parquet('league_shots.parquet')

@st.cache_data
def compute_baseline(_df):
    df = _df.copy()
    df['SIM_ZONE'] = df.apply(simulate_new_zones, args=(23.75,), axis=1)
    df['SIM_PTS']  = df['SIM_ZONE'].apply(lambda x: 3 if '3 (Sim)' in x else 2) * df['SHOT_MADE_FLAG']
    return df.groupby('SIM_ZONE')['SIM_PTS'].mean()

@st.cache_data
def compute_zone_data(_df, dist):
    """Returns processed df, zone_stats dict, shot_pct dict — cached by distance."""
    df = _df.copy()
    df['SIM_ZONE'] = df.apply(simulate_new_zones, args=(dist,), axis=1)
    df['SIM_PTS']  = df['SIM_ZONE'].apply(lambda x: 3 if '3 (Sim)' in x else 2) * df['SHOT_MADE_FLAG']
    zone_stats = df.groupby('SIM_ZONE')['SIM_PTS'].mean().to_dict()
    counts     = df.groupby('SIM_ZONE').size()
    shot_pct   = (counts / counts.sum() * 100).to_dict()
    return df, zone_stats, shot_pct

@st.cache_data
def render_hexbin_png(_df, line_dist):
    """Renders hexbin chart and returns PNG bytes — cached by distance."""
    fig = draw_hexbin_chart(line_dist, _df)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Ensure 'league_shots.parquet' is in this folder.")
    st.stop()

baseline_stats = compute_baseline(df)

# ── Processing (cached by distance) ──────────────────────────────────────────
df_processed, zone_stats, shot_pct = compute_zone_data(df, line_dist)

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2.5], gap='large')

with col1:
    st.subheader("Efficiency by Zone")

    # Sort zones by PPS descending; zones with no data sink to the bottom
    ranked = sorted(
        [z for z in ZONE_ORDER if zone_stats.get(z) is not None],
        key=lambda z: zone_stats[z],
        reverse=True,
    )
    missing = [z for z in ZONE_ORDER if zone_stats.get(z) is None]

    for rank_idx, zone in enumerate(ranked + missing, start=1):
        display = ZONE_DISPLAY[zone]
        pps     = zone_stats.get(zone)
        base    = baseline_stats.get(zone)

        if pps is None:
            st.markdown(f"""
<div style="background:#fdf6e8;border-radius:6px;margin-bottom:8px;padding:10px 14px;
            border-left:4px solid #c8b890;display:flex;align-items:center;gap:12px;">
  <span style="font-size:1.1rem;font-weight:700;color:#bfab8a;min-width:24px">—</span>
  <div style="flex:1">
    <div style="font-size:0.65rem;font-weight:700;letter-spacing:1px;color:#9a8860;text-transform:uppercase">{display}</div>
    <div style="font-size:1.3rem;font-weight:700;color:#bfab8a;line-height:1.1">N/A <span style="font-size:0.75rem;font-weight:400;color:#9a8860">pts/attempt</span></div>
    <div style="font-size:0.72rem;color:#9a8860">Zone not active at this distance</div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            color = pps_to_hsl(pps)
            fixed_zone = zone in {'Paint', 'Short Mid-Range'}
            delta = (pps - base if base is not None else None) if not fixed_zone else None
            if delta is not None:
                delta_str   = f"{delta:+.3f} vs baseline"
                delta_color = '#2a9d2a' if delta >= 0 else '#c0392b'
            elif fixed_zone:
                delta_str   = ''
                delta_color = '#888'
            else:
                delta_str   = '—'
                delta_color = '#888'
            st.markdown(f"""
<div style="background:#fdf6e8;border-radius:6px;margin-bottom:8px;padding:10px 14px;
            border-left:4px solid {color};display:flex;align-items:center;gap:12px;">
  <span style="font-size:1.1rem;font-weight:700;color:#7a6240;min-width:24px">#{rank_idx}</span>
  <div style="flex:1">
    <div style="font-size:0.65rem;font-weight:700;letter-spacing:1px;color:#5c4a1e;text-transform:uppercase">{display}</div>
    <div style="font-size:1.3rem;font-weight:700;color:{color};line-height:1.1">{pps:.3f} <span style="font-size:0.75rem;font-weight:400;color:#7a6240">pts/attempt</span></div>
    <div style="font-size:0.72rem;color:{delta_color}">{delta_str}</div>
  </div>
</div>""", unsafe_allow_html=True)

with col2:
    st.subheader("Spatial Efficiency Map")
    png = render_hexbin_png(df_processed, line_dist)
    st.image(png, use_container_width=True)
