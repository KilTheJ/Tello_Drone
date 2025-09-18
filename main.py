#!/usr/bin/env python3
# Analyse post-vol Tello depuis CSV (version Spyder, avec mapping datatest.csv)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Paramètres ----------
chemin_csv = "datatest.csv"     # ← ton CSV
outdir = Path("tello_out")
SHOW_PLOTS = True
# -------------------------------

SDK_FIELDS = [
    "pitch","roll","yaw","vgx","vgy","vgz","templ","temph","tof","h",
    "bat","baro","time","agx","agy","agz","mid","x","y","z"
]

def parse_state_string(s: str) -> dict:
    out = {}
    if not isinstance(s, str):
        return out
    for kv in s.strip().split(';'):
        if not kv or ':' not in kv:
            continue
        k, v = kv.split(':', 1)
        k = k.strip(); v = v.strip()
        try:
            if k in {"baro","agx","agy","agz"}:
                out[k] = float(v)
            else:
                out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        except Exception:
            out[k] = np.nan
    return out

def best_time_series(df: pd.DataFrame) -> pd.Series:
    if 'timestamp' in df.columns:
        return pd.to_numeric(df['timestamp'], errors='coerce')            # s
    if 'timestamp_ms' in df.columns:
        return pd.to_numeric(df['timestamp_ms'], errors='coerce') / 1000  # ms→s
    if 'time' in df.columns:
        return pd.to_numeric(df['time'], errors='coerce')                 # compteur SDK
    n = len(df)
    dt = 0.05
    if 'dt' in df.columns:
        med = pd.to_numeric(df['dt'], errors='coerce').dropna().median()
        if pd.notna(med) and med > 0:
            dt = float(med)
    return pd.Series(np.arange(n) * dt, index=df.index, name='t')

def load_and_parse(input_csv: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(input_csv)

    # --- Harmonisation des noms (cas "datatest.csv") ---
    rename_map = {
        'vx':'vgx', 'vy':'vgy', 'vz':'vgz',
        'batt':'bat',
        'battery':'bat',
        'temp':'templ',
        'temperature':'templ'
    }
    df_raw = df_raw.rename(columns=rename_map)

    # --- CSV type "state" (brut SDK) ou colonnes déjà séparées ---
    if 'state' in df_raw.columns:
        parsed = pd.DataFrame([parse_state_string(s) for s in df_raw['state']])
        extras = [c for c in df_raw.columns if c not in parsed.columns]
        df = pd.concat([df_raw[extras].reset_index(drop=True),
                        parsed.reset_index(drop=True)], axis=1)
    else:
        df = df_raw.copy()

    # Colonnes SDK manquantes → NaN
    for c in SDK_FIELDS:
        if c not in df.columns:
            df[c] = np.nan

    # Typage
    float_cols = {'baro','agx','agy','agz'}
    intlike_cols = {'pitch','roll','yaw','vgx','vgy','vgz','templ','temph','tof','h','bat','time','mid','x','y','z'}
    for c in df.columns:
        if c in float_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        if c in intlike_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Temps
    df['t_s'] = pd.to_numeric(best_time_series(df), errors='coerce')

    # Vitesse horizontale
    vgx = pd.to_numeric(df.get('vgx', np.nan), errors='coerce')
    vgy = pd.to_numeric(df.get('vgy', np.nan), errors='coerce')
    df['vxy'] = np.sqrt(np.square(vgx) + np.square(vgy))  # cm/s

    return df

def integrate_distance(t_s: pd.Series, v_series: pd.Series) -> float:
    t = pd.to_numeric(t_s, errors='coerce').to_numpy()
    v = pd.to_numeric(v_series, errors='coerce').to_numpy()
    mask = ~np.isnan(t) & ~np.isnan(v)
    if mask.sum() < 2:
        return float('nan')
    return float(np.trapz(v[mask] / 100.0, t[mask]))  # cm/s → m

def battery_stats(df: pd.DataFrame) -> dict:
    bat = pd.to_numeric(df.get('bat', np.nan), errors='coerce').dropna()
    if bat.empty:
        return {"start": np.nan, "end": np.nan, "drop": np.nan}
    return {"start": float(bat.iloc[0]), "end": float(bat.iloc[-1]), "drop": float(bat.iloc[0]-bat.iloc[-1])}

def anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for ax in ['vgx','vgy','vgz']:
        out[f'anom_{ax}'] = pd.to_numeric(df.get(ax, np.nan), errors='coerce').abs() > 1000  # >10 m/s
    h = pd.to_numeric(df.get('h', np.nan), errors='coerce')
    out['anom_h'] = (h < 0) | (h > 300)
    bat = pd.to_numeric(df.get('bat', np.nan), errors='coerce')
    out['anom_bat'] = (bat < 0) | (bat > 100)
    out['anomaly_any'] = out.any(axis=1)
    return out

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_numeric(df['t_s'], errors='coerce')
    vxy = pd.to_numeric(df['vxy'], errors='coerce')
    h = pd.to_numeric(df.get('h', np.nan), errors='coerce')

    duration_s = float(t.max() - t.min()) if t.notna().any() else np.nan
    dist_m = integrate_distance(t, vxy)
    vxy_max_m_s = float((vxy.max() if pd.notna(vxy.max()) else np.nan) / 100.0)
    h_max_m = float((h.max() if pd.notna(h.max()) else np.nan) / 100.0)

    yaw = pd.to_numeric(df.get('yaw', np.nan), errors='coerce')
    yaw_span = float(yaw.max() - yaw.min()) if yaw.notna().any() else np.nan

    bat = battery_stats(df)

    return pd.DataFrame([{
        "samples": int(len(df)),
        "duration_s": duration_s,
        "distance_horizontal_m": dist_m,
        "vxy_max_m_s": vxy_max_m_s,
        "h_max_m": h_max_m,
        "yaw_span_deg": yaw_span,
        "battery_start_pct": bat["start"],
        "battery_end_pct": bat["end"],
        "battery_drop_pct": bat["drop"]
    }])

def plot_quick(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    t = pd.to_numeric(df['t_s'], errors='coerce')

    def saveplot(ycol, ylabel):
        y = pd.to_numeric(df.get(ycol, np.nan), errors='coerce')
        mask = t.notna() & y.notna()
        if mask.sum() < 2:
            return
        plt.figure()
        plt.plot(t[mask], y[mask])
        plt.xlabel("Temps [s]"); plt.ylabel(ylabel); plt.title(ylabel)
        plt.savefig(outdir / f"{ycol}.png", bbox_inches='tight', dpi=150)
        plt.close()

    # Hauteur si dispo, sinon ToF si dispo
    if 'h' in df.columns and df['h'].notna().any():
        saveplot('h', 'Hauteur [cm]')
    elif 'tof' in df.columns and df['tof'].notna().any():
        saveplot('tof', 'ToF')

    saveplot('vxy', 'Vitesse horizontale [cm/s]')
    saveplot('yaw', 'Yaw [deg]')
    saveplot('bat', 'Batterie [%]')

# ---------- Exécution ----------
outdir.mkdir(exist_ok=True)
df = load_and_parse(Path(chemin_csv))
df = pd.concat([df, anomaly_flags(df)], axis=1)
summary = compute_summary(df)

df.to_csv(outdir / "cleaned_parsed.csv", index=False)
summary.to_csv(outdir / "summary.csv", index=False)

print("Résumé du vol :")
print(summary.to_string(index=False))

if SHOW_PLOTS:
    plot_quick(df, outdir)
    # Affichage interactif rapide
    if 'h' in df.columns and df['h'].notna().any():
        plt.figure(); plt.plot(df['t_s'], df['h'])
        plt.xlabel("Temps [s]"); plt.ylabel("Hauteur [cm]"); plt.title("Évolution de la hauteur"); plt.show()
    elif 'tof' in df.columns and df['tof'].notna().any():
        plt.figure(); plt.plot(df['t_s'], df['tof'])
        plt.xlabel("Temps [s]"); plt.ylabel("ToF"); plt.title("Évolution ToF = distance au sol"); plt.show()
