# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:57:44 2025

@author: libra
"""




#import ---------------------------------------------------------------------
from djitellopy import tello
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import threading
import cv2
import time 

#Variables globales ---------------------------------------------------------
global img

# Liste des commandes SDK ---------------------------------------------------

COMMANDS = [
        "takeoff", "land", "streamon", "streamoff", "emergency",
        "up x", "down x", "left x", "right x", "forward x", "back x",
        "cw x", "ccw x", "flip l/r/f/b",
        "go x y z speed", "curve x1 y1 z1 x2 y2 z2 speed", "stop",
        "speed x", "rc a b c d", "wifi ssid pass",
        "mon", "moff", "mdirection x",
        "speed?", "battery?", "time?", "wifi?", "sdk?", "sn?",
        "height?", "tof?", "baro?", "attitude?"
    ]


# Paramètres ----------------------------------------------------------------
data = "data_tello.csv"
outdir = Path("tello_out")
SHOW_PLOTS = True


#Connection au drone----------------------------------------------------------
Drone = tello.Tello()
Drone.connect()

#Commande du drone------------------------------------------------------------

    
def vol():
    time.sleep(2)  # laisser le temps au thread télémétrie de démarrer

    Drone.takeoff()
    time.sleep(5)
    # Drone.rotate_clockwise(360)
    # Drone.flip_back()
    # time.sleep(2)
    # Drone.flip_forward()
    # time.sleep(2)
    # Drone.flip_left()
    # time.sleep(2)
    # Drone.flip_right()
    # time.sleep(2)
    Drone.land()

    print("Séquence de vol terminée.")

#Récupération des infos de télémétrie et du flux vidéeo -----------------------

def listen_state():
    """ lance l'écoute des données de télémétrie du drone et les enregistre dans un CSV """
      
    
    # ouverture du fichier CSV en mode écriture
    with open(data, mode='w', newline='') as f:
        writer = None
    
        while True:
            #text = data.decode('ascii', errors='ignore').strip()

            raw_state = Drone.get_current_state()
            if not raw_state:
                continue

            parsed_state = {}
            for part in raw_state.split(';'):
                if ':' in part:
                    k, v = part.split(':', 1)
                    parsed_state[k] = v

            # Initialiser le writer avec les clés en première ligne
            if writer is None:
                fieldnames = ["timestamp"] + list(parsed_state.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            # Écrire une nouvelle ligne avec timestamp
            row = {"timestamp": time.time()}
            row.update(parsed_state)
            writer.writerow(row)
            f.flush()

    print("Acquisation des données de télémétrie du drone" )
 

    # Gestion vidéo 

def video_stream():
    Drone.streamon()
    while True:
        img = Drone.get_frame_read().frame
        img = cv2.resize(img, (1080, 720))
        cv2.imshow("DroneCapture", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
    
# Analyse post-vol des données de télémétrie ##################################

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
        return pd.to_numeric(df['timestamp'], errors='coerce')
    if 'time' in df.columns:
        return pd.to_numeric(df['time'], errors='coerce')
    return pd.Series(np.arange(len(df)) * 0.05, index=df.index, name='t')

def load_and_parse(input_csv: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(input_csv)

    # Harmonisation des noms
    rename_map = {
        'vx':'vgx', 'vy':'vgy', 'vz':'vgz',
        'batt':'bat','battery':'bat',
        'temp':'templ','temperature':'templ'
    }
    df_raw = df_raw.rename(columns=rename_map)

    if 'state' in df_raw.columns:
        parsed = pd.DataFrame([parse_state_string(s) for s in df_raw['state']])
        extras = [c for c in df_raw.columns if c not in parsed.columns]
        df = pd.concat([df_raw[extras].reset_index(drop=True),
                        parsed.reset_index(drop=True)], axis=1)
    else:
        df = df_raw.copy()

    for c in SDK_FIELDS:
        if c not in df.columns:
            df[c] = np.nan

    df['t_s'] = pd.to_numeric(best_time_series(df), errors='coerce')
    vgx = pd.to_numeric(df.get('vgx', np.nan), errors='coerce')
    vgy = pd.to_numeric(df.get('vgy', np.nan), errors='coerce')
    df['vxy'] = np.sqrt(np.square(vgx) + np.square(vgy))

    return df

def battery_stats(df: pd.DataFrame) -> dict:
    bat = pd.to_numeric(df.get('bat', np.nan), errors='coerce').dropna()
    if bat.empty:
        return {"start": np.nan, "end": np.nan, "drop": np.nan}
    return {"start": float(bat.iloc[0]), "end": float(bat.iloc[-1]), "drop": float(bat.iloc[0]-bat.iloc[-1])}

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_numeric(df['t_s'], errors='coerce')
    vxy = pd.to_numeric(df['vxy'], errors='coerce')
    h = pd.to_numeric(df.get('h', np.nan), errors='coerce')

    duration_s = float(t.max() - t.min()) if t.notna().any() else np.nan
    vxy_max_m_s = float((vxy.max() if pd.notna(vxy.max()) else np.nan) / 100.0)
    h_max_m = float((h.max() if pd.notna(h.max()) else np.nan) / 100.0)

    yaw = pd.to_numeric(df.get('yaw', np.nan), errors='coerce')
    yaw_span = float(yaw.max() - yaw.min()) if yaw.notna().any() else np.nan

    bat = battery_stats(df)

    return pd.DataFrame([{
        "samples": int(len(df)),
        "duration_s": duration_s,
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

    if 'h' in df.columns and df['h'].notna().any():
        saveplot('h', 'Hauteur [cm]')
    elif 'tof' in df.columns and df['tof'].notna().any():
        saveplot('tof', 'ToF [cm]')

    saveplot('vxy', 'Vitesse horizontale [cm/s]')
    saveplot('yaw', 'Yaw [°]')
    saveplot('bat', 'Batterie [%]')


    
    
    
# MAIN ########################################################################



if __name__ == "__main__":
    # Thread acquisition
    t = threading.Thread(target=listen_state, daemon=True)
    t.start()
    print('thread acquisition des données télémétrique')

    #Thread pour le vol
    vol_thread = threading.Thread(target=vol, daemon=True)
    vol_thread.start()
    print('thread vol')

    #Thread pour la capture du flux vidéo
    video_thread = threading.Thread(target=video_stream, daemon=True)
    video_thread.start()
    print('Thread vidéo lancé')



 # Boucle principale (permet d’interrompre à la main)
    try:
        while vol_thread.is_alive():  # attendre que le vol soit terminé
            time.sleep(1)
    except KeyboardInterrupt:
        print("Arrêt manuel")
        Drone.end()
        cv2.destroyAllWindows()

    # ---------- Analyse après vol ----------
    print("\n--- Analyse post-vol ---\n")
    outdir.mkdir(exist_ok=True)
    df = load_and_parse(Path(data))
    summary = compute_summary(df)

    df.to_csv(outdir / "cleaned_parsed.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)

    print("Résumé du vol :")
    print(summary.to_string(index=False))

    if SHOW_PLOTS:
        plot_quick(df, outdir)
        if 'h' in df.columns and df['h'].notna().any():
            plt.figure(); plt.plot(df['t_s'], df['h'])
            plt.xlabel("Temps [s]"); plt.ylabel("Hauteur [cm]"); plt.title("Évolution de la hauteur"); plt.show()
        elif 'tof' in df.columns and df['tof'].notna().any():
            plt.figure(); plt.plot(df['t_s'], df['tof'])
            plt.xlabel("Temps [s]"); plt.ylabel("ToF [cm]"); plt.title("Évolution ToF = distance au sol"); plt.show()
            


    