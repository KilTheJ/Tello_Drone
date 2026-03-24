# -*- coding: utf-8 -*-
"""
Programme Tello complet évolué :
- pilotage du drone
- acquisition télémétrie
- acquisition vidéo
- découpe vidéo en frames
- export matrices
- analyse post-vol
- reconstruction de trajectoire 3D à partir de l'odométrie et des mesures IMU/ToF
- fusion simple de hauteur h / tof
- fusion verticale odométrie + IMU + hauteur
"""

##############################################################################
# ---------- IMPORTS ----------
##############################################################################

from pathlib import Path
import csv
import time
import threading

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from djitellopy import tello


##############################################################################
# ---------- PARAMETRES GENERAUX ----------
##############################################################################

# Dossiers / fichiers de sortie
OUTDIR = Path("tello_out")
VIDEO_DIR = OUTDIR / "video"
CSV_FILE = OUTDIR / "data_tello.csv"
VIDEO_FILE = VIDEO_DIR / "output.mp4"
MATRICES_FILE = VIDEO_DIR / "matrices.txt"

# Fichiers de sortie pour la reconstruction 3D
TRAJ3D_CSV_FILE = OUTDIR / "trajectory_3d.csv"
TRAJ3D_PNG_FILE = OUTDIR / "trajectory_3d.png"
TRAJ3D_TIME_PNG_FILE = OUTDIR / "trajectory_3d_colored_time.png"
TRAJ2D_PNG_FILE = OUTDIR / "trajectory_2d_projection.png"

# Affichage des graphiques post-vol
SHOW_PLOTS = True

# Paramètres vidéo
FRAME_WIDTH = 1080
FRAME_HEIGHT = 720
VIDEO_FPS = 30.0

# Pas de temps pour la télémétrie (en secondes)
TELEMETRY_PERIOD = 0.10

# Nombre maximum de frames converties en matrices texte
MAX_MATRIX_FRAMES = 15

# Paramètres temporels
MIN_VALID_DT = 1e-3
MAX_VALID_DT = 0.5

# Paramètres de lissage
VELOCITY_SMOOTH_ALPHA = 0.25
IMU_SMOOTH_ALPHA = 0.20

# Paramètres de reconstruction XY
USE_YAW_FOR_XY = False
# False : recommandé si tu fais juste des translations sans rotation explicite
# True  : utile si tu fais des cw/ccw et veux intégrer le yaw au repère monde

# Paramètres de fusion verticale
HEIGHT_FUSION_ALPHA = 0.40
Z_MEAS_WEIGHT = 0.40
VZ_IMU_WEIGHT = 0.10

# Paramètres de seuils
MIN_FLIGHT_HEIGHT_CM = 8.0
FLIGHT_HEIGHT_THRESHOLD_CM = 12.0
FLIGHT_SPEED_THRESHOLD_CM_S = 10.0
GROUND_SPEED_ZERO_CLAMP_CM_S = 8.0

# Saturations capteurs / sécurité numérique
MAX_ABS_VXY_CM_S = 120.0
MAX_ABS_VZ_CM_S = 120.0
MAX_ABS_AZ_M_S2 = 4.0

# IMU
IMU_ACCEL_SCALE = 1.0
IMU_BIAS_MIN_SAMPLES = 8

# Liste des commandes affichées à l'utilisateur
COMMANDS = [
    "takeoff", "land", "battery?", "height?", "speed?", "time?", "emergency",
    "up x", "down x", "left x", "right x", "forward x", "back x",
    "cw x", "ccw x",
    "video_on", "video_off", "cutvideo",
    "exit"
]


##############################################################################
# ---------- VARIABLES GLOBALES PARTAGEES ----------
##############################################################################

Drone = None

# Gestion des threads
stop_telemetry_event = threading.Event()
stop_video_event = threading.Event()

telemetry_thread = None
video_thread = None

# Protection des commandes envoyées au drone
drone_lock = threading.Lock()

# Etat vidéo
video_running = False


##############################################################################
# ---------- OUTILS DE BASE ----------
##############################################################################

def ensure_directories():
    """Crée les dossiers de sortie si besoin."""
    OUTDIR.mkdir(exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def safe_streamoff():
    """Coupe le stream vidéo proprement si actif."""
    global Drone
    if Drone is None:
        return

    try:
        with drone_lock:
            if getattr(Drone, "stream_on", False):
                Drone.streamoff()
    except Exception as e:
        print(f"[STREAMOFF] Impossible de couper le flux : {e}")


def exp_smooth(series: pd.Series, alpha: float) -> pd.Series:
    """
    Lissage exponentiel simple.
    alpha proche de 0 => plus lissé
    alpha proche de 1 => plus réactif
    """
    s = pd.to_numeric(series, errors="coerce").copy()
    if s.dropna().empty:
        return s

    out = np.full(len(s), np.nan, dtype=float)
    valid_idx = np.where(~np.isnan(s.to_numpy(dtype=float)))[0]
    if len(valid_idx) == 0:
        return pd.Series(out, index=s.index)

    first = valid_idx[0]
    out[first] = s.iloc[first]

    for i in range(first + 1, len(s)):
        x = s.iloc[i]
        if np.isnan(x):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * x + (1 - alpha) * out[i - 1]

    return pd.Series(out, index=s.index)


def wrap_angle_deg_180(series_deg: pd.Series) -> pd.Series:
    """
    Replie un angle dans [-180, 180].
    """
    a = pd.to_numeric(series_deg, errors="coerce").to_numpy(dtype=float)
    a = (a + 180.0) % 360.0 - 180.0
    return pd.Series(a, index=series_deg.index)


def unwrap_angle_deg(series_deg: pd.Series) -> pd.Series:
    """
    Déroule un angle pour éviter les sauts ±180/±360 lors de l'intégration.
    """
    a = pd.to_numeric(series_deg, errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(a)):
        return pd.Series(a, index=series_deg.index)

    valid = ~np.isnan(a)
    out = a.copy()
    out[valid] = np.rad2deg(np.unwrap(np.deg2rad(a[valid])))
    return pd.Series(out, index=series_deg.index)


def body_accel_to_world_z(ax, ay, az, roll_deg, pitch_deg):
    """
    Projette l'accélération du repère drone vers l'axe vertical monde.
    Le yaw n'influe pas sur l'axe Z monde.
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)

    az_world = (
        -np.sin(pitch) * ax
        + np.cos(pitch) * np.sin(roll) * ay
        + np.cos(pitch) * np.cos(roll) * az
    )
    return az_world


def estimate_bias_from_stationary(series: pd.Series, stationary_mask: pd.Series, fallback_n=15) -> float:
    """
    Estime un biais IMU à partir des échantillons stationnaires.
    Si pas assez d'échantillons stationnaires, on prend les premiers samples.
    """
    s = pd.to_numeric(series, errors="coerce")

    stationary_values = s[stationary_mask & s.notna()]
    if len(stationary_values) >= IMU_BIAS_MIN_SAMPLES:
        return float(stationary_values.iloc[:fallback_n].mean())

    fallback = s.iloc[:fallback_n].dropna()
    if len(fallback) > 0:
        return float(fallback.mean())

    return 0.0


def body_velocity_to_world_xy(vx_body_m_s, vy_body_m_s, yaw_deg, use_yaw=True):
    """
    Convertit les vitesses horizontales du repère drone vers le repère monde.
    Si use_yaw=False, on garde directement les vitesses corps.
    """
    vx_body_m_s = np.asarray(vx_body_m_s, dtype=float)
    vy_body_m_s = np.asarray(vy_body_m_s, dtype=float)

    if not use_yaw:
        return vx_body_m_s, vy_body_m_s

    yaw_rad = np.deg2rad(np.asarray(yaw_deg, dtype=float))

    vx_world = np.cos(yaw_rad) * vx_body_m_s - np.sin(yaw_rad) * vy_body_m_s
    vy_world = np.sin(yaw_rad) * vx_body_m_s + np.cos(yaw_rad) * vy_body_m_s

    return vx_world, vy_world


##############################################################################
# ---------- CONNEXION AU DRONE ----------
##############################################################################

def connect_drone():
    """Connexion au drone avec djitellopy."""
    global Drone

    print("\n--- Programme en cours : CONNEXION AU DRONE ---\n")

    Drone = tello.Tello()

    with drone_lock:
        Drone.connect()
        battery = Drone.get_battery()

    print(f"Connexion OK au drone | Batterie : {battery}%")
    return Drone


##############################################################################
# ---------- ACQUISITION DE LA TELEMETRIE ----------
##############################################################################

def listen_state():
    """
    Interroge régulièrement le drone,
    récupère les données de télémétrie,
    puis les enregistre dans un CSV.
    """
    print("Démarrage du thread de télémétrie...")

    writer = None
    csv_file = None

    try:
        csv_file = open(CSV_FILE, mode="w", newline="", encoding="utf-8")

        while not stop_telemetry_event.is_set():
            try:
                state = Drone.get_current_state()

                if not state:
                    time.sleep(TELEMETRY_PERIOD)
                    continue

                if writer is None:
                    fieldnames = ["timestamp"] + list(state.keys())
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()

                row = {"timestamp": time.time()}
                row.update(state)
                writer.writerow(row)
                csv_file.flush()

            except Exception as e:
                print(f"[TELEMETRIE] Erreur de lecture : {e}")

            time.sleep(TELEMETRY_PERIOD)

    finally:
        if csv_file is not None:
            csv_file.close()

    print("Arrêt du thread de télémétrie.")


##############################################################################
# ---------- CAPTURE DE LA VIDEO ----------
##############################################################################

def getVideo():
    """
    Lance le flux vidéo du drone,
    enregistre la vidéo dans un fichier,
    et affiche les frames en temps réel.
    """
    global video_running

    print("Démarrage du thread vidéo...")

    out = None
    frame_reader = None

    try:
        with drone_lock:
            if not getattr(Drone, "stream_on", False):
                Drone.streamon()

            frame_reader = Drone.get_frame_read()

        time.sleep(1.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(VIDEO_FILE),
            fourcc,
            VIDEO_FPS,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        if not out.isOpened():
            raise RuntimeError("Impossible d'ouvrir le fichier vidéo pour écriture.")

        video_running = True

        while not stop_video_event.is_set():
            frame = frame_reader.frame

            if frame is None:
                time.sleep(0.01)
                continue

            frame = frame.copy()
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            out.write(frame)

            cv2.imshow("Tello Video", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Arrêt vidéo demandé via la touche q")
                stop_video_event.set()
                break

    except Exception as e:
        print(f"[VIDEO] Erreur : {e}")

    finally:
        video_running = False

        if out is not None:
            out.release()

        cv2.destroyAllWindows()
        safe_streamoff()

        print("Arrêt du thread vidéo.")


def start_video_thread():
    """Lance le thread vidéo s'il n'est pas déjà actif."""
    global video_thread

    if video_thread is not None and video_thread.is_alive():
        print("La vidéo tourne déjà.")
        return

    stop_video_event.clear()
    video_thread = threading.Thread(target=getVideo, daemon=True)
    video_thread.start()


def stop_video_thread():
    """Demande l'arrêt propre du thread vidéo."""
    global video_thread

    if video_thread is None or not video_thread.is_alive():
        print("La vidéo n'est pas en cours.")
        return

    stop_video_event.set()
    video_thread.join(timeout=8)

    if video_thread.is_alive():
        print("Le thread vidéo ne s'est pas arrêté complètement dans le temps imparti.")
    else:
        print("Thread vidéo stoppé.")


def cutVideo():
    """
    Découpe la vidéo enregistrée en frames PNG,
    puis convertit les premières images en matrices texte.
    """
    print("\n--- Programme en cours : DECOUPAGE DE LA VIDEO ---\n")

    if video_running or (video_thread is not None and video_thread.is_alive()):
        print("Impossible de découper : la vidéo est encore en cours d'enregistrement.")
        print("Fais d'abord 'video_off', puis relance 'cutvideo'.")
        return

    if not VIDEO_FILE.exists() or VIDEO_FILE.stat().st_size == 0:
        print("Aucune vidéo disponible à découper.")
        return

    vidcap = cv2.VideoCapture(str(VIDEO_FILE))

    if not vidcap.isOpened():
        print("Impossible d'ouvrir le fichier vidéo.")
        return

    success, image = vidcap.read()
    count = 0

    while success:
        frame_path = VIDEO_DIR / f"frame{count}.png"
        cv2.imwrite(str(frame_path), image)

        success, image = vidcap.read()
        count += 1
        print(f"Frame : {count} | lecture suivante OK = {success}")

    vidcap.release()

    with open(MATRICES_FILE, "w", encoding="utf-8") as f:
        for i in range(min(MAX_MATRIX_FRAMES, count)):
            img_path = VIDEO_DIR / f"frame{i}.png"
            img = cv2.imread(str(img_path), 0)

            if img is not None:
                f.write(f"\n--- MATRICE FRAME {i} ---\n")
                np.savetxt(f, img, fmt="%d")

    print(f"Découpage terminé | {count} frames extraites.")
    print(f"Matrices enregistrées dans : {MATRICES_FILE}")


##############################################################################
# ---------- ANALYSE POST-VOL ----------
##############################################################################

SDK_FIELDS = [
    "pitch", "roll", "yaw", "vgx", "vgy", "vgz", "templ", "temph",
    "tof", "h", "bat", "baro", "time", "agx", "agy", "agz",
    "mid", "x", "y", "z"
]


def parse_state_string(s: str) -> dict:
    out = {}

    if not isinstance(s, str):
        return out

    for kv in s.strip().split(";"):
        if not kv or ":" not in kv:
            continue

        k, v = kv.split(":", 1)
        k = k.strip()
        v = v.strip()

        try:
            if k in {"baro", "agx", "agy", "agz"}:
                out[k] = float(v)
            else:
                out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        except Exception:
            out[k] = np.nan

    return out


def best_time_series(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return pd.to_numeric(df["timestamp"], errors="coerce")
    if "time" in df.columns:
        return pd.to_numeric(df["time"], errors="coerce")
    return pd.Series(np.arange(len(df)) * TELEMETRY_PERIOD, index=df.index, name="t")


def load_and_parse(input_csv: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(input_csv)

    rename_map = {
        "vx": "vgx",
        "vy": "vgy",
        "vz": "vgz",
        "batt": "bat",
        "battery": "bat",
        "temp": "templ",
        "temperature": "templ",
        "height": "h"
    }
    df_raw = df_raw.rename(columns=rename_map)

    if "state" in df_raw.columns:
        parsed = pd.DataFrame([parse_state_string(s) for s in df_raw["state"]])
        extras = [c for c in df_raw.columns if c not in parsed.columns]
        df = pd.concat(
            [df_raw[extras].reset_index(drop=True), parsed.reset_index(drop=True)],
            axis=1
        )
    else:
        df = df_raw.copy()

    for c in SDK_FIELDS:
        if c not in df.columns:
            df[c] = np.nan

    # Temps relatif [s]
    t_raw = pd.to_numeric(best_time_series(df), errors="coerce")
    if t_raw.notna().sum() >= 2:
        df["t_s"] = t_raw - t_raw.iloc[0]
    else:
        df["t_s"] = np.arange(len(df)) * TELEMETRY_PERIOD

    # Colonnes numériques utiles
    for c in ["pitch", "roll", "yaw", "vgx", "vgy", "vgz", "h", "tof", "bat", "agx", "agy", "agz"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")

    df["vxy"] = np.sqrt(np.square(df["vgx"]) + np.square(df["vgy"]))

    return df


def battery_stats(df: pd.DataFrame) -> dict:
    bat = pd.to_numeric(df.get("bat", np.nan), errors="coerce").dropna()

    if bat.empty:
        return {"start": np.nan, "end": np.nan, "drop": np.nan}

    return {
        "start": float(bat.iloc[0]),
        "end": float(bat.iloc[-1]),
        "drop": float(bat.iloc[0] - bat.iloc[-1])
    }


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_numeric(df["t_s"], errors="coerce")
    vxy = pd.to_numeric(df.get("vxy", np.nan), errors="coerce")
    h = pd.to_numeric(df.get("h", np.nan), errors="coerce")
    tof = pd.to_numeric(df.get("tof", np.nan), errors="coerce")
    yaw = pd.to_numeric(df.get("yaw", np.nan), errors="coerce")
    z_m = pd.to_numeric(df.get("z_m", np.nan), errors="coerce")

    duration_s = float(t.max() - t.min()) if t.notna().any() else np.nan
    vxy_max_m_s = float((vxy.max() if pd.notna(vxy.max()) else np.nan) / 100.0)
    h_max_m = float((h.max() if pd.notna(h.max()) else np.nan) / 100.0)
    tof_max_m = float((tof.max() if pd.notna(tof.max()) else np.nan) / 100.0)
    yaw_span = float(yaw.max() - yaw.min()) if yaw.notna().any() else np.nan
    z_max_m = float(z_m.max()) if z_m.notna().any() else np.nan

    bat = battery_stats(df)

    traveled_2d_m = np.nan
    if "step_dist_2d_m" in df.columns:
        step_dist_2d = pd.to_numeric(df["step_dist_2d_m"], errors="coerce")
        if step_dist_2d.notna().any():
            traveled_2d_m = float(step_dist_2d.fillna(0).sum())

    traveled_3d_m = np.nan
    if "step_dist_3d_m" in df.columns:
        step_dist_3d = pd.to_numeric(df["step_dist_3d_m"], errors="coerce")
        if step_dist_3d.notna().any():
            traveled_3d_m = float(step_dist_3d.fillna(0).sum())

    return pd.DataFrame([{
        "samples": int(len(df)),
        "duration_s": duration_s,
        "vxy_max_m_s": vxy_max_m_s,
        "h_max_m": h_max_m,
        "tof_max_m": tof_max_m,
        "z_max_m": z_max_m,
        "yaw_span_deg": yaw_span,
        "traveled_2d_m": traveled_2d_m,
        "traveled_3d_m": traveled_3d_m,
        "battery_start_pct": bat["start"],
        "battery_end_pct": bat["end"],
        "battery_drop_pct": bat["drop"]
    }])


def fuse_height(h, tof, alpha=HEIGHT_FUSION_ALPHA):
    """
    Fusion simple de la hauteur :
    alpha * h + (1 - alpha) * tof
    """
    if pd.isna(h):
        return tof
    if pd.isna(tof):
        return h
    return alpha * h + (1.0 - alpha) * tof


def reconstruct_trajectory_3d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruction 3D relative du drone :

    - X/Y : odométrie (vgx, vgy), avec yaw optionnel
    - Z   : fusion de la hauteur mesurée (h/tof), de vgz et de l'IMU verticale
    """
    traj = df.copy()

    # ------------------------------------------------------------------
    # Temps
    # ------------------------------------------------------------------
    traj["t_s"] = pd.to_numeric(traj["t_s"], errors="coerce")
    traj["dt_s"] = traj["t_s"].diff()
    traj.loc[(traj["dt_s"] < MIN_VALID_DT) | (traj["dt_s"] > MAX_VALID_DT), "dt_s"] = np.nan
    traj["dt_s"] = traj["dt_s"].fillna(TELEMETRY_PERIOD)

    # ------------------------------------------------------------------
    # Fusion hauteur
    # ------------------------------------------------------------------
    traj["h_fused"] = traj.apply(
        lambda row: fuse_height(row.get("h", np.nan), row.get("tof", np.nan)),
        axis=1
    )
    z_meas_m = pd.to_numeric(traj["h_fused"], errors="coerce") / 100.0

    # ------------------------------------------------------------------
    # Vitesses odométriques
    # ------------------------------------------------------------------
    vgx_cm_s = pd.to_numeric(traj.get("vgx", np.nan), errors="coerce").fillna(0.0)
    vgy_cm_s = pd.to_numeric(traj.get("vgy", np.nan), errors="coerce").fillna(0.0)
    vgz_cm_s = pd.to_numeric(traj.get("vgz", np.nan), errors="coerce").fillna(0.0)

    vgx_cm_s = exp_smooth(vgx_cm_s, VELOCITY_SMOOTH_ALPHA).fillna(0.0).clip(-MAX_ABS_VXY_CM_S, MAX_ABS_VXY_CM_S)
    vgy_cm_s = exp_smooth(vgy_cm_s, VELOCITY_SMOOTH_ALPHA).fillna(0.0).clip(-MAX_ABS_VXY_CM_S, MAX_ABS_VXY_CM_S)
    vgz_cm_s = exp_smooth(vgz_cm_s, VELOCITY_SMOOTH_ALPHA).fillna(0.0).clip(-MAX_ABS_VZ_CM_S, MAX_ABS_VZ_CM_S)

    speed_xy_cm_s = np.sqrt(vgx_cm_s**2 + vgy_cm_s**2)

    # ------------------------------------------------------------------
    # Détection du vol utile
    # ------------------------------------------------------------------
    flight_mask = (
        (traj["h_fused"].fillna(0.0) > FLIGHT_HEIGHT_THRESHOLD_CM) |
        (speed_xy_cm_s > FLIGHT_SPEED_THRESHOLD_CM_S) |
        (np.abs(vgz_cm_s) > FLIGHT_SPEED_THRESHOLD_CM_S)
    )
    traj["flight_mask"] = flight_mask.astype(int)

    # Blocage du bruit hors phase de vol
    ground_like = ~flight_mask
    vgx_cm_s.loc[ground_like & (np.abs(vgx_cm_s) < GROUND_SPEED_ZERO_CLAMP_CM_S)] = 0.0
    vgy_cm_s.loc[ground_like & (np.abs(vgy_cm_s) < GROUND_SPEED_ZERO_CLAMP_CM_S)] = 0.0
    vgz_cm_s.loc[ground_like & (np.abs(vgz_cm_s) < GROUND_SPEED_ZERO_CLAMP_CM_S)] = 0.0

    # ------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------
    yaw_deg = pd.to_numeric(traj.get("yaw", np.nan), errors="coerce")
    yaw_unwrapped_deg = unwrap_angle_deg(yaw_deg).ffill().fillna(0.0)

    roll_deg = pd.to_numeric(traj.get("roll", np.nan), errors="coerce").ffill().fillna(0.0)
    pitch_deg = pd.to_numeric(traj.get("pitch", np.nan), errors="coerce").ffill().fillna(0.0)

    # ------------------------------------------------------------------
    # Passage des vitesses XY vers le repère monde
    # ------------------------------------------------------------------
    vx_body_m_s = vgx_cm_s.to_numpy(dtype=float) / 100.0
    vy_body_m_s = vgy_cm_s.to_numpy(dtype=float) / 100.0
    vz_odom_m_s = vgz_cm_s.to_numpy(dtype=float) / 100.0

    vx_world_m_s, vy_world_m_s = body_velocity_to_world_xy(
        vx_body_m_s,
        vy_body_m_s,
        yaw_unwrapped_deg.to_numpy(dtype=float),
        use_yaw=USE_YAW_FOR_XY
    )

    # ------------------------------------------------------------------
    # IMU : projection vers l'axe vertical monde
    # ------------------------------------------------------------------
    agx = pd.to_numeric(traj.get("agx", np.nan), errors="coerce")
    agy = pd.to_numeric(traj.get("agy", np.nan), errors="coerce")
    agz = pd.to_numeric(traj.get("agz", np.nan), errors="coerce")

    agx = exp_smooth(agx, IMU_SMOOTH_ALPHA).ffill().fillna(0.0) * IMU_ACCEL_SCALE
    agy = exp_smooth(agy, IMU_SMOOTH_ALPHA).ffill().fillna(0.0) * IMU_ACCEL_SCALE
    agz = exp_smooth(agz, IMU_SMOOTH_ALPHA).ffill().fillna(0.0) * IMU_ACCEL_SCALE

    stationary_mask = (
        (traj["h_fused"].fillna(0.0) < MIN_FLIGHT_HEIGHT_CM) &
        (speed_xy_cm_s < GROUND_SPEED_ZERO_CLAMP_CM_S) &
        (np.abs(vgz_cm_s) < GROUND_SPEED_ZERO_CLAMP_CM_S)
    )

    bias_agx = estimate_bias_from_stationary(agx, stationary_mask)
    bias_agy = estimate_bias_from_stationary(agy, stationary_mask)
    bias_agz = estimate_bias_from_stationary(agz, stationary_mask)

    agx_dyn = agx - bias_agx
    agy_dyn = agy - bias_agy
    agz_dyn = agz - bias_agz

    az_world_dyn = np.array([
        body_accel_to_world_z(ax, ay, az, r, p)
        for ax, ay, az, r, p in zip(
            agx_dyn.to_numpy(dtype=float),
            agy_dyn.to_numpy(dtype=float),
            agz_dyn.to_numpy(dtype=float),
            roll_deg.to_numpy(dtype=float),
            pitch_deg.to_numpy(dtype=float)
        )
    ], dtype=float)

    az_world_dyn = np.clip(az_world_dyn, -MAX_ABS_AZ_M_S2, MAX_ABS_AZ_M_S2)

    # ------------------------------------------------------------------
    # Intégration 3D
    # ------------------------------------------------------------------
    n = len(traj)
    dt = traj["dt_s"].to_numpy(dtype=float)

    x = np.zeros(n, dtype=float)
    y = np.zeros(n, dtype=float)
    z = np.zeros(n, dtype=float)
    vz_fused = np.zeros(n, dtype=float)

    step_dist_2d_m = np.zeros(n, dtype=float)
    step_dist_3d_m = np.zeros(n, dtype=float)

    for i in range(1, n):
        # XY
        dx = vx_world_m_s[i] * dt[i]
        dy = vy_world_m_s[i] * dt[i]
        x[i] = x[i - 1] + dx
        y[i] = y[i - 1] + dy

        # Z
        vz_from_imu = vz_fused[i - 1] + az_world_dyn[i] * dt[i]
        vz_fused[i] = (1.0 - VZ_IMU_WEIGHT) * vz_odom_m_s[i] + VZ_IMU_WEIGHT * vz_from_imu

        z_pred = z[i - 1] + vz_fused[i] * dt[i]

        if np.isfinite(z_meas_m.iloc[i]):
            z[i] = (1.0 - Z_MEAS_WEIGHT) * z_pred + Z_MEAS_WEIGHT * z_meas_m.iloc[i]
        else:
            z[i] = z_pred

        if not bool(flight_mask.iloc[i]) and z[i] < 0.03:
            z[i] = 0.0
            vz_fused[i] = 0.0

        z[i] = max(0.0, z[i])

        dz = z[i] - z[i - 1]
        step_dist_2d_m[i] = np.sqrt(dx**2 + dy**2)
        step_dist_3d_m[i] = np.sqrt(dx**2 + dy**2 + dz**2)

    # ------------------------------------------------------------------
    # Colonnes de sortie
    # ------------------------------------------------------------------
    traj["yaw_unwrapped_deg"] = yaw_unwrapped_deg

    traj["vx_body_m_s"] = vx_body_m_s
    traj["vy_body_m_s"] = vy_body_m_s
    traj["vz_odom_m_s"] = vz_odom_m_s

    traj["vx_world_m_s"] = vx_world_m_s
    traj["vy_world_m_s"] = vy_world_m_s
    traj["vz_fused_m_s"] = vz_fused

    traj["az_world_dyn_m_s2"] = az_world_dyn
    traj["z_meas_m"] = z_meas_m

    traj["x_m"] = x
    traj["y_m"] = y
    traj["z_m"] = z

    traj["step_dist_2d_m"] = step_dist_2d_m
    traj["step_dist_3d_m"] = step_dist_3d_m
    traj["dist_2d_m"] = np.cumsum(step_dist_2d_m)
    traj["dist_3d_m"] = np.cumsum(step_dist_3d_m)

    return traj


def plot_quick(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    t = pd.to_numeric(df["t_s"], errors="coerce")

    def saveplot(ycol, ylabel):
        y = pd.to_numeric(df.get(ycol, np.nan), errors="coerce")
        mask = t.notna() & y.notna()

        if mask.sum() < 2:
            return

        plt.figure()
        plt.plot(t[mask], y[mask])
        plt.xlabel("Temps [s]")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(True, alpha=0.3)
        plt.savefig(outdir / f"{ycol}.png", bbox_inches="tight", dpi=150)
        plt.close()

    if "h" in df.columns and df["h"].notna().any():
        saveplot("h", "Hauteur [cm]")
    if "tof" in df.columns and df["tof"].notna().any():
        saveplot("tof", "ToF [cm]")
    if "h_fused" in df.columns and df["h_fused"].notna().any():
        saveplot("h_fused", "Hauteur fusionnée [cm]")

    saveplot("vxy", "Vitesse horizontale [cm/s]")
    saveplot("yaw", "Yaw [°]")
    saveplot("bat", "Batterie [%]")

    if "vx_world_m_s" in df.columns:
        saveplot("vx_world_m_s", "Vx monde [m/s]")
    if "vy_world_m_s" in df.columns:
        saveplot("vy_world_m_s", "Vy monde [m/s]")
    if "vz_fused_m_s" in df.columns:
        saveplot("vz_fused_m_s", "Vz fusionnée [m/s]")
    if "az_world_dyn_m_s2" in df.columns:
        saveplot("az_world_dyn_m_s2", "Az monde dynamique [m/s²]")
    if "z_m" in df.columns:
        saveplot("z_m", "Altitude reconstruite Z [m]")


def plot_trajectory_2d_projection(df: pd.DataFrame):
    """
    Projection 2D de la trajectoire 3D sur le plan XY.
    """
    if "x_m" not in df.columns or "y_m" not in df.columns:
        return

    x = pd.to_numeric(df["x_m"], errors="coerce")
    y = pd.to_numeric(df["y_m"], errors="coerce")
    t = pd.to_numeric(df["t_s"], errors="coerce")

    mask = x.notna() & y.notna() & t.notna()
    if mask.sum() < 2:
        print("Pas assez de points pour tracer la projection 2D.")
        return

    plt.figure(figsize=(7, 7))
    plt.plot(x[mask], y[mask], linewidth=2, label="Projection XY")
    plt.scatter(x[mask].iloc[0], y[mask].iloc[0], s=60, marker="o", label="Départ")
    plt.scatter(x[mask].iloc[-1], y[mask].iloc[-1], s=60, marker="x", label="Arrivée")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Projection 2D de la trajectoire 3D")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAJ2D_PNG_FILE, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_trajectory_3d(df: pd.DataFrame):
    """
    Trace la trajectoire 3D reconstruite.
    """
    required_cols = {"x_m", "y_m", "z_m", "t_s"}
    if not required_cols.issubset(df.columns):
        return

    x = pd.to_numeric(df["x_m"], errors="coerce")
    y = pd.to_numeric(df["y_m"], errors="coerce")
    z = pd.to_numeric(df["z_m"], errors="coerce")
    t = pd.to_numeric(df["t_s"], errors="coerce")

    mask = x.notna() & y.notna() & z.notna() & t.notna()
    if mask.sum() < 2:
        print("Pas assez de points pour tracer la trajectoire 3D.")
        return

    # Figure 3D simple
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x[mask], y[mask], z[mask], linewidth=2, label="Trajectoire 3D")
    ax.scatter(x[mask].iloc[0], y[mask].iloc[0], z[mask].iloc[0], s=60, marker="o", label="Départ")
    ax.scatter(x[mask].iloc[-1], y[mask].iloc[-1], z[mask].iloc[-1], s=60, marker="x", label="Arrivée")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Trajectoire 3D reconstruite du drone")
    ax.legend()
    plt.tight_layout()
    plt.savefig(TRAJ3D_PNG_FILE, dpi=180, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # Figure 3D colorée par le temps
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x[mask], y[mask], z[mask], c=t[mask], s=16)
    ax.plot(x[mask], y[mask], z[mask], alpha=0.35)
    ax.scatter(x[mask].iloc[0], y[mask].iloc[0], z[mask].iloc[0], s=60, marker="o", label="Départ")
    ax.scatter(x[mask].iloc[-1], y[mask].iloc[-1], z[mask].iloc[-1], s=60, marker="x", label="Arrivée")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Trajectoire 3D colorée par le temps")
    fig.colorbar(sc, ax=ax, label="Temps [s]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(TRAJ3D_TIME_PNG_FILE, dpi=180, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def post_flight_analysis():
    """
    Analyse des données de vol :
    1. chargement du CSV de télémétrie
    2. nettoyage des données
    3. fusion simple de hauteur
    4. reconstruction 3D
    5. calcul des statistiques du vol
    6. sauvegarde des résultats
    7. génération des graphiques
    """
    print("\n--- Programme en cours : ANALYSE POST-VOL ---\n")

    csv_path = Path(CSV_FILE)
    OUTDIR.mkdir(exist_ok=True)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("Aucune donnée de télémétrie disponible.")
        return

    try:
        # ------------------------------------------------------------------
        # Chargement et nettoyage
        # ------------------------------------------------------------------
        df = load_and_parse(csv_path)

        # ------------------------------------------------------------------
        # Fusion hauteur h / tof
        # ------------------------------------------------------------------
        df["h_fused"] = df.apply(
            lambda row: fuse_height(row["h"], row["tof"]),
            axis=1
        )

        # ------------------------------------------------------------------
        # Reconstruction 3D
        # ------------------------------------------------------------------
        df = reconstruct_trajectory_3d(df)

        # ------------------------------------------------------------------
        # Résumé
        # ------------------------------------------------------------------
        summary = compute_summary(df)

        # ------------------------------------------------------------------
        # Sauvegarde
        # ------------------------------------------------------------------
        df.to_csv(OUTDIR / "cleaned_parsed.csv", index=False)

        traj_export_cols = [
            "t_s", "dt_s", "flight_mask",
            "roll", "pitch", "yaw", "yaw_unwrapped_deg",
            "vgx", "vgy", "vgz", "vxy",
            "agx", "agy", "agz",
            "vx_world_m_s", "vy_world_m_s", "vz_odom_m_s", "vz_fused_m_s",
            "az_world_dyn_m_s2",
            "h", "tof", "h_fused", "z_meas_m",
            "x_m", "y_m", "z_m",
            "step_dist_2d_m", "step_dist_3d_m",
            "dist_2d_m", "dist_3d_m"
        ]
        traj_export_cols = [c for c in traj_export_cols if c in df.columns]
        df[traj_export_cols].to_csv(TRAJ3D_CSV_FILE, index=False)

        summary.to_csv(OUTDIR / "summary.csv", index=False)

        print("Résumé du vol :")
        print(summary.to_string(index=False))

        # ------------------------------------------------------------------
        # Graphiques
        # ------------------------------------------------------------------
        plot_quick(df, OUTDIR)
        plot_trajectory_2d_projection(df)
        plot_trajectory_3d(df)

        if SHOW_PLOTS:
            # Comparaison hauteur brute / ToF / fusion / Z reconstruite
            plt.figure()
            if df["h"].notna().any():
                plt.plot(df["t_s"], df["h"] / 100.0, label="h (drone) [m]")
            if df["tof"].notna().any():
                plt.plot(df["t_s"], df["tof"] / 100.0, label="tof (capteur) [m]")
            if df["h_fused"].notna().any():
                plt.plot(df["t_s"], df["h_fused"] / 100.0, label="hauteur fusionnée [m]")
            if "z_m" in df.columns and df["z_m"].notna().any():
                plt.plot(df["t_s"], df["z_m"], label="z reconstruite [m]")
            plt.xlabel("Temps [s]")
            plt.ylabel("Altitude [m]")
            plt.title("Fusion verticale et reconstruction Z")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()

            # Vitesses monde
            if {"vx_world_m_s", "vy_world_m_s", "vz_fused_m_s"}.issubset(df.columns):
                plt.figure()
                plt.plot(df["t_s"], df["vx_world_m_s"], label="Vx monde")
                plt.plot(df["t_s"], df["vy_world_m_s"], label="Vy monde")
                plt.plot(df["t_s"], df["vz_fused_m_s"], label="Vz fusionnée")
                plt.xlabel("Temps [s]")
                plt.ylabel("Vitesse [m/s]")
                plt.title("Vitesses reconstruites")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.show()

        print(f"\nFichiers générés :")
        print(f"- {OUTDIR / 'cleaned_parsed.csv'}")
        print(f"- {OUTDIR / 'summary.csv'}")
        print(f"- {TRAJ3D_CSV_FILE}")
        print(f"- {TRAJ2D_PNG_FILE}")
        print(f"- {TRAJ3D_PNG_FILE}")
        print(f"- {TRAJ3D_TIME_PNG_FILE}")

    except Exception as e:
        print(f"Analyse post-vol impossible : {e}")


##############################################################################
# ---------- COMMANDES DE VOL ----------
##############################################################################

def execute_command(cmd: str):
    """
    Interprète la commande tapée par l'utilisateur
    et appelle la bonne méthode du drone.
    """
    parts = cmd.strip().split()

    if not parts:
        return

    action = parts[0].lower()

    try:
        print(f"[COMMANDE] {cmd}")

        # ---------- Commandes simples ----------
        if action == "takeoff":
            with drone_lock:
                Drone.takeoff()

        elif action == "land":
            with drone_lock:
                Drone.land()

        elif action == "battery?":
            with drone_lock:
                print(f"Batterie : {Drone.get_battery()} %")

        elif action == "height?":
            try:
                with drone_lock:
                    print(f"Hauteur : {Drone.get_height()} cm")
            except Exception:
                state = Drone.get_current_state()
                print(f"Hauteur : {state.get('h', 'N/A')} cm")

        elif action == "speed?":
            state = Drone.get_current_state()
            print(
                "Vitesses [cm/s] | "
                f"vgx={state.get('vgx', 'N/A')} | "
                f"vgy={state.get('vgy', 'N/A')} | "
                f"vgz={state.get('vgz', 'N/A')}"
            )

        elif action == "time?":
            state = Drone.get_current_state()
            print(f"Temps moteur / vol : {state.get('time', 'N/A')} s")

        elif action == "emergency":
            with drone_lock:
                Drone.emergency()

        # ---------- Gestion vidéo ----------
        elif action == "video_on":
            start_video_thread()

        elif action == "video_off":
            stop_video_thread()

        elif action == "cutvideo":
            cutVideo()

        # ---------- Mouvements avec argument ----------
        elif action in {"up", "down", "left", "right", "forward", "back", "cw", "ccw"}:
            if len(parts) != 2:
                print("Commande invalide. Exemple : forward 50")
                return

            value = int(parts[1])

            with drone_lock:
                if action == "up":
                    Drone.move_up(value)
                elif action == "down":
                    Drone.move_down(value)
                elif action == "left":
                    Drone.move_left(value)
                elif action == "right":
                    Drone.move_right(value)
                elif action == "forward":
                    Drone.move_forward(value)
                elif action == "back":
                    Drone.move_back(value)
                elif action == "cw":
                    Drone.rotate_clockwise(value)
                elif action == "ccw":
                    Drone.rotate_counter_clockwise(value)

        else:
            print("Commande inconnue.")

    except ValueError:
        print("Erreur : l'argument doit être un entier.")
    except Exception as e:
        print(f"[COMMANDE] Erreur : {e}")


##############################################################################
# ---------- MAIN ----------
##############################################################################

if __name__ == "__main__":

    ensure_directories()

    try:
        # ---------- ETAPE 1 : connexion au drone ----------
        connect_drone()

        # ---------- ETAPE 2 : lancement du thread de télémétrie ----------
        stop_telemetry_event.clear()
        telemetry_thread = threading.Thread(target=listen_state, daemon=True)
        telemetry_thread.start()

        # ---------- ETAPE 3 : boucle principale ----------
        while True:
            print("\nTAPEZ 'exit' POUR ARRETER LE PROGRAMME")
            print("\nCOMMANDES DISPONIBLES :")
            print(", ".join(COMMANDS))

            cmd = input("Commande: ").strip()

            if cmd.lower() == "exit":
                break

            execute_command(cmd)

    except KeyboardInterrupt:
        print("\nArrêt manuel du programme.")

    except Exception as e:
        print(f"Impossible de lancer le programme : {e}")

    finally:
        print("\n--- ARRET PROPRE DU PROGRAMME ---\n")

        # Arrêt télémétrie
        stop_telemetry_event.set()
        if telemetry_thread is not None and telemetry_thread.is_alive():
            telemetry_thread.join(timeout=3)

        # Arrêt vidéo
        if video_thread is not None and video_thread.is_alive():
            stop_video_event.set()
            video_thread.join(timeout=8)

        # Coupure du stream si encore actif
        safe_streamoff()

        # Découpe auto de la vidéo si elle existe
        if VIDEO_FILE.exists() and VIDEO_FILE.stat().st_size > 0:
            try:
                cutVideo()
            except Exception as e:
                print(f"[POST-VIDEO] Découpage automatique impossible : {e}")

        # Analyse post-vol
        post_flight_analysis()

        print("\n--- Programme terminé ---\n")
