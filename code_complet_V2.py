# -*- coding: utf-8 -*-
"""
Programme Tello complet évolué :
- pilotage du drone
- acquisition télémétrie
- acquisition vidéo
- découpe vidéo en frames
- export matrices
- analyse post-vol
- reconstruction de trajectoire 2D à partir de l'odométrie (vgx, vgy, yaw)
- fusion simple de hauteur h / tof
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

# Fichiers de sortie pour la reconstruction
TRAJ_CSV_FILE = OUTDIR / "trajectory_2d.csv"
TRAJ_PNG_FILE = OUTDIR / "trajectory_2d.png"
TRAJ_TIME_PNG_FILE = OUTDIR / "trajectory_2d_colored_time.png"

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

# Paramètres de reconstruction
MIN_VALID_DT = 1e-3
MAX_VALID_DT = 0.5

# Lissage exponentiel des vitesses
VELOCITY_SMOOTH_ALPHA = 0.25

# Hauteur mini pour considérer que le drone est réellement en vol
MIN_FLIGHT_HEIGHT_CM = 8.0

# Seuil de vitesse sous lequel on force à 0 près du sol
GROUND_SPEED_ZERO_CLAMP_CM_S = 8.0

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
    for c in ["vgx", "vgy", "vgz", "yaw", "h", "tof", "bat", "agx", "agy", "agz"]:
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
    vxy = pd.to_numeric(df["vxy"], errors="coerce")
    h = pd.to_numeric(df.get("h", np.nan), errors="coerce")
    tof = pd.to_numeric(df.get("tof", np.nan), errors="coerce")
    yaw = pd.to_numeric(df.get("yaw", np.nan), errors="coerce")

    duration_s = float(t.max() - t.min()) if t.notna().any() else np.nan
    vxy_max_m_s = float((vxy.max() if pd.notna(vxy.max()) else np.nan) / 100.0)
    h_max_m = float((h.max() if pd.notna(h.max()) else np.nan) / 100.0)
    tof_max_m = float((tof.max() if pd.notna(tof.max()) else np.nan) / 100.0)
    yaw_span = float(yaw.max() - yaw.min()) if yaw.notna().any() else np.nan

    bat = battery_stats(df)

    traveled_2d_m = np.nan
    if "step_dist_m" in df.columns:
        step_dist = pd.to_numeric(df["step_dist_m"], errors="coerce")
        if step_dist.notna().any():
            traveled_2d_m = float(step_dist.fillna(0).sum())

    return pd.DataFrame([{
        "samples": int(len(df)),
        "duration_s": duration_s,
        "vxy_max_m_s": vxy_max_m_s,
        "h_max_m": h_max_m,
        "tof_max_m": tof_max_m,
        "yaw_span_deg": yaw_span,
        "traveled_2d_m": traveled_2d_m,
        "battery_start_pct": bat["start"],
        "battery_end_pct": bat["end"],
        "battery_drop_pct": bat["drop"]
    }])


def fuse_height(h, tof, alpha=0.4):
    """
    Fusion simple de la hauteur :
    alpha * h + (1 - alpha) * tof
    """
    if pd.isna(h):
        return tof
    if pd.isna(tof):
        return h
    return alpha * h + (1.0 - alpha) * tof


def reconstruct_trajectory_2d(df: pd.DataFrame) -> pd.DataFrame:
    traj = df.copy()

    # ----------------------------------------------------------------------
    # Temps et pas de temps
    # ----------------------------------------------------------------------
    traj["t_s"] = pd.to_numeric(traj["t_s"], errors="coerce")
    traj["dt_s"] = traj["t_s"].diff()

    # Nettoyage des dt aberrants
    traj.loc[(traj["dt_s"] < MIN_VALID_DT) | (traj["dt_s"] > MAX_VALID_DT), "dt_s"] = np.nan
    traj["dt_s"] = traj["dt_s"].fillna(TELEMETRY_PERIOD)

    # ----------------------------------------------------------------------
    # Fusion de hauteur
    # ----------------------------------------------------------------------
    traj["h_fused"] = traj.apply(
        lambda row: fuse_height(row.get("h", np.nan), row.get("tof", np.nan)),
        axis=1
    )

    # ----------------------------------------------------------------------
    # Données odométriques
    # ----------------------------------------------------------------------
    vgx_cm_s = pd.to_numeric(traj.get("vgx", np.nan), errors="coerce").fillna(0.0)
    vgy_cm_s = pd.to_numeric(traj.get("vgy", np.nan), errors="coerce").fillna(0.0)
    yaw_deg = pd.to_numeric(traj.get("yaw", np.nan), errors="coerce")

    # Lissage léger des vitesses
    vgx_cm_s = exp_smooth(vgx_cm_s, VELOCITY_SMOOTH_ALPHA).fillna(0.0)
    vgy_cm_s = exp_smooth(vgy_cm_s, VELOCITY_SMOOTH_ALPHA).fillna(0.0)

    # Yaw déroulé pour éviter les sauts
    yaw_deg_unwrapped = unwrap_angle_deg(yaw_deg).ffill().fillna(0.0)
    yaw_rad = np.deg2rad(yaw_deg_unwrapped.to_numpy(dtype=float))

    # ----------------------------------------------------------------------
    # Suppression des "mouvements fantômes" au sol
    # ----------------------------------------------------------------------
    h_fused = pd.to_numeric(traj["h_fused"], errors="coerce")
    near_ground = h_fused.fillna(0.0) < MIN_FLIGHT_HEIGHT_CM

    speed_norm = np.sqrt(vgx_cm_s**2 + vgy_cm_s**2)
    clamp_mask = near_ground & (speed_norm < GROUND_SPEED_ZERO_CLAMP_CM_S)

    vgx_cm_s.loc[clamp_mask] = 0.0
    vgy_cm_s.loc[clamp_mask] = 0.0

    # ----------------------------------------------------------------------
    # Passage repère drone -> repère monde
    # ----------------------------------------------------------------------
    vx_body_m_s = vgx_cm_s.to_numpy(dtype=float) / 100.0
    vy_body_m_s = vgy_cm_s.to_numpy(dtype=float) / 100.0

    vx_world_m_s = np.cos(yaw_rad) * vx_body_m_s - np.sin(yaw_rad) * vy_body_m_s
    vy_world_m_s = np.sin(yaw_rad) * vx_body_m_s + np.cos(yaw_rad) * vy_body_m_s

    traj["vx_body_m_s"] = vx_body_m_s
    traj["vy_body_m_s"] = vy_body_m_s
    traj["vx_world_m_s"] = vx_world_m_s
    traj["vy_world_m_s"] = vy_world_m_s
    traj["yaw_unwrapped_deg"] = yaw_deg_unwrapped

    # ----------------------------------------------------------------------
    # Intégration
    # ----------------------------------------------------------------------
    x = np.zeros(len(traj), dtype=float)
    y = np.zeros(len(traj), dtype=float)
    step_dist = np.zeros(len(traj), dtype=float)

    dt = traj["dt_s"].to_numpy(dtype=float)

    for i in range(1, len(traj)):
        dx = vx_world_m_s[i] * dt[i]
        dy = vy_world_m_s[i] * dt[i]

        x[i] = x[i - 1] + dx
        y[i] = y[i - 1] + dy
        step_dist[i] = np.sqrt(dx**2 + dy**2)

    traj["x_m"] = x
    traj["y_m"] = y
    traj["step_dist_m"] = step_dist
    traj["dist_2d_m"] = np.cumsum(step_dist)

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


def plot_trajectory_2d(df: pd.DataFrame):
    """
    Trace la trajectoire 2D reconstruite.
    """
    if "x_m" not in df.columns or "y_m" not in df.columns:
        return

    x = pd.to_numeric(df["x_m"], errors="coerce")
    y = pd.to_numeric(df["y_m"], errors="coerce")
    t = pd.to_numeric(df["t_s"], errors="coerce")

    mask = x.notna() & y.notna() & t.notna()
    if mask.sum() < 2:
        print("Pas assez de points pour tracer la trajectoire 2D.")
        return

    # Figure classique
    plt.figure(figsize=(7, 7))
    plt.plot(x[mask], y[mask], linewidth=2, label="Trajectoire reconstruite")
    plt.scatter(x[mask].iloc[0], y[mask].iloc[0], s=60, marker="o", label="Départ")
    plt.scatter(x[mask].iloc[-1], y[mask].iloc[-1], s=60, marker="x", label="Arrivée")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Trajectoire 2D reconstruite du drone")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAJ_PNG_FILE, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    # Figure colorée par le temps
    plt.figure(figsize=(7, 7))
    sc = plt.scatter(x[mask], y[mask], c=t[mask], s=18)
    plt.plot(x[mask], y[mask], alpha=0.4)
    plt.colorbar(sc, label="Temps [s]")
    plt.scatter(x[mask].iloc[0], y[mask].iloc[0], s=60, marker="o", label="Départ")
    plt.scatter(x[mask].iloc[-1], y[mask].iloc[-1], s=60, marker="x", label="Arrivée")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Trajectoire 2D colorée par le temps")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAJ_TIME_PNG_FILE, dpi=180, bbox_inches="tight")
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
    4. reconstruction 2D
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
            lambda row: fuse_height(row["h"], row["tof"], alpha=0.4),
            axis=1
        )

        # ------------------------------------------------------------------
        # Reconstruction 2D
        # ------------------------------------------------------------------
        df = reconstruct_trajectory_2d(df)

        # ------------------------------------------------------------------
        # Résumé du vol
        # ------------------------------------------------------------------
        summary = compute_summary(df)

        # ------------------------------------------------------------------
        # Sauvegarde
        # ------------------------------------------------------------------
        df.to_csv(OUTDIR / "cleaned_parsed.csv", index=False)

        traj_export_cols = [
            "t_s", "dt_s",
            "yaw", "yaw_unwrapped_deg",
            "vgx", "vgy", "vxy",
            "vx_world_m_s", "vy_world_m_s",
            "h", "tof", "h_fused",
            "x_m", "y_m", "step_dist_m", "dist_2d_m"
        ]
        traj_export_cols = [c for c in traj_export_cols if c in df.columns]
        df[traj_export_cols].to_csv(TRAJ_CSV_FILE, index=False)

        summary.to_csv(OUTDIR / "summary.csv", index=False)

        print("Résumé du vol :")
        print(summary.to_string(index=False))

        # ------------------------------------------------------------------
        # Graphiques
        # ------------------------------------------------------------------
        plot_quick(df, OUTDIR)
        plot_trajectory_2d(df)

        if SHOW_PLOTS:
            # Comparaison hauteur brute / ToF / fusion
            if df["h"].notna().any() or df["tof"].notna().any():
                plt.figure()
                if df["h"].notna().any():
                    plt.plot(df["t_s"], df["h"], label="h (drone)")
                if df["tof"].notna().any():
                    plt.plot(df["t_s"], df["tof"], label="tof (capteur)")
                if df["h_fused"].notna().any():
                    plt.plot(df["t_s"], df["h_fused"], label="hauteur fusionnée")
                plt.xlabel("Temps [s]")
                plt.ylabel("Hauteur [cm]")
                plt.title("Fusion de capteurs pour l'estimation de la hauteur")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.show()

            # Vitesses monde
            if "vx_world_m_s" in df.columns and "vy_world_m_s" in df.columns:
                plt.figure()
                plt.plot(df["t_s"], df["vx_world_m_s"], label="Vx monde")
                plt.plot(df["t_s"], df["vy_world_m_s"], label="Vy monde")
                plt.xlabel("Temps [s]")
                plt.ylabel("Vitesse [m/s]")
                plt.title("Vitesses dans le repère monde")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.show()

        print(f"\nFichiers générés :")
        print(f"- {OUTDIR / 'cleaned_parsed.csv'}")
        print(f"- {OUTDIR / 'summary.csv'}")
        print(f"- {TRAJ_CSV_FILE}")
        print(f"- {TRAJ_PNG_FILE}")
        print(f"- {TRAJ_TIME_PNG_FILE}")

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
