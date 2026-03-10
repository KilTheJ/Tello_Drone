# -*- coding: utf-8 -*-
"""
Programme Tello complet :
- pilotage du drone
- acquisition télémétrie
- acquisition vidéo
- découpe vidéo en frames
- export matrices
- analyse post-vol
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
                # Lecture de l'état courant du drone
                state = Drone.get_current_state()

                if not state:
                    time.sleep(TELEMETRY_PERIOD)
                    continue

                # Création de l'en-tête CSV à partir des clés du dictionnaire
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

        # On laisse un petit délai pour que le flux s'initialise
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

            # Copie locale pour éviter des effets de concurrence
            frame = frame.copy()

            # Redimensionnement
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Sauvegarde dans le MP4
            out.write(frame)

            # Affichage temps réel
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

    # Sécurité : on ne découpe pas tant que la vidéo est en cours d'écriture
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
            img = cv2.imread(str(img_path), 0)  # niveau de gris

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

    df["t_s"] = pd.to_numeric(best_time_series(df), errors="coerce")

    vgx = pd.to_numeric(df.get("vgx", np.nan), errors="coerce")
    vgy = pd.to_numeric(df.get("vgy", np.nan), errors="coerce")
    df["vxy"] = np.sqrt(np.square(vgx) + np.square(vgy))

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

    duration_s = float(t.max() - t.min()) if t.notna().any() else np.nan
    vxy_max_m_s = float((vxy.max() if pd.notna(vxy.max()) else np.nan) / 100.0)
    h_max_m = float((h.max() if pd.notna(h.max()) else np.nan) / 100.0)

    yaw = pd.to_numeric(df.get("yaw", np.nan), errors="coerce")
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
        plt.savefig(outdir / f"{ycol}.png", bbox_inches="tight", dpi=150)
        plt.close()

    if "h" in df.columns and df["h"].notna().any():
        saveplot("h", "Hauteur [cm]")
    elif "tof" in df.columns and df["tof"].notna().any():
        saveplot("tof", "ToF [cm]")

    saveplot("vxy", "Vitesse horizontale [cm/s]")
    saveplot("yaw", "Yaw [°]")
    saveplot("bat", "Batterie [%]")


def post_flight_analysis():
    """
    Analyse des données de vol :
    1. chargement du CSV de télémétrie
    2. nettoyage des données
    3. calcul des statistiques du vol
    4. sauvegarde des résultats
    5. génération des graphiques
    """

    print("\n--- Programme en cours : ANALYSE POST-VOL ---\n")

    csv_path = Path(CSV_FILE)
    OUTDIR.mkdir(exist_ok=True)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("Aucune donnée de télémétrie disponible.")
        return

    try:
        df = load_and_parse(csv_path)
        summary = compute_summary(df)

        df.to_csv(OUTDIR / "cleaned_parsed.csv", index=False)
        summary.to_csv(OUTDIR / "summary.csv", index=False)

        print("Résumé du vol :")
        print(summary.to_string(index=False))

        if SHOW_PLOTS:
            plot_quick(df, OUTDIR)

            if "h" in df.columns and df["h"].notna().any():
                plt.figure()
                plt.plot(df["t_s"], df["h"])
                plt.xlabel("Temps [s]")
                plt.ylabel("Hauteur [cm]")
                plt.title("Évolution de la hauteur")
                plt.show()

            elif "tof" in df.columns and df["tof"].notna().any():
                plt.figure()
                plt.plot(df["t_s"], df["tof"])
                plt.xlabel("Temps [s]")
                plt.ylabel("ToF [cm]")
                plt.title("Évolution ToF = distance au sol")
                plt.show()

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
