
""" Created on Mon Mar  9 16:10:47 2026 """

""" Created on Mon Mar  9 2026 """

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

# Liste des commandes affichées à l'utilisateur
COMMANDS = [
    "takeoff", "land", "streamon", "streamoff", "battery?",
    "height?", "speed?", "time?", "emergency",
    "up x", "down x", "left x", "right x", "forward x", "back x",
    "cw x", "ccw x",
    "video_on", "video_off", "cutvideo",
    "exit"
]


##############################################################################
# ---------- VARIABLES GLOBALES PARTAGEES ----------
##############################################################################

# Objet drone
Drone = None

# Gestion des threads
stop_telemetry_event = threading.Event()
stop_video_event = threading.Event()

telemetry_thread = None
video_thread = None

# Indique si la vidéo tourne déjà
video_running = False


##############################################################################
# ---------- OUTILS DE BASE ----------
##############################################################################

def ensure_directories():
    """Crée les dossiers de sortie si besoin."""
    OUTDIR.mkdir(exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def safe_int(value, default=np.nan):
    """Convertit proprement une valeur en int ou renvoie NaN."""
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=np.nan):
    """Convertit proprement une valeur en float ou renvoie NaN."""
    try:
        return float(value)
    except Exception:
        return default


##############################################################################
# ---------- CONNEXION AU DRONE ----------
##############################################################################

def connect_drone():
    """Connexion au drone avec djitellopy."""
    global Drone

    print("\n--- Programme en cours : CONNEXION AU DRONE ---\n")

    Drone = tello.Tello()
    Drone.connect()

    battery = Drone.get_battery()
    print(f"Connexion OK au drone | Batterie : {battery}%")

    return Drone


##############################################################################
# ---------- ACQUISITION DE LA TELEMETRIE ----------
##############################################################################

def listen_state():
    """
    Fonction qui interroge régulièrement le drone,
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
                # Récupération de l'état complet du drone sous forme de dictionnaire
                state = Drone.get_current_state()

                # Si aucun état n'est renvoyé, on attend un peu puis on recommence
                if not state:
                    time.sleep(TELEMETRY_PERIOD)
                    continue

                # Initialisation de l'écriture CSV avec les clés disponibles
                if writer is None:
                    fieldnames = ["timestamp"] + list(state.keys())
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()

                # Ecriture d'une ligne avec timestamp local + état du drone
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

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter(
        str(VIDEO_FILE),
        fourcc,
        VIDEO_FPS,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    try:
        Drone.streamon()
        frame_reader = Drone.get_frame_read()
        video_running = True

        while not stop_video_event.is_set():
            frame = frame_reader.frame

            if frame is None:
                continue

            # Redimensionnement de la frame
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Sauvegarde de la frame dans la vidéo
            out.write(frame)

            # Affichage temps réel
            cv2.imshow("Tello Video", frame)
            key = cv2.waitKey(1) & 0xFF

            # Touche q pour couper la vidéo
            if key == ord("q"):
                print("J'me coupe la vidéo")
                stop_video_event.set()
                break

    except Exception as e:
        print(f"[VIDEO] Erreur : {e}")

    finally:
        out.release()
        cv2.destroyAllWindows()

        try:
            Drone.streamoff()
        except Exception:
            pass

        video_running = False
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
    video_thread.join(timeout=5)
    print("Thread vidéo stoppé.")


def cutVideo():
    """
    Découpe la vidéo enregistrée en frames PNG,
    puis convertit les premières images en matrices texte.
    """

    print("\n--- Programme en cours : DECOUPAGE DE LA VIDEO ---\n")

    if not VIDEO_FILE.exists() or VIDEO_FILE.stat().st_size == 0:
        print("Aucune vidéo disponible à découper.")
        return

    vidcap = cv2.VideoCapture(str(VIDEO_FILE))
    success, image = vidcap.read()
    count = 0

    while success:
        frame_path = VIDEO_DIR / f"frame{count}.png"
        cv2.imwrite(str(frame_path), image)

        success, image = vidcap.read()
        count += 1
        print("Frame :", count, success)

    vidcap.release()

    with open(MATRICES_FILE, "w", encoding="utf-8") as f:
        for i in range(min(15, count)):
            img_path = VIDEO_DIR / f"frame{i}.png"
            img = cv2.imread(str(img_path), 0)

            if img is not None:
                f.write(f"\n--- MATRICE FRAME {i} ---\n")
                np.savetxt(f, img, fmt="%d")

    print(f"Découpage terminé | {count} frames extraites.")


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

    # Harmonisation des noms éventuels
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

    # Si un champ 'state' existe, on le parse
    if "state" in df_raw.columns:
        parsed = pd.DataFrame([parse_state_string(s) for s in df_raw["state"]])
        extras = [c for c in df_raw.columns if c not in parsed.columns]
        df = pd.concat(
            [df_raw[extras].reset_index(drop=True), parsed.reset_index(drop=True)],
            axis=1
        )
    else:
        df = df_raw.copy()

    # Ajout des colonnes manquantes
    for c in SDK_FIELDS:
        if c not in df.columns:
            df[c] = np.nan

    # Série temporelle
    df["t_s"] = pd.to_numeric(best_time_series(df), errors="coerce")

    # Vitesse horizontale
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

    print("\n--- Programme en cours : ETAPE analyse post-vol ---\n")

    csv_path = Path(CSV_FILE)
    OUTDIR.mkdir(exist_ok=True)

    # Vérifier si le fichier existe et contient des données
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("Aucune donnée de télémétrie disponible (drone non connecté ou aucun vol).")
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
            Drone.takeoff()

        elif action == "land":
            Drone.land()

        elif action == "battery?":
            print(f"Batterie : {Drone.get_battery()} %")

        elif action == "height?":
            try:
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
            Drone.emergency()

        # ---------- Gestion vidéo ----------
        elif action == "video_on":
            start_video_thread()

        elif action == "video_off":
            stop_video_thread()

        elif action == "streamon":
            start_video_thread()

        elif action == "streamoff":
            stop_video_thread()

        elif action == "cutvideo":
            cutVideo()

        # ---------- Mouvements avec argument ----------
        elif action in {"up", "down", "left", "right", "forward", "back", "cw", "ccw"}:
            if len(parts) != 2:
                print("Commande invalide. Exemple : forward 50")
                return

            value = int(parts[1])

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

        # ---------- ETAPE 3 : boucle principale de commande ----------
        while True:
            print("\nTAPEZ exit POUR ARRETER LE VOL ET PASSER AU POST TRAITEMENT :")
            print("\nCOMMANDES DE VOL DISPONIBLES :")
            print(", ".join(COMMANDS))

            cmd = input("Commande: ").strip()

            if cmd.lower() == "exit":
                break

            execute_command(cmd)

    except KeyboardInterrupt:
        print("Arrêt manuel")

    except Exception as e:
        print(f"Impossible de lancer le programme : {e}")

    finally:
        # ---------- ETAPE 4 : arrêt propre des threads ----------
        stop_telemetry_event.set()

        if telemetry_thread is not None and telemetry_thread.is_alive():
            telemetry_thread.join(timeout=2)

        if video_thread is not None and video_thread.is_alive():
            stop_video_event.set()
            video_thread.join(timeout=5)

        # ---------- ETAPE 5 : coupure du stream si besoin ----------
        try:
            Drone.streamoff()
        except Exception:
            pass

        # ---------- ETAPE 6 : analyse post-vol ----------
        post_flight_analysis()

        print("\n--- Programme terminé ---\n")