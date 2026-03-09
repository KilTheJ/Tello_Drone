import cv2
import numpy as np
import threading
from djitellopy import tello  # si tu utilises djitellopy

# Path 
path = "output/"
f = open(path + "matrices.txt", "a")

# Start Connection With Drone
Drone = tello.Tello()
Drone.connect()

# Gestion du thread pour la vidéo
stop_event = threading.Event()

def getVideo():
    # Start Camera Display Stream
    Drone.streamon()

    frame_width = 1080
    frame_height = 720
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(path + "output.mp4", fourcc, 60.0, (frame_width, frame_height))

    while not stop_event.is_set():
        # Get frame from drone camera 
        frame = Drone.get_frame_read().frame
        frame = cv2.resize(frame, (frame_width, frame_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Sauvegarde de la frame 
        out.write(frame)

        # Affichage
        cv2.imshow('RGB Image', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("J'me coupe la vidéo")
            stop_event.set()
            break

    # Libération des ressources vidéo
    out.release()
    Drone.streamoff()
    cv2.destroyAllWindows()

def cutVideo():
    vidcap = cv2.VideoCapture(path + 'output.mp4')
    success, image = vidcap.read()
    count = 0

    while success:
        cv2.imwrite(path + f"frame{count}.png", image)
        success, image = vidcap.read()
        count += 1
        print('Frame :', count, success)

    vidcap.release()

    # Conversion en matrices
    for i in range(min(15, count)):
        img = cv2.imread(path + f"frame{i}.png", 0)
        np.savetxt(f, img)

# Lancement du thread vidéo
t = threading.Thread(target=getVideo, daemon=True)
t.start()

# On attend que le thread vidéo se termine
t.join()

# Une fois le thread terminé, on découpe la vidéo
cutVideo()

# On ferme le fichier à la fin
f.close()
