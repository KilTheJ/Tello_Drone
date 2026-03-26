# Drone Tello Telemetry Lab

Projet d’analyse et de fusion de données de télémétrie pour drones DJI Tello.  
Le but est d’acquérir, enregistrer, traiter et visualiser les données de vol afin de reconstruire des trajectoires, analyser les signaux embarqués et expérimenter des approches simples de fusion de capteurs.  
Le projet s’inscrit dans une démarche de laboratoire logiciel de télémétrie pour rejouer et étudier des vols réels.  

## Fonctionnalités

- Pilotage du drone DJI Tello
- Acquisition de la télémétrie en temps réel
- Enregistrement des données dans des fichiers exploitables
- Capture vidéo pendant le vol
- Extraction d’images / frames
- Analyse post-vol avec Python
- Visualisation des signaux :
  - batterie
  - température
  - angles
  - vitesses
  - accélérations IMU
- Reconstruction de trajectoire :
  - 2D
  - 3D
- Fusion simple de données capteurs :
  - hauteur estimée
  - ToF
  - odométrie / IMU selon le code final

---

## Prérequis

### Matériel

* 1 drone DJI Tello
* 1 ordinateur portable avec Wi-Fi
* Batterie suffisamment chargée sur le drone
* Un espace sécurisé pour les essais

### Logiciels

* Python 3.12
* pip
* Git
* Un IDE ou éditeur comme :

  * VS Code
  * PyCharm
  * Spyder (utilisé pour ce projet)
  * Jupyter Notebook

---

## Dépendances Python

Le projet utilise principalement :

* `djitellopy`
* `numpy`
* `pandas`
* `matplotlib`
* `plotly`
* `scipy`
* `opencv-python`

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/TON-UTILISATEUR/TON-DEPOT.git
cd TON-DEPOT
```

### 2. Créer un environnement virtuel

#### Sur macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Sur Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Mettre à jour pip

```bash
python -m pip install --upgrade pip
```

### 4. Installer les dépendances

```bash
pip install djitellopy numpy pandas matplotlib plotly scipy opencv-python
```

---

## Vérification rapide de l’installation (obligatoire !)

Tester que les bibliothèques principales s’importent correctement :

```bash
python -c "import cv2, numpy, pandas, matplotlib, scipy, plotly; print('OK')"
```

Tester ensuite la bibliothèque Tello :

```bash
python -c "from djitellopy import Tello; print('djitellopy OK')"
```

---

## Connexion au drone

Avant d’exécuter le code :

1. Allumer le drone DJI Tello
2. Connecter l’ordinateur au réseau Wi-Fi du drone
   Exemple : `TELLO-XXXXXX`
3. Vérifier que l’ordinateur est bien connecté au drone et non à un autre réseau
4. Fermer les applications qui pourraient utiliser la caméra ou saturer le Wi-Fi si nécessaire

> Le Tello communique via Wi-Fi.
> Si le PC n’est pas connecté au bon réseau, le script ne parlera à personne.

---

## Ce que fait le programme

L’exécution inclut :

* connexion au drone
* affichage des commandes de vol et vidéo
* récupération des états de télémétrie
* enregistrement dans un CSV
* démarrage du flux vidéo
* sauvegarde de la vidéo
* extraction de frames
* calculs de reconstruction de trajectoire
* génération de graphes et fichiers de synthèse

Les résultats sont sauvegardés dans un dossier :

```bash
tello_out/
```

avec :

* `data_tello.csv` : données brutes de télémétrie
* `cleaned_parsed.csv` : données nettoyées / interprétées
* `summary.csv` : résumé du vol
* `video/output.mp4` : vidéo enregistrée
* `video/matrices.txt` : matrices ou informations issues des frames
* autres fichiers avec les graphes

---

## Utilisation typique

### 1. Préparer le drone

* batterie chargée
* zone de vol dégagée
* drone posé à plat

### 2. Connecter le PC au Tello

* connexion Wi-Fi au drone

### 3. Lancer le script

```bash
python main.py
```

### 4. Exécuter le vol

Le vol est manuel avec commandes prédéfinies

### 5. Récupérer les résultats

À la fin, consulter les fichiers générés dans `tello_out/`

---

## Analyse des résultats

Les données produites permettent notamment de :

* tracer l’évolution de la batterie
* analyser les températures
* afficher les angles `pitch`, `roll`, `yaw`
* visualiser les vitesses `vgx`, `vgy`, `vgz`
* exploiter les accélérations `agx`, `agy`, `agz`
* reconstruire une trajectoire 2D/3D
* comparer la hauteur estimée avec les mesures ToF
* tester des filtrages ou fusions simples

---

## Problèmes fréquents

### 1. `ModuleNotFoundError`

Une dépendance Python n’est pas installée.

Solution :

Installer le module manquant individuellement.

---

### 2. Le drone ne se connecte pas

Ca vient souvent de là.

Vérifier :

* que le drone est allumé
* que le PC est connecté au Wi-Fi du Tello
* qu’aucun autre réseau ne prend la priorité
* que le pare-feu ne bloque pas la communication

---

### 3. `cv2` ne s’importe pas

Problème d’environnement Python ou installation OpenCV cassée.

Solution :

```bash
pip install --upgrade pip
pip install opencv-python
```

Puis tester :

```bash
python -c "import cv2; print(cv2.__version__)"
```

---

### 4. Le flux vidéo ne démarre pas

Vérifier :

* la bonne connexion au drone
* que le code appelle bien l’activation du stream
* qu’aucune autre application ne monopolise le flux

---

### 5. Le script se lance mais rien ne s’enregistre

Vérifier :

* les droits d’écriture dans le dossier du projet
* que les dossiers de sortie sont bien créés
* que le vol a réellement démarré
* qu’aucune erreur silencieuse n’a interrompu un thread

---

## Recommandations

* Toujours tester d’abord avec un vol très court
* Vérifier la batterie avant chaque session
* Ne pas lancer directement des manœuvres agressives
* Commencer par valider :

  1. la connexion
  2. la télémétrie
  3. la vidéo
  4. l’analyse post-vol

---

## Auteurs

Projet réalisé dans le cadre d’un travail d’analyse et de fusion de données de télémétrie pour drone DJI Tello, dans le cadre de la 3ème année
du parcours Systèmes Numériques de l'école d'ingénieurs de SeaTech Toulon.
Etudiants : A. DURANDO ; K. JOLIVET ; O. PIAT
