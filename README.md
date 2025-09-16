# Tello_Drone
L’objectif final est de mettre en place un laboratoire logiciel de télémétrie permettant de rejouer, analyser et expérimenter sur des vols réels enregistrés.

Objectifs du projet :
1. Acquérir et stocker les données de télémétrie d’un Tello lors de vols réels (export de
logs).
2. Explorer les données avec Python (Pandas, Matplotlib, Plotly...).
3. Développer des outils de visualisation : évolution batterie/température, signaux IMU
(accélérations, vitesses, angles), trajectoires 2D/3D.
4. Implémenter des algorithmes de fusion (par ex. moyenne glissante, Kalman simple).
5. Expérimenter la reconstruction de trajectoires 2D/3D à partir de l’odométrie et des
mesures IMU/ToF.
6. Étudier la détection et le suivi de personnes/objets à partir de la télémétrie visuelle.
   
Matériel & outils :
- DJI Tello, PC portable
- Python
- Pandas, Numpy, Matplotlib, Plotly, Scipy, OpenCV (pour analyse visuelle)
- Jupyter Notebook, Git/GitHub

Cahier des charges et extensions possibles :
- Étude des données de télémétrie et organisation en formats exploitables (CSV, JSON).
- Développement d’un pipeline de traitement (import, filtrage, nettoyage, visualisation).
- Mise en place d’une base d’algorithmes de fusion et de reconstruction (Kalman,
complémentaire, etc.).
- Extensions possibles :
- Détection et suivi de personnes/objets,
- Visualisation en temps réel,
- Export et partage des datasets (base commune de vols),
- Comparaison entre plusieurs vols ou plusieurs drones.

Environnement de développement
● Jupyter Notebook, Visual Studio Code, PyCharm
● Git / GitHub
● Matplotlib 3D, Plotly, ou Blender (optionnel pour animation)
Livrables attendus
● Scripts Python pour l’acquisition, le filtrage et l’analyse de la télémétrie sur Github.
● Visualisations claires (courbes, cartes 2D/3D, animations).
● Implémentation d’au moins un algorithme de fusion de données.
● Démonstration vidéo d’un cas d’usage (ex. reconstruction 3D d’un vol, suivi de
trajectoire).
● Poster scientifique/technique présentant la démarche, les résultats et les
perspectives.
Critères d’évaluation
● Qualité de l’acquisition et de l’organisation des données.
● Pertinence des visualisations et analyses proposées.
● Originalité et efficacité des algorithmes de fusion de données développés.
● Clarté et rigueur du code fourni (lisibilité, modularité, commentaires).
● Qualité du poster (concision, design, rigueur scientifique).
● Autonomie et prise d’initiatives.
