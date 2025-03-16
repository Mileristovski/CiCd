# Machine Learning Workbench

## Description
Ce projet sert de Workbench pour expérimenter et rechercher différents algorithmes de Machine Learning. Il permet de tester, comparer et analyser différentes approches en utilisant des fichiers `.csv` comme source de données. L'application est exposée sous forme d'API pour faciliter l'interaction avec les modèles entraînés.

## Technologies utilisées
- Python
- Pandas (pour la manipulation des données)
- Scikit-learn (pour l'entraînement des modèles)
- Flask (pour l'API)
- Docker (pour la conteneurisation de l'application)

## Utilisation
1. Cloner le dépôt :
   ```bash
   git clone https://github.com/Mileristovski/MachineLearning-Workbench.git
   cd MachineLearning-Workbench
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Exécuter l'API :
   ```bash
   python app.py
   ```

## Exécution avec Docker
1. Construire l'image Docker :
   ```bash
   docker build -t ml-workbench .
   ```
2. Exécuter le conteneur :
   ```bash
   docker run --rm -p 5000:5000 ml-workbench
   ```

## Objectifs du projet
- Tester et comparer différents algorithmes de Machine Learning
- Expérimenter le prétraitement des données et l'ingénierie des features
- Visualiser les performances des modèles
- Servir de base pour de futurs projets ML

