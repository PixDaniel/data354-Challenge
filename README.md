# Assistant Juridique - Droit du Travail Ivoirien

Un assistant conversationnel spécialisé dans le droit du travail ivoirien, utilisant plusieurs approches avancées de génération augmentée par recherche (RAG).

![Assistant Juridique](https://img.shields.io/badge/Assistant-Juridique-blue)
![Droit du Travail](https://img.shields.io/badge/Droit-Travail-green)
![Côte d'Ivoire](https://img.shields.io/badge/C%C3%B4te-d'Ivoire-orange)

## 📋 Table des Matières

- [Présentation](#présentation)
- [Approches RAG Implémentées](#approches-rag-implémentées)
- [Architecture du Projet](#architecture-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Évaluation des Modèles](#évaluation-des-modèles)
- [Interface Web](#interface-web)
- [Licence](#licence)

## 🎯 Présentation

Ce projet implémente un agent conversationnel spécialisé dans le droit du travail ivoirien. Il utilise le document "Code du Travail Ivoirien 2021" comme base de connaissances et exploite plusieurs approches de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles aux questions juridiques.

L'objectif est de comparer différentes approches RAG et d'identifier celle qui offre les meilleures performances pour ce domaine spécifique.

## 🚀 Approches RAG Implémentées

Le projet implémente quatre approches RAG distinctes :

1. **Baseline RAG** : Approche classique de RAG qui utilise une recherche vectorielle simple pour trouver les passages pertinents.

2. **Corrective RAG (CRAG)** : Version améliorée qui vérifie la pertinence des documents récupérés avant de générer une réponse, avec possibilité de reformuler la requête si nécessaire.

3. **GraphRAG** : Utilise un graphe de connaissances (Neo4j) pour représenter les concepts juridiques et leurs relations, permettant une compréhension plus profonde du domaine.

4. **AgenticRAG** : Agent autonome capable de décomposer des questions complexes, d'utiliser différents outils spécialisés et de synthétiser les informations pour produire une réponse complète.

## 🏗️ Architecture du Projet

```
Law_IA/
├── app.py                      # Application Streamlit
├── requirements.txt            # Dépendances Python
├── .env                        # Variables d'environnement (non versionné)
├── .gitignore                  # Fichiers à ignorer par Git
├── src/
│   ├── data/
│   │   ├── Code du Travail Ivoirien 2021.pdf   # Document source
│   │   ├── faiss_index/                        # Base vectorielle FAISS
│   │   └── data_loader.py                      # Script de chargement et vectorisation
│   ├── rag_approaches/
│   │   ├── baseline_rag.py                     # Implémentation du Baseline RAG
│   │   ├── crag.py                             # Implémentation du Corrective RAG
│   │   ├── graph_builder.py                    # Construction du graphe de connaissances
│   │   ├── graph_rag.py                        # Implémentation du GraphRAG
│   │   └── agentic_rag.py                      # Implémentation de l'AgenticRAG
│   └── evaluation/
│       ├── create_test_set.py                  # Génération du jeu de données d'évaluation
│       ├── evaluate_models.py                  # Évaluation comparative des modèles
│       ├── evaluation_dataset.csv              # Jeu de données d'évaluation
│       └── evaluation_results.csv              # Résultats de l'évaluation
```

## 📥 Installation

1. **Clonez le dépôt** :

   ```bash
   git clone git@github.com:PixDaniel/data354-Challenge.git
   cd legal_assistant
   ```

2. **Activer un environement virtuelle** :

   ```bash
   conda activate
   ```

3. **Installez les dépendances** :

   ```bash
   pip install -r requirements.txt
   ```

4. **Configurez les variables d'environnement** :
   Créez un fichier `.env` à la racine du projet avec les informations suivantes :

   ```
   GEMINI_API_KEY="VOTRE_CLE_API_GEMINI"
   NEO4J_URI="URI_DE_VOTRE_BASE_NEO4J"
   NEO4J_USERNAME="neo4j"
   NEO4J_PASSWORD="VOTRE_MOT_DE_PASSE"
   ```

5. **Placez le document source** :
   Placez le fichier "Code du Travail Ivoirien 2021.pdf" dans le dossier `src/data/`.

## 🔧 Utilisation

### Préparation des Données

1. **Création de la base vectorielle** :

   ```bash
   python src/data/data_loader.py
   ```

2. **Construction du graphe de connaissances** (pour GraphRAG) :
   ```bash
   python src/rag_approaches/graph_builder.py
   ```

### Test des Modèles Individuels

Vous pouvez tester chaque modèle individuellement :

```bash
# Baseline RAG
python src/rag_approaches/baseline_rag.py

# Corrective RAG
python src/rag_approaches/crag.py

# GraphRAG
python src/rag_approaches/graph_rag.py

# AgenticRAG
python src/rag_approaches/agentic_rag.py
```

## 📊 Évaluation des Modèles

1. **Génération du jeu de données d'évaluation** :

   ```bash
   python src/evaluation/create_test_set.py
   ```

2. **Lancement de l'évaluation comparative** :
   ```bash
   python src/evaluation/evaluate_models.py
   ```

Les résultats de l'évaluation seront sauvegardés dans `src/evaluation/evaluation_results.csv`.

## 🌐 Interface Web

Lancez l'interface web Streamlit :

```bash
streamlit run app.py
```

L'interface sera accessible à l'adresse `http://localhost:8501` et vous permettra de :

- Poser des questions sur le droit du travail ivoirien
- Choisir le modèle RAG à utiliser
- Visualiser les sources utilisées pour générer les réponses

## 📄 Licence

Ce projet est sous licence [MIT](LICENSE).
