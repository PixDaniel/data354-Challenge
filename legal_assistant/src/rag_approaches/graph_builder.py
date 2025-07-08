import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.graphs import Neo4jGraph

# --- Chargement des variables d'environnement ---
load_dotenv()

# Clés API et identifiants Neo4j
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([GEMINI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("Veuillez configurer GEMINI_API_KEY, NEO4J_URI, NEO4J_USERNAME, et NEO4J_PASSWORD dans votre fichier .env")

# --- Initialisation du graphe Neo4j ---
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# --- Initialisation du LLM et du Parser ---
llm = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GEMINI_API_KEY, temperature=0)
json_parser = JsonOutputParser()

# --- Prompt pour l'extraction de connaissances ---
# Ce prompt guide le LLM pour qu'il extraie les entités et relations sous forme de JSON
graph_extraction_prompt = ChatPromptTemplate.from_template(
    """
    Vous êtes un expert en extraction de connaissances pour la construction de graphes.
    Votre tâche est d'analyser le texte suivant, extrait du Code du Travail Ivoirien, et d'en extraire les entités et les relations pertinentes.

    Instructions :
    1.  **Identifiez les entités clés** : ce peuvent être des concepts juridiques (ex: "Licenciement", "Contrat de travail"), des acteurs (ex: "Employeur", "Salarié"), ou des articles de loi.
    2.  **Identifiez les relations** qui lient ces entités. Utilisez des verbes d'action clairs et standardisés (ex: "REGIT", "A_DROIT_A", "EST_SOUMIS_A", "DEFINIT").
    3.  **Structurez votre sortie en JSON** avec deux clés : "nodes" et "relationships".
        *   Pour "nodes", chaque élément doit avoir un "id" (le nom de l'entité, normalisé) et un "label" (sa catégorie, ex: 'Concept', 'Acteur', 'Article').
        *   Pour "relationships", chaque élément doit avoir un "source" (l'id du nœud de départ), une "target" (l'id du nœud d'arrivée) et un "type" (le nom de la relation en majuscules).

    Exemple de sortie :
    {{
        "nodes": [
            {{"id": "Contrat de travail", "label": "Concept"}},
            {{"id": "Employeur", "label": "Acteur"}}
        ],
        "relationships": [
            {{"source": "Employeur", "target": "Contrat de travail", "type": "PROPOSE"}}
        ]
    }}

    Texte à analyser :
    ---
    {chunk_text}
    ---

    Sortie JSON :
    """
)

# --- Chaîne LangChain pour l'extraction ---
extraction_chain = graph_extraction_prompt | llm | json_parser

def process_and_ingest_pdf(pdf_path: str):
    """
    Charge un PDF, le découpe, extrait les connaissances et les ingère dans Neo4j.
    """
    print(f"Chargement du document : {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    chunks = text_splitter.split_documents(documents)
    print(f"Document découpé en {len(chunks)} chunks.")

    # Nettoyage initial du graphe pour éviter les doublons lors des exécutions successives
    graph.query("MATCH (n) DETACH DELETE n")
    print("Graphe Neo4j existant nettoyé.")

    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        print(f"--- Traitement du chunk {i+1}/{total_chunks} ---")
        try:
            # Extraction des données structurées
            extracted_data = extraction_chain.invoke({"chunk_text": chunk.page_content})
            
            nodes = extracted_data.get("nodes", [])
            relationships = extracted_data.get("relationships", [])

            if not nodes and not relationships:
                print("Aucune donnée extraite pour ce chunk.")
                continue

            print(f"  > Nœuds extraits : {len(nodes)}")
            print(f"  > Relations extraites : {len(relationships)}")
            
            # Ingestion dans Neo4j
            for node in nodes:
                graph.query(
                    "MERGE (n:`{label}` {{id: $id}})".format(label=node['label']),
                    params=node
                )
            
            for rel in relationships:
                graph.query(
                    """
                    MATCH (a {{id: $source}}), (b {{id: $target}})
                    MERGE (a)-[r:`{type}`]->(b)
                    """.format(type=rel['type']),
                    params=rel
                )

        except Exception as e:
            print(f"Erreur lors du traitement du chunk {i+1}: {e}")
            continue
    
    print("\n--- Processus d'ingestion terminé ! ---")
    print("Le graphe de connaissances a été construit dans Neo4j.")

if __name__ == '__main__':
    pdf_file_path = "src/data/Code du Travail Ivoirien 2021.pdf"
    if os.path.exists(pdf_file_path):
        process_and_ingest_pdf(pdf_file_path)
    else:
        print(f"Erreur : Le fichier {pdf_file_path} n'a pas été trouvé.")
        print("Veuillez le placer dans le dossier 'src/data' avant de lancer le script.") 