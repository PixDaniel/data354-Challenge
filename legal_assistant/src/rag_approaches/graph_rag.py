import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate

# --- Chargement des variables d'environnement ---
load_dotenv()

# Clés API et identifiants Neo4j
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([GEMINI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError("Veuillez configurer GEMINI_API_KEY, NEO4J_URI, NEO4J_USERNAME, et NEO4J_PASSWORD dans votre fichier .env")

# --- Template de Prompt pour la conversion Text-to-Cypher ---
# Ce prompt guide le LLM pour qu'il génère des requêtes Cypher à partir de la question de l'utilisateur
CYPHER_GENERATION_TEMPLATE = """
Vous êtes un expert en traduction de langage naturel vers des requêtes Cypher pour Neo4j.
Votre tâche est de convertir la question de l'utilisateur en une requête Cypher qui peut être exécutée sur une base de données dont le schéma est fourni ci-dessous.
Répondez uniquement avec la requête Cypher, sans aucune autre explication ou texte.

Schéma du graphe :
{schema}

Question de l'utilisateur :
{question}

Requête Cypher :
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

class GraphRAG:
    """
    Implémentation d'un agent conversationnel basé sur un graphe de connaissances (GraphRAG).
    """
    def __init__(self):
        # Initialisation de la connexion au graphe
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        # Rafraîchit le schéma pour s'assurer qu'il est à jour
        self.graph.refresh_schema()

        # Initialisation du LLM
        self.llm = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GEMINI_API_KEY, temperature=0)
        
        # Création de la chaîne de QA sur le graphe
        self.chain = GraphCypherQAChain.from_llm(
            cypher_llm=self.llm,
            qa_llm=self.llm,
            graph=self.graph,
            verbose=True, # Affiche les requêtes générées et les résultats
            cypher_prompt=CYPHER_GENERATION_PROMPT,
        )

    def ask(self, question: str) -> str:
        """
        Pose une question à l'agent GraphRAG.
        """
        print("---Génération et exécution de la requête Cypher---")
        result = self.chain.invoke({"query": question})
        return result['result']

if __name__ == '__main__':
    # Attendre que le script graph_builder.py ait fini de s'exécuter
    print("Initialisation de l'agent GraphRAG...")
    
    try:
        graph_rag_agent = GraphRAG()

        # Testez avec des questions qui exploitent les relations du graphe
        question_1 = "Quelles sont les obligations de l'employeur en matière de contrat de travail ?"
        print(f"\nQuestion : {question_1}")
        response_1 = graph_rag_agent.ask(question_1)
        print(f"\nRéponse : {response_1}")

        question_2 = "Quels sont les droits d'un salarié en cas de licenciement ?"
        print(f"\nQuestion : {question_2}")
        response_2 = graph_rag_agent.ask(question_2)
        print(f"\nRéponse : {response_2}")
        
    except Exception as e:
        print(f"\nUne erreur est survenue. Assurez-vous que le graphe a bien été construit par 'graph_builder.py'.")
        print(f"Erreur technique : {e}") 