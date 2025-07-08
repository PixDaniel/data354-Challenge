import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import JsonOutputParser

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY non trouvée dans le fichier .env")

# --- Modèles et Parsers ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=GEMINI_API_KEY, temperature=0.5)
json_parser = JsonOutputParser()

# --- Prompt pour la génération de Q/R ---
qa_generation_prompt = ChatPromptTemplate.from_template(
    """
    Vous êtes un expert juriste spécialisé dans le droit du travail ivoirien.
    Votre tâche est de générer un ensemble de paires de questions-réponses précises à partir du morceau de texte suivant, extrait du Code du Travail.
    
    Pour chaque question, la réponse doit être une citation directe ou un résumé très proche du texte fourni. C'est notre "vérité terrain" (ground truth).
    Générez au moins 2 paires de questions-réponses.
    
    Structurez votre sortie en JSON avec une clé "qa_pairs" contenant une liste de dictionnaires. Chaque dictionnaire doit avoir les clés "question" et "ground_truth" (la réponse).

    Exemple de sortie :
    {{
      "qa_pairs": [
        {{
          "question": "Quelle est la durée maximale de la période d'essai pour un travailleur journalier ?",
          "ground_truth": "La période d'essai pour un travailleur journalier ou à l'heure ne peut excéder la première journée de travail."
        }},
        {{
          "question": "Comment le contrat de travail est-il défini ?",
          "ground_truth": "Le contrat de travail est une convention par laquelle une personne, le travailleur, s'engage à mettre son activité professionnelle sous la direction et l'autorité d'une autre personne, l'employeur, qui s'oblige à lui payer en contrepartie une rémunération."
        }}
      ]
    }}

    Texte à analyser :
    ---
    {chunk_text}
    ---

    Sortie JSON :
    """
)

# --- Chaîne LangChain ---
qa_chain = qa_generation_prompt | llm | json_parser

def create_evaluation_dataset(pdf_path: str, output_path: str, num_chunks_to_process: int = 20):
    """
    Génère un jeu de données d'évaluation (questions/réponses) à partir d'un PDF.
    """
    print("Chargement et découpage du PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=256)
    chunks = text_splitter.split_documents(documents)
    print(f"{len(chunks)} chunks créés. Traitement de {num_chunks_to_process} chunks pour créer le jeu de test.")
    
    all_qa_pairs = []
    
    # On ne traite qu'un sous-ensemble pour ne pas faire trop d'appels API
    chunks_to_process = chunks[:num_chunks_to_process]
    
    total_chunks = len(chunks_to_process)
    for i, chunk in enumerate(chunks_to_process):
        print(f"--- Génération Q/R pour le chunk {i+1}/{total_chunks} ---")
        try:
            generated_data = qa_chain.invoke({"chunk_text": chunk.page_content})
            qa_pairs = generated_data.get("qa_pairs", [])
            if qa_pairs:
                print(f"  > {len(qa_pairs)} paires Q/R générées.")
                all_qa_pairs.extend(qa_pairs)
            else:
                print("  > Aucune paire Q/R générée pour ce chunk.")
        except Exception as e:
            print(f"Erreur lors de la génération pour le chunk {i+1}: {e}")
            continue
            
    # Sauvegarde dans un fichier CSV
    df = pd.DataFrame(all_qa_pairs)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nJeu de données d'évaluation créé et sauvegardé à l'emplacement : {output_path}")
    print(f"Nombre total de paires Q/R générées : {len(df)}")

if __name__ == '__main__':
    pdf_file_path = "src/data/Code du Travail Ivoirien 2021.pdf"
    output_csv_path = "src/evaluation/evaluation_dataset.csv"
    if os.path.exists(pdf_file_path):
        create_evaluation_dataset(pdf_file_path, output_csv_path)
    else:
        print(f"Erreur : Le fichier {pdf_file_path} n'a pas été trouvé.") 