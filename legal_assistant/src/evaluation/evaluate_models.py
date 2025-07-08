import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.metrics.critique import harmfulness
from datasets import Dataset
from langchain_core.documents import Document

# Import our RAG models
from src.rag_approaches.baseline_rag import BaselineRAG
from src.rag_approaches.crag import CorrectiveRAG
from src.rag_approaches.graph_rag import GraphRAG
from src.rag_approaches.agentic_rag import AgenticRAG

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY non trouvée dans le fichier .env")

class RAGEvaluator:
    """
    Classe pour évaluer et comparer différents modèles RAG.
    """
    def __init__(self, evaluation_dataset_path="src/evaluation/evaluation_dataset.csv"):
        """
        Initialise l'évaluateur avec le jeu de données d'évaluation.
        """
        print("Initialisation de l'évaluateur...")
        self.df = pd.read_csv(evaluation_dataset_path)
        print(f"Jeu de données chargé avec {len(self.df)} questions.")
        
        # Initialisation des modèles
        print("Initialisation des modèles RAG...")
        self.models = {
            "Baseline RAG": BaselineRAG(),
            "Corrective RAG": CorrectiveRAG(),
            "Graph RAG": GraphRAG(),
            "Agentic RAG": AgenticRAG()
        }
        
        # Métriques d'évaluation
        self.metrics = {
            "Faithfulness": faithfulness,
            "Answer Relevancy": answer_relevancy,
            "Context Precision": context_precision,
            "Context Recall": context_recall,
            "Harmfulness": harmfulness
        }
        
        # Résultats
        self.results = {}
    
    def evaluate_model(self, model_name, num_samples=10):
        """
        Évalue un modèle spécifique sur un sous-ensemble du jeu de données.
        """
        print(f"\n--- Évaluation du modèle: {model_name} ---")
        model = self.models[model_name]
        
        # Sélection d'un sous-ensemble aléatoire pour l'évaluation
        sample_df = self.df.sample(min(num_samples, len(self.df)), random_state=42)
        
        answers = []
        contexts = []
        questions = []
        ground_truths = []
        
        # Génération des réponses pour chaque question
        for i, row in enumerate(sample_df.itertuples()):
            print(f"Question {i+1}/{len(sample_df)}: {row.question}")
            try:
                # Pour les modèles qui retournent des contextes (documents)
                if hasattr(model, 'retriever'):
                    retrieved_docs = model.retriever.invoke(row.question)
                    context = [doc.page_content for doc in retrieved_docs]
                else:
                    # Pour les modèles qui n'exposent pas directement le retriever
                    context = ["Contexte non disponible pour ce modèle"]
                
                # Génération de la réponse
                answer = model.ask(row.question)
                
                answers.append(answer)
                contexts.append(context)
                questions.append(row.question)
                ground_truths.append(row.ground_truth)
                
                print(f"  > Réponse générée: {answer[:100]}...")
            except Exception as e:
                print(f"  > Erreur lors de la génération de la réponse: {e}")
        
        # Conversion en format Dataset pour RAGAs
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        })
        
        # Calcul des métriques
        results = {}
        for metric_name, metric_func in self.metrics.items():
            try:
                print(f"Calcul de la métrique: {metric_name}")
                score = metric_func.score(eval_dataset)
                results[metric_name] = score
                print(f"  > Score {metric_name}: {score}")
            except Exception as e:
                print(f"  > Erreur lors du calcul de la métrique {metric_name}: {e}")
                results[metric_name] = np.nan
        
        self.results[model_name] = results
        return results
    
    def evaluate_all_models(self, num_samples=10):
        """
        Évalue tous les modèles et compare leurs performances.
        """
        for model_name in self.models.keys():
            self.evaluate_model(model_name, num_samples)
        
        return self.get_comparison_table()
    
    def get_comparison_table(self):
        """
        Génère un tableau de comparaison des performances des modèles.
        """
        if not self.results:
            return "Aucune évaluation n'a été effectuée."
        
        # Création d'un DataFrame pour la comparaison
        comparison_df = pd.DataFrame(self.results).T
        
        # Calcul d'un score global (moyenne des métriques)
        comparison_df["Score Global"] = comparison_df.mean(axis=1)
        
        # Tri par score global
        comparison_df = comparison_df.sort_values("Score Global", ascending=False)
        
        return comparison_df

def main():
    print("=== ÉVALUATION COMPARATIVE DES MODÈLES RAG ===")
    
    # Vérification de l'existence du jeu de données d'évaluation
    eval_dataset_path = "src/evaluation/evaluation_dataset.csv"
    if not os.path.exists(eval_dataset_path):
        print(f"Erreur: Le fichier {eval_dataset_path} n'existe pas.")
        print("Veuillez d'abord exécuter le script create_test_set.py.")
        return
    
    # Initialisation et exécution de l'évaluateur
    evaluator = RAGEvaluator(eval_dataset_path)
    
    # Pour accélérer l'évaluation, on peut réduire le nombre d'échantillons
    comparison = evaluator.evaluate_all_models(num_samples=5)
    
    print("\n=== RÉSULTATS DE L'ÉVALUATION ===")
    print(comparison)
    
    # Sauvegarde des résultats
    comparison.to_csv("src/evaluation/evaluation_results.csv")
    print("\nRésultats sauvegardés dans src/evaluation/evaluation_results.csv")
    
    # Identification du meilleur modèle
    best_model = comparison.index[0]
    print(f"\nLe meilleur modèle est: {best_model}")
    print(f"Score global: {comparison.loc[best_model, 'Score Global']:.4f}")

if __name__ == "__main__":
    main() 