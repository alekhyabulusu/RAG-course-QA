"""
Evaluate DistilBERT and Retrieval Baseline with  ROUGE and Cosine Similarity
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import Dict, List, Tuple
import os

# Install required packages if needed
os.system("pip install rouge-score nltk scikit-learn transformers sentence-transformers -q")

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class DualModelEvaluator:
    """Evaluate both DistilBERT and Retrieval Baseline"""
    
    def __init__(self):
        print("🚀 Initializing Dual Model Evaluator...")
        
        # Initialize components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Load vector store
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory="data/chroma",
            embedding_function=self.embeddings,
            collection_name="course_materials"
        )
        
        # Initialize DistilBERT
        print("Loading DistilBERT model...")
        self.qa_pipeline = pipeline("question-answering", 
                                   model="distilbert-base-cased-distilled-squad")
        
        print("✅ Evaluator ready!")
    
    def calculate_metrics(self, question: str, answer: str, reference: str = None) -> Dict:
        """Calculate all evaluation metrics"""
        
        metrics = {}
        
        # 1. Cosine Similarity (Query-Answer Relevance)
        q_emb = self.embedding_model.encode([question])
        a_emb = self.embedding_model.encode([answer])
        metrics['cosine_similarity'] = float(cosine_similarity(q_emb, a_emb)[0][0])
        
        if reference:
            ref_tokens = word_tokenize(reference.lower())
            ans_tokens = word_tokenize(answer.lower())
            
            #ROUGE Scores
            rouge_scores = self.rouge_scorer.score(reference, answer)
            metrics['rouge1_f1'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2_f1'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL_f1'] = rouge_scores['rougeL'].fmeasure
        else:
            metrics['rouge1_f1'] = None
            metrics['rouge2_f1'] = None
            metrics['rougeL_f1'] = None
        
        return metrics
    
    def evaluate_retrieval_baseline(self, question: str, k: int = 5) -> Tuple[str, Dict, float]:
        """Evaluate retrieval-only baseline"""
        
        start_time = time.time()
        
        # Retrieve relevant chunks
        docs = self.vectorstore.similarity_search(question, k=k)
        
        # Concatenate retrieved content as "answer"
        if docs:
            answer = " ".join([doc.page_content[:200] for doc in docs[:3]])
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
            answer_summary = f"Retrieved {len(docs)} chunks from: {', '.join(sources[:2])}"
        else:
            answer = "No relevant content found."
            answer_summary = answer
        
        response_time = time.time() - start_time
        
        return answer_summary, {"full_answer": answer[:500], "num_chunks": len(docs)}, response_time
    
    def evaluate_distilbert(self, question: str, k: int = 3) -> Tuple[str, Dict, float]:
        """Evaluate DistilBERT model"""
        
        start_time = time.time()
        
        # Retrieve context
        docs = self.vectorstore.similarity_search(question, k=k)
        
        if docs:
            # Combine context for QA
            context = " ".join([doc.page_content[:200] for doc in docs[:2]])
            
            # Get answer from DistilBERT
            try:
                result = self.qa_pipeline(question=question, context=context[:512])
                answer = result['answer']
                confidence = result['score']
                metadata = {"confidence": confidence, "num_chunks": len(docs)}
            except Exception as e:
                answer = f"Error: {str(e)[:100]}"
                metadata = {"error": True, "num_chunks": 0}
        else:
            answer = "No context found for question."
            metadata = {"no_context": True, "num_chunks": 0}
        
        response_time = time.time() - start_time
        
        return answer, metadata, response_time
    
    def evaluate_on_test_set(self) -> pd.DataFrame:
        """Run evaluation on test questions"""
        
        # Test questions with optional reference answers
        test_set = [
            {
                "question": "What is the attention mechanism in transformers?",
                "reference": "Attention mechanism allows models to focus on different parts of input when processing each element"
            },
            {
                "question": "How does tokenization work in LLMs?",
                "reference": "Tokenization splits text into smaller units that models can process"
            },
            {
                "question": "What is fine-tuning?",
                "reference": "Fine-tuning adapts a pre-trained model to specific tasks"
            },
            {
                "question": "Explain the difference between GPT and BERT",
                "reference": "GPT is autoregressive for generation, BERT is bidirectional for understanding"
            },
            {
                "question": "What are embeddings?",
                "reference": "Embeddings are vector representations of text capturing semantic meaning"
            }
        ]
        
        results = []
        
        print("\n" + "="*70)
        print("EVALUATING BOTH MODELS")
        print("="*70 + "\n")
        
        for i, test_case in enumerate(test_set, 1):
            question = test_case["question"]
            reference = test_case.get("reference", None)
            
            print(f"Question {i}: {question[:50]}...")
            
            # 1. Evaluate Retrieval Baseline
            print("  - Testing Retrieval Baseline...", end="")
            baseline_answer, baseline_meta, baseline_time = self.evaluate_retrieval_baseline(question)
            baseline_metrics = self.calculate_metrics(question, 
                                                     baseline_meta["full_answer"], 
                                                     reference)
            
            baseline_result = {
                "question": question[:100],
                "model": "Retrieval_Baseline",
                "answer": baseline_answer[:200],
                "response_time": baseline_time,
                "num_chunks": baseline_meta["num_chunks"],
                **baseline_metrics
            }
            results.append(baseline_result)
            print(f" ✓ (Cosine: {baseline_metrics['cosine_similarity']:.3f})")
            
            # 2. Evaluate DistilBERT
            print("  - Testing DistilBERT...", end="")
            distil_answer, distil_meta, distil_time = self.evaluate_distilbert(question)
            distil_metrics = self.calculate_metrics(question, distil_answer, reference)
            
            distil_result = {
                "question": question[:100],
                "model": "DistilBERT",
                "answer": distil_answer[:200],
                "response_time": distil_time,
                "num_chunks": distil_meta.get("num_chunks", 0),
                "confidence": distil_meta.get("confidence", None),
                **distil_metrics
            }
            results.append(distil_result)
            print(f" ✓ (Cosine: {distil_metrics['cosine_similarity']:.3f})")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"data/processed/model_evaluation_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        return df
    
    def generate_comparison_report(self, df: pd.DataFrame):
        """Generate detailed comparison report"""
        
        print("\n" + "="*70)
        print("COMPARATIVE EVALUATION REPORT")
        print("="*70 + "\n")
        
        # Separate by model
        baseline_df = df[df['model'] == 'Retrieval_Baseline']
        distil_df = df[df['model'] == 'DistilBERT']
        
        # Comparison table
        comparison = pd.DataFrame({
            'Metric': ['Avg Cosine Similarity', 'Avg ROUGE-L', 
                      'Avg Response Time', 'Avg Chunks Used'],
            'Retrieval_Baseline': [
                f"{baseline_df['cosine_similarity'].mean():.3f}",
                f"{baseline_df['rougeL_f1'].mean():.3f}" if baseline_df['rougeL_f1'].notna().any() else "N/A",
                f"{baseline_df['response_time'].mean():.3f}s",
                f"{baseline_df['num_chunks'].mean():.1f}"
            ],
            'DistilBERT': [
                f"{distil_df['cosine_similarity'].mean():.3f}",
                f"{distil_df['rougeL_f1'].mean():.3f}" if distil_df['rougeL_f1'].notna().any() else "N/A",
                f"{distil_df['response_time'].mean():.3f}s",
                f"{distil_df['num_chunks'].mean():.1f}"
            ]
        })
        
        print(comparison.to_string(index=False))
        
        # Performance Summary
        print("\n📊 PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        # Cosine similarity comparison
        baseline_cosine = baseline_df['cosine_similarity'].mean()
        distil_cosine = distil_df['cosine_similarity'].mean()
        
        if distil_cosine > baseline_cosine:
            print(f"✅ DistilBERT has better relevance (+{(distil_cosine-baseline_cosine):.3f})")
        else:
            print(f"✅ Retrieval Baseline has better relevance (+{(baseline_cosine-distil_cosine):.3f})")
        
        # Speed comparison
        baseline_speed = baseline_df['response_time'].mean()
        distil_speed = distil_df['response_time'].mean()
        
        if baseline_speed < distil_speed:
            print(f"✅ Retrieval Baseline is {distil_speed/baseline_speed:.1f}x faster")
        else:
            print(f"✅ DistilBERT is {baseline_speed/distil_speed:.1f}x faster")
        
        # Statistical summary
        print("\n📈 DETAILED METRICS:")
        print("-" * 40)
        print(df.groupby('model').agg({
            'cosine_similarity': ['mean', 'std', 'min', 'max'],
            'response_time': ['mean', 'std', 'min', 'max']
        }).round(3))

def main():
    """Run complete evaluation"""
    
    # Initialize evaluator
    evaluator = DualModelEvaluator()
    
    # Run evaluation
    df = evaluator.evaluate_on_test_set()
    
    # Generate report
    evaluator.generate_comparison_report(df)
    
    print("\n✅ Evaluation complete!")
    print("\n📊 Key Findings:")
    print("1. Both models evaluated on same questions")
    print("2. Metrics calculated: Cosine Similarity, ROUGE")
    print("3. Response times measured for performance comparison")
    print("4. Results saved to DataFrame for further analysis")
    
    return df

if __name__ == "__main__":
    df = main()
    
    # Display final DataFrame
    print("\n📋 COMPLETE RESULTS DATAFRAME:")
    print(df.to_string())
