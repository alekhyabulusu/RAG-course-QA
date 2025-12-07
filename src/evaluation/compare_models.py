"""Compare Multiple LLMs on Course Q&A - Including FREE Options"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv


sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.retrieval.rag_system import SimpleRAG

load_dotenv()

class ModelComparison:
    """Comparing different LLMs using the RAG system"""
    
    def __init__(self):
        print("🚀 Initializing Model Comparison Framework...")
        self.rag = SimpleRAG()
        self.results = []
        print("✅ Ready to compare models!")
    
    def test_retrieval_baseline(self, question: str) -> dict:
        """Baseline: Just retrieval without LLM"""
        start = time.time()
        chunks = self.rag.search(question, k=5)
        response_time = time.time() - start
        
        sources = list(set([c['source'] for c in chunks]))
        
        return {
            "model": "Retrieval-Only-Baseline",
            "question": question,
            "answer": f"Retrieved {len(chunks)} chunks from: {', '.join(sources[:3])}",
            "num_chunks": len(chunks),
            "response_time": response_time,
            "cost": 0.00,
            "status": "success"
        }
    
    
    
    def test_huggingface_model(self, question: str) -> dict:
        """Test FREE Hugging Face model (LLaMA alternative)"""
        try:
            from transformers import pipeline
            
            print("   Loading HF model (first time downloads)...")
            
            # Get context from RAG
            chunks = self.rag.search(question, k=3)
            context = " ".join([c['content'][:150] for c in chunks[:2]])
            
            # Use a question-answering model
            qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad"
            )
            
            start = time.time()
            result = qa_pipeline(question=question, context=context)
            response_time = time.time() - start
            
            return {
                "model": "huggingface-distilbert",
                "question": question,
                "answer": result['answer'],
                "num_chunks": len(chunks),
                "response_time": response_time,
                "cost": 0.00,  # FREE!
                "status": "success"
            }
            
        except Exception as e:
            return {
                "model": "huggingface-distilbert",
                "question": question,
                "answer": f"Error: {str(e)[:100]}",
                "num_chunks": 0,
                "response_time": 0,
                "cost": 0,
                "status": "error"
            }
    
    def run_full_comparison(self, questions: list, models: list = None) -> pd.DataFrame:
        """Run comparison across all specified models"""
        
        if models is None:
            models = [
                "retrieval_baseline",
                "huggingface",  
            ]
        
        print("\n🔬 COMPREHENSIVE MODEL COMPARISON")
        print("="*60)
        print(f"Testing {len(models)} models on {len(questions)} questions")
        print("="*60)
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Question {i}: {question[:60]}...")
            
            for model in models:
                print(f"   Testing {model}...", end="")
                
                if model == "retrieval_baseline":
                    result = self.test_retrieval_baseline(question)
                elif model == "huggingface":
                    result = self.test_huggingface_model(question)
                else:
                    continue
                
                self.results.append(result)
                print(f" {result['status']} ({result['response_time']:.2f}s)")
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = Path("data/processed") / f"model_comparison_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        return df
    
    def analyze_results(self, df: pd.DataFrame):
        """Analyze and display comparison results"""
        
        print("\n" + "="*60)
        print("📊 COMPARISON ANALYSIS")
        print("="*60)
        
        # Group by model
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            # Calculate metrics
            success_rate = (model_df['status'] == 'success').mean() * 100
            avg_time = model_df[model_df['status'] == 'success']['response_time'].mean()
            total_cost = model_df['cost'].sum()
            avg_chunks = model_df['num_chunks'].mean()
            
            print(f"\n📈 {model}:")
            print(f"   Success Rate: {success_rate:.1f}%")
            if success_rate > 0:
                print(f"   Avg Response Time: {avg_time:.2f}s")
            print(f"   Avg Chunks Used: {avg_chunks:.1f}")
            print(f"   Total Cost: ${total_cost:.4f}")
            
            # Show sample answer if successful
            success_answers = model_df[model_df['status'] == 'success']['answer']
            if not success_answers.empty:
                sample = success_answers.iloc[0][:150]
                print(f"   Sample Answer: {sample}...")
        
        # Best model by success rate
        success_by_model = df.groupby('model')['status'].apply(
            lambda x: (x == 'success').mean()
        ).sort_values(ascending=False)
        
        print("\n🏆 Models Ranked by Success Rate:")
        for i, (model, rate) in enumerate(success_by_model.items(), 1):
            print(f"   {i}. {model}: {rate*100:.1f}%")

def main():
    """Run the complete comparison study"""
    
    # Questions from your LLM course
    test_questions = [
        "What is the attention mechanism in transformers?",
        "Explain the difference between encoder and decoder models",
        "What is fine-tuning in language models?",
        "How does tokenization work in LLMs?",
        "What are embeddings and why are they important?",
    ]
    
    # Initialize comparison framework
    comparison = ModelComparison()
    
    # Define models to test
    available_models = ["retrieval_baseline", "huggingface"]  
    
    
    # Run comparison
    df = comparison.run_full_comparison(
        questions=test_questions[:3],  # Start with 3 questions
        models=available_models
    )
    
    # Analyze results
    comparison.analyze_results(df)
    
    print("\n✅ Comparison study complete!")
    print("\n📋 Your project deliverables are ready:")
    print("1. ✅ Processed documents in data/processed/")
    print("2. ✅ Vector embeddings in data/chroma/")
    print("3. ✅ Model comparison results in CSV")
    print("4. ✅ Performance metrics for your report")

if __name__ == "__main__":
    # Install transformers if needed for HuggingFace
    os.system("pip install transformers -q")
    main()