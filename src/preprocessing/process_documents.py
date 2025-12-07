import os
import json
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

def process_pdfs():
    # Setup paths
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDFs
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    if not pdf_files:
        print("No PDF files found in data/raw/")
        return
    
    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    all_chunks = []
    
    # Process each PDF
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Create chunks
            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
                
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
    
    # Save results
    output_file = output_dir / "processed_documents.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Successfully processed {len(all_chunks)} chunks from {len(pdf_files)} files")
    print(f"📁 Output saved to: {output_file}")

if __name__ == "__main__":
    process_pdfs()
