import os
import json
import pandas as pd
from datasets import load_dataset
from langchain.docstore.document import Document
from src.rag_baseline import build_baseline_rag, baseline_answer
from src.rag_with_fsm import fsm_rag_answer
import google.generativeai as genai
from tqdm import tqdm

# === KONFIGURASI ===
SAMPLE_SIZE = 50  # Ubah ke 100 atau lebih untuk hasil yang lebih valid
DATA_DIR = "./data"
INDEX_DIR = "./index/chroma_db"
OUTPUT_FILE = "experiment_results.csv"

# Set API Key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("EROR: GEMINI_API_KEY tidak ditemukan di environment variables.")
    exit(1)

def main():
    print(f"--- Memulai Eksperimen (n={SAMPLE_SIZE}) ---")

    # 1. Load Dataset
    print("Loading dataset SQuAD v2...")
    dataset = load_dataset('squad_v2', split='validation', cache_dir=os.path.join(DATA_DIR, "squad_v2"))

    # Filter dataset yang hanya punya jawaban (untuk testing ground truth)
    valid_data = [item for item in dataset if item['answers']['text']][:SAMPLE_SIZE]

    # 2. Persiapkan Dokumen & Retriever
    print("Membangun Vector Index...")
    documents = []
    for item in valid_data:
        doc = Document(
            page_content=item['context'],
            metadata={
                'question': item['question'],
                'answer': item['answers']['text'][0]
            }
        )
        documents.append(doc)

    retriever = build_baseline_rag(documents)

    # 3. Jalankan Eksperimen
    results = []

    print("Running Pipeline...")
    for i, item in enumerate(tqdm(valid_data)):
        query = item['question']
        ground_truth = item['answers']['text'][0]

        # --- Baseline RAG ---
        try:
            b_res = baseline_answer(query, retriever, api_key=api_key, return_context=True)
            b_answer = b_res['answer']
            b_status = "success"
        except Exception as e:
            b_answer = str(e)
            b_status = "error"

        # --- RAG + FSM ---
        try:
            f_res = fsm_rag_answer(query, retriever, api_key=api_key)
            f_answer = f_res['answer']
            f_status = f_res['status']
            f_states = "|".join(f_res['states_passed'])
        except Exception as e:
            f_answer = str(e)
            f_status = "system_error"
            f_states = "failed"

        # Simpan hasil per baris
        results.append({
            "id": i,
            "question": query,
            "ground_truth": ground_truth,
            "baseline_answer": b_answer,
            "baseline_status": b_status,
            "fsm_answer": f_answer,
            "fsm_status": f_status,
            "fsm_states_passed": f_states
        })

    # 4. Simpan ke CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Eksperimen Selesai! Hasil disimpan ke: {OUTPUT_FILE}")

    # 5. Summary Sederhana
    print("\n--- RINGKASAN ---")
    print(f"Total Sampel: {len(df)}")
    print(f"FSM Berhasil (Success): {len(df[df['fsm_status'] == 'success'])}")
    print(f"FSM Menangkap Error: {len(df[df['fsm_status'] == 'error'])}")

    # Hitung jika jawaban baseline kosong/error tapi FSM berhasil menangkapnya
    hallucination_detected = len(df[(df['baseline_status'] == 'success') & (df['fsm_status'] == 'error')])
    print(f"Potensi Hallucination Ditangkap FSM: {hallucination_detected}")

if __name__ == "__main__":
    main()
