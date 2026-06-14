# Penelitian RAG + FSM Validation

Project ini bertujuan untuk menguji peningkatan reliabilitas sistem RAG menggunakan Finite State Machine (FSM) sebagai lapisan validasi.

## Struktur Project
- `src/`: Logika inti (FSM, RAG Baseline, RAG+FSM).
- `notebooks/`: Eksperimentasi dan visualisasi hasil.
- `data/`: Penyimpanan dataset (SQuAD, MS MARCO).
- `index/`: Penyimpanan Vector Database (ChromaDB).
- `main_experiment.py`: Script untuk menjalankan eksperimen otomatis.

## Cara Menjalankan
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set API Key:**
   Pastikan environment variable `GEMINI_API_KEY` sudah diset.
3. **Jalankan Eksperimen:**
   ```bash
   python main_experiment.py
   ```
4. **Analisis Hasil:**
   Buka `notebooks/02_Visualisasi_Hasil.ipynb` untuk melihat grafik dan statistik.

## Metodologi
Sistem membandingkan RAG standar (Baseline) dengan RAG yang divalidasi oleh 5 state FSM:
1. `Query Validation`
2. `Retrieval Validation`
3. `Context Validation`
4. `Generation`
5. `Output Validation`
