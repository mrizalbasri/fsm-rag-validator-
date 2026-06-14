# FSM-based Pipeline Validation for RAG Systems
## Improving Reliability of AI-generated Responses

**Jenis:** Paper Jurnal / Konferensi  
**Bidang:** Formal Language & Automata + Artificial Intelligence  
**Tools:** Python, LangChain, FSM (transitions), RAGAS, Gemini API  

---

## 1. Latar Belakang (Background)

Retrieval-Augmented Generation (RAG) adalah teknik AI yang menggabungkan:
- **Retrieval** — mencari dokumen relevan dari knowledge base
- **Augmentation** — menggabungkan dokumen dengan pertanyaan user
- **Generation** — LLM menjawab berdasarkan dokumen tersebut

**Masalah utama RAG saat ini:**
- AI sering *hallucinate* (menjawab percaya diri tapi salah)
- Tidak ada mekanisme formal yang mengontrol setiap tahap pipeline
- Tidak ada validasi apakah dokumen yang di-retrieve benar-benar relevan

**Solusi yang diusulkan:**
Menggunakan **Finite State Machine (FSM)** dari teori Formal Language & Automata sebagai lapisan validasi di setiap tahap pipeline RAG.

---

## 2. Kontribusi Paper

1. Merancang arsitektur FSM untuk validasi pipeline RAG
2. Mendefinisikan state, transisi, dan kondisi validasi di setiap tahap
3. Mengukur peningkatan reliabilitas jawaban AI (faithfulness, hallucination rate)
4. Membandingkan RAG biasa vs RAG + FSM validation

---

## 3. Arsitektur Sistem

```
USER QUERY
    |
    v
[STATE 1: Query Validation]
- Cek apakah query tidak kosong
- Cek panjang query wajar
- Cek query bukan injeksi/berbahaya
    |
    v (valid) atau ERROR (invalid)
    |
[STATE 2: Retrieval Validation]  
- Cek apakah dokumen ditemukan
- Cek similarity score >= threshold
- Cek jumlah dokumen cukup
    |
    v (valid) atau RETRY (tidak relevan)
    |
[STATE 3: Context Validation]
- Cek konteks tidak kosong
- Cek konteks relevan dengan query
- Cek panjang konteks dalam batas
    |
    v (valid) atau REJECT
    |
[STATE 4: Generation]
- LLM generate jawaban
    |
    v
[STATE 5: Output Validation]
- Cek jawaban tidak kosong
- Cek jawaban sesuai format
- Cek jawaban tidak berisi hallucination pattern
    |
    v
FINAL ANSWER ke User
```

---

## 4. Dataset

### Dataset 1: SQuAD v2.0
- **Sumber:** HuggingFace (`squad_v2`)
- **Kegunaan:** Uji deteksi hallucination (ada pertanyaan yang sengaja tidak bisa dijawab)
- **Jumlah yang dipakai:** 500 sampel dari validation set

```python
from datasets import load_dataset
squad = load_dataset("squad_v2", split="validation[:500]")
```

### Dataset 2: MS MARCO
- **Sumber:** HuggingFace (`ms_marco`, `v2.1`)
- **Kegunaan:** Uji relevansi retrieval dokumen
- **Jumlah yang dipakai:** 500 sampel

```python
from datasets import load_dataset
marco = load_dataset("ms_marco", "v2.1", split="validation[:500]")
```

---

## 5. Tech Stack & Tools

| Komponen | Library | Fungsi |
|---|---|---|
| RAG Pipeline | LangChain / LlamaIndex | Bangun alur RAG |
| FSM | `transitions` (Python) | Logika state machine |
| Vector DB | ChromaDB / FAISS | Simpan & cari dokumen |
| LLM | Gemini API (Google) | Generate jawaban |
| Evaluasi | RAGAS | Ukur kualitas RAG |
| Dataset | HuggingFace `datasets` | Load SQuAD & MS MARCO |

### Install semua dependencies:
```bash
pip install langchain langchain-community
pip install transitions
pip install chromadb
pip install ragas
pip install google-generativeai
pip install datasets
pip install sentence-transformers
```

---

## 6. Langkah-langkah Implementasi

### Langkah 1: Setup Environment
```python
# install_requirements.py
import subprocess
packages = [
    "langchain", "langchain-community",
    "transitions", "chromadb",
    "ragas", "google-generativeai",
    "datasets", "sentence-transformers"
]
for pkg in packages:
    subprocess.run(["pip", "install", pkg])
```

### Langkah 2: Buat FSM Validator
```python
# fsm_validator.py
from transitions import Machine

class RAGStateMachine:
    states = [
        'idle',
        'query_validation',
        'retrieval',
        'retrieval_validation', 
        'context_validation',
        'generation',
        'output_validation',
        'success',
        'error'
    ]
    
    def __init__(self):
        self.machine = Machine(
            model=self,
            states=RAGStateMachine.states,
            initial='idle'
        )
        # Define transitions
        self.machine.add_transition('start', 'idle', 'query_validation')
        self.machine.add_transition('query_valid', 'query_validation', 'retrieval')
        self.machine.add_transition('query_invalid', 'query_validation', 'error')
        self.machine.add_transition('retrieve', 'retrieval', 'retrieval_validation')
        self.machine.add_transition('docs_valid', 'retrieval_validation', 'context_validation')
        self.machine.add_transition('docs_invalid', 'retrieval_validation', 'error')
        self.machine.add_transition('context_valid', 'context_validation', 'generation')
        self.machine.add_transition('context_invalid', 'context_validation', 'error')
        self.machine.add_transition('generate', 'generation', 'output_validation')
        self.machine.add_transition('output_valid', 'output_validation', 'success')
        self.machine.add_transition('output_invalid', 'output_validation', 'error')
        self.machine.add_transition('reset', 'error', 'idle')
    
    def validate_query(self, query: str) -> bool:
        if not query or len(query.strip()) < 3:
            return False
        if len(query) > 500:
            return False
        return True
    
    def validate_retrieval(self, docs, threshold=0.5) -> bool:
        if not docs or len(docs) == 0:
            return False
        # Cek similarity score
        for doc in docs:
            if hasattr(doc, 'metadata') and doc.metadata.get('score', 0) < threshold:
                return False
        return True
    
    def validate_output(self, answer: str) -> bool:
        if not answer or len(answer.strip()) < 5:
            return False
        # Cek pola hallucination sederhana
        hallucination_patterns = [
            "i don't know", "i cannot", "as an ai"
        ]
        for pattern in hallucination_patterns:
            if pattern in answer.lower():
                return False
        return True
```

### Langkah 3: Bangun RAG Pipeline Biasa (Baseline)
```python
# rag_baseline.py
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

def build_baseline_rag(documents):
    # Split dokumen
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    # Buat vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return retriever

def baseline_answer(query, retriever):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])
    
    # Panggil Gemini API
    genai.configure(api_key="YOUR_GEMINI_API_KEY")
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    
    return response.text
```

### Langkah 4: Bangun RAG + FSM (Proposed)
```python
# rag_with_fsm.py
from fsm_validator import RAGStateMachine
from rag_baseline import build_baseline_rag, baseline_answer

def fsm_rag_answer(query, retriever):
    fsm = RAGStateMachine()
    result = {"answer": None, "status": None, "states_passed": []}
    
    # State 1: Query Validation
    fsm.start()
    result["states_passed"].append("query_validation")
    if fsm.validate_query(query):
        fsm.query_valid()
    else:
        fsm.query_invalid()
        result["status"] = "error"
        result["answer"] = "Query tidak valid."
        return result
    
    # State 2: Retrieval
    fsm.retrieve()
    docs = retriever.get_relevant_documents(query)
    result["states_passed"].append("retrieval_validation")
    
    if fsm.validate_retrieval(docs):
        fsm.docs_valid()
    else:
        fsm.docs_invalid()
        result["status"] = "error"
        result["answer"] = "Dokumen relevan tidak ditemukan."
        return result
    
    # State 3: Context Validation
    context = "\n".join([d.page_content for d in docs])
    result["states_passed"].append("context_validation")
    if context and len(context) > 10:
        fsm.context_valid()
    else:
        fsm.context_invalid()
        result["status"] = "error"
        result["answer"] = "Konteks tidak valid."
        return result
    
    # State 4: Generation
    fsm.generate()
    answer = baseline_answer(query, retriever)
    result["states_passed"].append("generation")
    
    # State 5: Output Validation
    result["states_passed"].append("output_validation")
    if fsm.validate_output(answer):
        fsm.output_valid()
        result["status"] = "success"
        result["answer"] = answer
    else:
        fsm.output_invalid()
        result["status"] = "error"
        result["answer"] = "Output tidak valid."
    
    return result
```

### Langkah 5: Load Dataset & Jalankan Eksperimen
```python
# experiment.py
from datasets import load_dataset
from rag_baseline import build_baseline_rag, baseline_answer
from rag_with_fsm import fsm_rag_answer
from langchain.docstore.document import Document

# Load dataset SQuAD v2
dataset = load_dataset("squad_v2", split="validation[:500]")

# Konversi ke dokumen LangChain
documents = []
for item in dataset:
    if item['answers']['text']:  # Hanya yang punya jawaban
        doc = Document(
            page_content=item['context'],
            metadata={"question": item['question'], "answer": item['answers']['text'][0]}
        )
        documents.append(doc)

# Build retriever
retriever = build_baseline_rag(documents)

# Jalankan eksperimen
baseline_results = []
fsm_results = []

for item in dataset[:100]:  # Test 100 sampel dulu
    query = item['question']
    
    # Baseline
    baseline_ans = baseline_answer(query, retriever)
    baseline_results.append(baseline_ans)
    
    # FSM
    fsm_ans = fsm_rag_answer(query, retriever)
    fsm_results.append(fsm_ans)

print(f"Baseline errors: {sum(1 for r in baseline_results if not r)}")
print(f"FSM errors caught: {sum(1 for r in fsm_results if r['status'] == 'error')}")
```

### Langkah 6: Evaluasi dengan RAGAS
```python
# evaluate.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset

def evaluate_rag(questions, answers, contexts, ground_truths):
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )
    return result

# Bandingkan baseline vs FSM
print("=== Baseline RAG ===")
baseline_score = evaluate_rag(
    questions, baseline_answers, baseline_contexts, ground_truths
)
print(baseline_score)

print("\n=== RAG + FSM Validation ===")
fsm_score = evaluate_rag(
    questions, fsm_answers, fsm_contexts, ground_truths
)
print(fsm_score)
```

---

## 7. Struktur Paper (Outline)

```
1. ABSTRACT (150-250 kata)

2. INTRODUCTION
   - Latar belakang RAG & masalahnya
   - Motivasi penggunaan FSM
   - Kontribusi paper
   
3. RELATED WORK
   - RAG systems (Lewis et al., 2020)
   - Hallucination in LLM
   - Formal verification dengan automata
   - FSM dalam AI pipeline
   
4. METHODOLOGY
   4.1 Arsitektur FSM yang diusulkan
   4.2 Definisi state & transisi
   4.3 Kondisi validasi tiap state
   4.4 Integrasi dengan pipeline RAG
   
5. EXPERIMENTAL SETUP
   5.1 Dataset (SQuAD v2, MS MARCO)
   5.2 Implementation details
   5.3 Evaluation metrics (Faithfulness, Answer Relevancy, Hallucination Rate)
   
6. RESULTS & DISCUSSION
   6.1 Perbandingan Baseline vs Proposed
   6.2 Analisis per-state mana yang paling berpengaruh
   6.3 Error analysis
   
7. CONCLUSION & FUTURE WORK

8. REFERENCES
```

---

## 8. Metrik Evaluasi

| Metrik | Deskripsi | Target |
|---|---|---|
| **Faithfulness** | Seberapa setia jawaban dengan dokumen sumber | Naik dari baseline |
| **Answer Relevancy** | Seberapa relevan jawaban dengan pertanyaan | Naik dari baseline |
| **Context Recall** | Seberapa banyak info penting terambil | Naik dari baseline |
| **Hallucination Rate** | % jawaban yang tidak didukung dokumen | Turun dari baseline |
| **Error Detection Rate** | % error yang berhasil ditangkap FSM | Metrik baru paper ini |

---

## 9. Timeline Pengerjaan (Estimasi)

| Minggu | Kegiatan |
|---|---|
| Minggu 1 | Setup environment, install library, pahami dataset |
| Minggu 2 | Implementasi RAG baseline |
| Minggu 3 | Rancang & implementasi FSM validator |
| Minggu 4 | Integrasi FSM ke RAG pipeline |
| Minggu 5 | Jalankan eksperimen & evaluasi RAGAS |
| Minggu 6 | Analisis hasil, visualisasi grafik |
| Minggu 7-8 | Penulisan paper |
| Minggu 9 | Revisi & submission |

---

## 10. Referensi Kunci yang Perlu Dibaca

1. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
2. Ji, Z. et al. (2023). *Survey of Hallucination in Natural Language Generation*. ACM Computing Surveys.
3. Hopcroft, J.E., Motwani, R., Ullman, J.D. *Introduction to Automata Theory, Languages, and Computation*.
4. Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv.
5. Guu, K. et al. (2020). *REALM: Retrieval-Augmented Language Model Pre-Training*. ICML.

---

*Dokumen ini dibuat sebagai panduan project paper: FSM-based Pipeline Validation for RAG Systems*
