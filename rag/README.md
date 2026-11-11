# Lab: Build a RAG System – from Retrieval to Agentic RAG

**Course:** Agentic AI, FAST–NUCES Karachi
**Instructor:** Dr. Jawwad Ahmed Shamsi
**Student:** Farrukh Imran (23K-0776)

A step-by-step Retrieval-Augmented Generation (RAG) system built on a small sample university handbook. It starts with chunking and retrieval and ends with an agent that decides for itself when to search. It runs fully on Google Colab (free T4 GPU) with no API key.

## Models

| Role | Model |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` |
| Keyword search | `rank_bm25` (BM25Okapi) |

## What was implemented

| Task | Topic | What I built |
|---|---|---|
| 0 | Setup | Loaded the embedding model and LLM, `embed()` and `generate()` helpers |
| 1 | Chunking | `chunk_words(text, size, overlap)` – sliding word window |
| 2 | Retrieval | `retrieve()` – cosine similarity (dot product of normalised vectors), top-k |
| 3 | Evaluation | `recall_at_k()` and a chunk-size comparison (40 / 100 / 250 words) |
| 4 | Grounded answers | `build_prompt()` with numbered passages and citations, plus refusal when the answer is missing |
| 5 | Hybrid search | `rrf()` – Reciprocal Rank Fusion of dense + BM25 results |
| 6 | HyDE | `hyde_retrieve()` – search with an LLM-written hypothetical answer |
| 7 | Lost in the middle | Gold passage placed first / middle / last among 41 sentence passages |
| 8 | Agentic RAG (challenge) | `agent()` with a `SEARCH:` / `ANSWER:` loop and a Self-RAG-style `check_support()` |

## Results

### Chunking (Task 1)
- Handbook length: 718 words
- Chunks with `size=100, overlap=20`: **9**

### Retrieval (Task 2)
- "How many books can I borrow from the library?" → chunk 5 ranked first (score **0.530**)
- Out-of-vocabulary question (laptop charger in the exam) → low and close scores (0.366, 0.364, 0.321), so the retrieval was unsure

### Recall@k by chunk size (Task 3)

| Chunk size (words) | Number of chunks | Recall@1 | Recall@3 |
|---|---|---|---|
| 40 | 23 | 1.00 | 1.00 |
| 100 | 9 | 1.00 | 1.00 |
| 250 | 4 | 1.00 | 1.00 |

All six test questions were found at every chunk size. The handbook is small, so the sizes could not be told apart.

### Grounded answers (Task 4)
- "Can I take a reference book home?" → **"No. [1]"** (correct citation)
- Hostel fee for international students (not in the handbook) → **"I could not find this in the documents."**
- Without the refusal instruction in the system prompt, the model still did not invent a fee, but gave a long, unfocused answer. The system prompt matters.

### Hybrid search (Task 5)
Query `F-4015`:

| Method | Top-3 chunks | Evidence found |
|---|---|---|
| Dense | [4, 6, 3] | Yes |
| BM25 | [4, 0, 2] | Yes |
| Hybrid (RRF) | [4, 3, 2] | Yes |

Hybrid Recall@3 = **1.0** (same as dense). The gain is small here because the document has only 9 chunks.

### HyDE (Task 6)
Query: "Can I get my money back if I drop out?"
- Plain search: correct passage at rank **2**
- HyDE: correct passage at rank **2**
- The hypothetical passage was generic (it mentioned "financial aid") and had none of the handbook's refund details, so HyDE did not help.

### Lost in the middle (Task 7)

| Question | First | Middle | Last |
|---|---|---|---|
| Library fine per day | True | True | True |
| Medical leave days | True | True | True |
| Probation courses | True | True | True |

The "lost in the middle" pattern did **not** appear. With 3 easy questions on a small model, this is not strong evidence either way.

### Agentic RAG (Task 8)
Question: *If I withdraw in the first week, how much of my fee do I get back, and which form do I use?*

```
step 0: SEARCH: Withdrawal policy details
step 1: SEARCH: Withdrawal policy details
step 2: ANSWER: If you withdraw in the first week, you receive a 75% refund of your tuition fees. Use the "Withdrawal Form" to initiate the process.

SUPPORTED   If you withdraw in the first week, you receive a 75% refund of your tuition fees.
SUPPORTED   Use the "Withdrawal Form" to initiate the process.
```

Findings:
- The refund amount (75%) is correct.
- The form name is wrong. The handbook says **F-4015**, but the model made up "Withdrawal Form", and the support checker wrongly marked it as SUPPORTED.
- The agent repeated the same search query and gave no `[n]` citations.
- For "What is 2+2?", the agent still searched (and wrote an empty `SEARCH:` once) before answering 4, which wastes time and compute.
- The home-made `SEARCH:` / `ANSWER:` format is fragile, which is the motivation for function calling.

## Key takeaways
1. Chunk size trades off recall against the amount of text in the prompt, and bigger is not automatically better.
2. A clear "say you don't know" instruction in the system prompt makes RAG answers safer.
3. Hybrid search matters most on large collections with exact terms like IDs and form numbers.
4. Small models can invent details and a small-model checker can miss it. Verify important facts.

## How to run
1. Open `rag_lab_completed.ipynb` in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order. The first run downloads about 3 GB of models.

## Files
- `rag_lab_completed.ipynb` – completed notebook with outputs
- `Lab_Report_23K-0776.pdf` – lab report with answers to all questions
- `README.md` – this file
