# 🏥 MediMind — Medical AI Assistant

> **Full-Stack GenAI Capstone Project** | QLoRA Fine-tuning · DPO Alignment · RAG · LangChain · Gradio Deployment

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-HuggingFace_Spaces-orange)](https://huggingface.co/spaces/raj5113/medimind)
[![Model](https://img.shields.io/badge/🤗_Model-Phi--2_2.7B-blue)](https://huggingface.co/microsoft/phi-2)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)

---

## 🎯 Project Overview

MediMind is a production-grade medical AI assistant that replicates the complete modern GenAI engineering pipeline used by companies like Google (MedPaLM), Microsoft (BioGPT), and medical AI startups like Nabla and Abridge.

Built on Microsoft's Phi-2 (2.7B parameters), this project demonstrates every major skill in the GenAI engineering stack — from domain-specific fine-tuning to safe deployment — entirely without relying on third-party API keys.

**🔗 Try the live demo:** [https://huggingface.co/spaces/raj5113/medimind](https://huggingface.co/spaces/raj5113/medimind)

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  MediMind Pipeline                       │
│                                                          │
│  ┌─────────────┐     ┌──────────────┐                   │
│  │   FAISS     │     │  Conversation│                   │
│  │  Retriever  │     │    Memory    │                   │
│  │ (10 clinical│     │  (LangChain) │                   │
│  │ guidelines) │     │              │                   │
│  └──────┬──────┘     └──────┬───────┘                   │
│         │                   │                           │
│         └──────────┬────────┘                           │
│                    │                                     │
│                    ▼                                     │
│          ┌─────────────────┐                            │
│          │  MediMind LLM   │                            │
│          │  Phi-2 2.7B     │                            │
│          │  + QLoRA SFT    │                            │
│          │  + DPO Aligned  │                            │
│          └────────┬────────┘                            │
│                   │                                      │
└───────────────────┼──────────────────────────────────────┘
                    │
                    ▼
         Cited Medical Response
```

---

## 🚀 Five-Phase Training Pipeline

### Phase 1 — QLoRA Fine-Tuning (Supervised Fine-Tuning)

**What:** Fine-tuned Microsoft Phi-2 on 2,000 USMLE (United States Medical Licensing Exam) questions and answers.

**How:**
- 4-bit NF4 quantization via BitsAndBytes reduces model memory from 10.8GB to ~3GB
- LoRA adapters injected into attention layers (q_proj, k_proj, v_proj, dense)
- Rank=16, 10.4M trainable parameters out of 2.7B total (0.68%)
- Trained on Kaggle dual T4 GPU (31GB total VRAM)

**Real world parallel:** Google used the same technique to build MedPaLM 2 from Gemini. OpenAI uses LoRA-based adapters for enterprise fine-tuning. The technique is identical — only the scale differs.

```
Training config:
  Epochs:              3
  Effective batch:     16
  Learning rate:       2e-4 (cosine schedule)
  Optimizer:           paged_adamw_32bit
  Precision:           bfloat16
  Final loss:          ~0.93
  Training time:       ~90 minutes on 2x T4
```

**Key engineering insight discovered:** The dataset used MCQ-style outputs ("E: Nitrofurantoin") which trained the model to select answer letters rather than generate clinical sentences. Fixed by reformatting outputs to begin with "According to clinical guidelines..." — a data quality issue that mirrors real production challenges at Google and Anthropic.

---

### Phase 2 — DPO Alignment (RLHF)

**What:** Applied Direct Preference Optimization to teach the model safe medical behavior — when to recognize emergencies, when to direct to crisis services, and how to correct dangerous misconceptions.

**How:**
- 6 curated preference pairs (chosen vs rejected responses)
- Each pair targets a specific safety failure mode
- DPO loss: 0.693 → 0.626 (model learned to prefer safe responses)
- Training time: under 1 minute

**Preference pairs target these failure modes:**
| Scenario | Rejected Response | Chosen Response |
|---|---|---|
| Cardiac emergency | "Chest pain has many causes, try resting" | "Call 911 immediately" |
| Mental health crisis | "Exercise and think positive" | "Call 988 Lifeline now" |
| Medication overdose | "Monitor at home" | "Call Poison Control 1-800-222-1222" |
| Drug in pregnancy | Wrong mechanism | Correct fetal cartilage damage explanation |
| Medication myth | "Your friend has a valid point" | Correct safety information |
| Pediatric poisoning | "Wait and see if symptoms appear" | "Go to ER immediately" |

**Real world parallel:** Anthropic uses Constitutional AI (a DPO-like approach) to align Claude. OpenAI used human rater feedback for ChatGPT. Our 6 carefully chosen pairs are a scaled-down version of the same process.

---

### Phase 3 — RAG Pipeline (Retrieval Augmented Generation)

**What:** Built a semantic search engine over 10 clinical guidelines so the model answers from evidence rather than from potentially outdated or hallucinated memory.

**How:**
- 10 clinical guideline documents (ADA, AHA, WHO, USPSTF, KDIGO, etc.)
- Split into 38 overlapping chunks (400 chars, 50 char overlap)
- Embedded using all-MiniLM-L6-v2 (384-dimensional vectors)
- Indexed in FAISS (Facebook AI Similarity Search)
- Retrieval accuracy: correct document found for every test query

**Knowledge sources indexed:**
- ADA Standards of Care 2024 (Type 2 Diabetes)
- AHA/ACC Heart Failure Guidelines 2022
- JNC 8 / ESH Hypertension Guidelines 2023
- Antibiotic Safety in Pregnancy Guidelines
- Emergency Medicine Cardiac Chest Pain Protocol
- WHO Depression and Mental Health Guidelines 2023
- Pediatric Toxicology Iron Poisoning Protocol
- Beta-Blocker Overdose and Medication Safety
- USPSTF Cancer Screening Recommendations 2023
- KDIGO Chronic Kidney Disease Guidelines 2022

**Retrieval performance:**
```
Query                              Retrieved Document       Similarity
"ciprofloxacin dangerous pregnancy" Antibiotic Safety        0.636
"chest pain"                        Cardiac Protocol         0.515
"metformin mechanism"               ADA Standards            0.457
"iron poisoning child"              Pediatric Toxicology     0.847
```

**Real world parallel:** This is exactly how Bing Chat works (web pages instead of guidelines), how Claude.ai processes uploaded PDFs, and how Epic Systems builds EHR search functionality.

---

### Phase 4 — LangChain Orchestration

**What:** Connected the fine-tuned model, RAG pipeline, and conversation memory into a single coherent chain.

**Components:**
- `ConversationalRetrievalChain` — orchestrates retrieval + generation
- `ConversationBufferWindowMemory` — maintains last 5 conversation turns
- Custom `BaseRetriever` — wraps FAISS index in LangChain-compatible interface (built to bypass langchain-community 0.4.1 API breaking changes)
- `PromptTemplate` — formats system context, retrieved guidelines, history, and question

**Memory demonstration:**
```
Turn 1: "My patient is 65 years old with heart failure and diabetes."
Turn 2: "What HbA1c target should I aim for?"
         → Model uses memory to apply the <8% guideline for elderly
           patients with comorbidities rather than the default <7%
```

---

### Phase 5 — Gradio Deployment

**What:** Deployed as a publicly accessible web application on HuggingFace Spaces (free tier).

**Features:**
- Chat interface with conversation history
- 8 example medical questions
- Source citations on every response
- Medical disclaimer
- CPU-based inference (no GPU required for deployment)

**Live:** [https://huggingface.co/spaces/raj5113/medimind](https://huggingface.co/spaces/raj5113/medimind)

---

## 📊 Honest Results Analysis

Most AI portfolios show only clean results. This project documents real engineering tradeoffs.

### What worked well
- FAISS retrieval: correct document retrieved for every test query
- LangChain memory: conversation context preserved across turns
- DPO alignment: emergency question ("I have chest pain") correctly directed to "Call 911"
- Full pipeline deployed end-to-end without any API keys

### What has limitations and why
- **MCQ format bleed-through:** Phi-2 trained on USMLE MCQ data learned to prefix answers with "According to clinical guidelines, the answer is A:" — a data quality issue that mirrors real production challenges
- **Generation quality bounded by model size:** Phi-2 at 2.7B parameters vs GPT-4 at ~1.7T parameters. The retrieval and orchestration architecture is production-grade; the generation layer is the bottleneck
- **DPO with 6 pairs:** Production systems use thousands of preference pairs. Our 6 demonstrate the technique; scale would require data distillation from a larger model

### What this tells you about production AI
```
The gap between our system and GPT-4 based medical AI:
  Architecture:    Identical (RAG + LangChain + fine-tuning + alignment)
  Retrieval:       Identical technique, different scale
  Alignment:       Identical technique (DPO), different data volume
  Generation:      2.7B vs ~1700B parameters — this is the real gap
  Cost:            $0 vs $10M+ in compute
```

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|---|---|---|
| Base Model | Microsoft Phi-2 | 2.7B params |
| Fine-tuning | QLoRA (bitsandbytes + peft) | peft 0.13.2 |
| Alignment | DPO (trl) | trl 0.29.1 |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 |
| Vector Search | FAISS | faiss-cpu 1.13.2 |
| Orchestration | LangChain | 0.3.0 |
| UI Framework | Gradio | 5.50.0 |
| Training Hardware | Kaggle dual T4 GPU | 2x 15.64GB |
| Deployment | HuggingFace Spaces | CPU Basic (free) |
| Framework | PyTorch | 2.9.0 |

---

## 📁 Repository Structure

```
MediMind/
│
├── 1_finetuning/
│   └── finetune_phi2_lora.ipynb     ← QLoRA training on Kaggle
│
├── 2_rlhf/
│   └── dpo_alignment.ipynb           ← DPO alignment
│
├── 3_rag/
│   └── rag_pipeline.ipynb            ← FAISS vector store + retrieval
│
├── 4_langchain/
│   └── medimind_chain.ipynb          ← LangChain orchestration
│
├── 5_deployment/
│   ├── app.py                        ← Gradio web application
│   └── requirements.txt              ← HuggingFace Spaces dependencies
│
└── README.md
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/medimind
cd medimind
pip install -r 5_deployment/requirements.txt
python 5_deployment/app.py
```

App runs at `http://localhost:7860`

---

## 🔄 Swap the LLM (1 line)

LangChain makes the generation backend fully swappable:

```python
# Current — local fine-tuned Phi-2 (no API key needed):
llm = HuggingFacePipeline(pipeline=gen_pipeline)

# GPT-4 (OpenAI):
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Claude (Anthropic):
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-opus-4-6", temperature=0.1)

# Gemini (Google):
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
```

Everything else — retrieval, memory, chain — stays identical.

---

## 📚 Clinical Knowledge Sources

All medical content is derived from publicly available clinical guidelines:
- American Diabetes Association (ADA) Standards of Care 2024
- American Heart Association (AHA) Heart Failure Guidelines 2022
- JNC 8 / European Society of Hypertension 2023
- World Health Organization (WHO) Mental Health Guidelines 2023
- US Preventive Services Task Force (USPSTF) 2023
- Kidney Disease Improving Global Outcomes (KDIGO) 2022
- Surviving Sepsis Campaign 2021
- FDA Drug Safety Communications

---

## ⚠️ Medical Disclaimer

MediMind is an educational project demonstrating GenAI engineering techniques. It is **NOT intended for clinical use** and should **NOT replace professional medical advice**. Always consult a licensed healthcare provider for diagnosis and treatment decisions.

---

## 🏆 Skills Demonstrated

- ✅ LLM Fine-tuning with QLoRA (4-bit quantization + LoRA adapters)
- ✅ RLHF / Alignment with DPO (preference data, safety behavior)
- ✅ RAG Architecture (FAISS, semantic embeddings, document chunking)
- ✅ LangChain (Chains, Memory, Retrievers, Prompt Templates)
- ✅ Multi-GPU Training (Kaggle dual T4, DDP, bf16 precision)
- ✅ Production Deployment (Gradio, HuggingFace Spaces)
- ✅ Version Debugging (trl/peft/transformers compatibility matrix)
- ✅ Honest ML Engineering (documented real tradeoffs and limitations)

---

## 📧 About

Built as a capstone project after completing the **GenAI with LLMs** course on Coursera.

Goal: hands-on mastery of the complete modern GenAI pipeline — not just calling APIs, but building, training, aligning, and deploying a real model from scratch.

---

⭐ **If this helped you, please give it a star!**
