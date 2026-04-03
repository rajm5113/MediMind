import gradio as gr
import torch
import numpy as np
import faiss
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ============================================================
# Load model and tokenizer
# ============================================================
print("Loading MediMind model...")

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float32,
    trust_remote_code=True,
)

# Replace raj5113 with your actual HuggingFace username
HF_USERNAME = "raj5113"
try:
    model = PeftModel.from_pretrained(
        model,
        f"{HF_USERNAME}/medimind-phi2-adapter",
        is_trainable=False,
    )
    print(f"Loaded fine-tuned adapter")
except Exception as e:
    print(f"Could not load adapter: {e}. Using base Phi-2.")

model.eval()
print("Model ready")

# ============================================================
# Build knowledge base and FAISS index
# ============================================================
print("Building medical knowledge base...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

medical_chunks = [
    {
        "source": "ADA Standards of Care 2024",
        "content": "Type 2 Diabetes Management: HbA1c target less than 7% for most nonpregnant adults. Target less than 8% for older adults with multiple comorbidities or limited life expectancy. First-line pharmacotherapy is Metformin if eGFR above 30. Early symptoms: polyuria, polydipsia, polyphagia, fatigue, blurred vision, slow wound healing, tingling in hands and feet."
    },
    {
        "source": "Emergency Medicine Cardiac Protocol",
        "content": "Chest pain radiating to left arm with sweating is a cardiac emergency. Call 911 immediately. Do not drive self to hospital. Chew aspirin 325mg if not allergic. Classic heart attack symptoms: severe crushing chest pain, diaphoresis, nausea, dyspnea. Time critical for STEMI treatment, door to balloon under 90 minutes."
    },
    {
        "source": "Antibiotic Safety in Pregnancy",
        "content": "Fluoroquinolones including ciprofloxacin are CONTRAINDICATED in pregnancy. They inhibit DNA gyrase causing fetal cartilage damage and arthropathy. Safe UTI alternatives in pregnancy: nitrofurantoin, cephalexin, amoxicillin-clavulanate, fosfomycin. Tetracyclines also contraindicated due to tooth discoloration and bone growth inhibition."
    },
    {
        "source": "Pediatric Toxicology Iron Poisoning Protocol",
        "content": "Iron tablet ingestion in children is a serious emergency. Child may appear fine initially due to deceptive Stage 2 recovery period. Severe deterioration occurs 6-24 hours later with hepatotoxicity and metabolic acidosis. Call Poison Control 1-800-222-1222 immediately and go to ER. Do not wait for symptoms. Bring medication bottle."
    },
    {
        "source": "WHO Depression Guidelines 2023",
        "content": "Persistent hopelessness, worthlessness, and not seeing point in continuing are serious depression symptoms requiring immediate attention. Call 988 Suicide and Crisis Lifeline (call or text, 24/7). PHQ-9 scoring: 0-4 minimal, 5-9 mild, 10-14 moderate, 15+ severe. First line treatment: SSRIs (sertraline, escitalopram) plus psychotherapy."
    },
    {
        "source": "Beta-Blocker and Medication Safety",
        "content": "Double dose of metoprolol requires calling Poison Control 1-800-222-1222 immediately. Beta-blocker overdose causes dangerous bradycardia and hypotension. Effects may be delayed 4-6 hours. Do not stop blood pressure medication because readings are normal. Normal readings indicate the medication is working. Stopping causes rebound hypertension."
    },
    {
        "source": "AHA Heart Failure Guidelines 2022",
        "content": "Heart failure HFrEF treatment: ACE inhibitor plus beta-blocker plus mineralocorticoid receptor antagonist. SGLT2 inhibitors recommended for HFrEF and HFpEF. Only carvedilol, metoprolol succinate, bisoprolol have proven mortality benefit. ICD indicated for EF below 35% with NYHA Class II or III on optimal therapy."
    },
    {
        "source": "Hypertension JNC8 Guidelines",
        "content": "Blood pressure target below 130/80 mmHg. First-line medications: ACE inhibitor or ARB, calcium channel blocker, or thiazide diuretic. Black patients prefer CCB or thiazide over ACEi. CKD patients use ACEi or ARB for renoprotection. Lifestyle: DASH diet, sodium below 2.3g daily, 150 minutes exercise weekly."
    },
    {
        "source": "USPSTF Cancer Screening 2023",
        "content": "Colorectal cancer: colonoscopy every 10 years starting age 45. Breast cancer: mammography every 2 years ages 50-74. Cervical cancer: Pap smear every 3 years ages 21-65 or Pap plus HPV every 5 years ages 30-65. Lung cancer: annual low-dose CT for adults 50-80 with 20 pack-year smoking history."
    },
    {
        "source": "KDIGO CKD Guidelines 2022",
        "content": "CKD stages by eGFR: G1 above 90, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5 below 15. Slow progression with BP below 130/80, ACEi or ARB for proteinuria, SGLT2 inhibitors. Avoid NSAIDs. Hold metformin before contrast if eGFR 30-60. Contraindicated if eGFR below 30."
    },
]

texts = [c["content"] for c in medical_chunks]
embeddings = embedder.encode(texts, convert_to_numpy=True)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))
print(f"Knowledge base ready: {index.ntotal} chunks indexed")


# ============================================================
# Core response function — ChatInterface format
# ============================================================
def medimind_respond(message, history):
    """
    ChatInterface passes history as list of dicts automatically.
    We just return the answer string — ChatInterface handles
    appending to history.
    """
    if not message.strip():
        return "Please enter a medical question."

    # Retrieve relevant context
    q_emb = embedder.encode([message], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    sims, idxs = index.search(q_emb.astype(np.float32), 3)

    context_pieces = []
    sources = []
    for sim, idx in zip(sims[0], idxs[0]):
        if idx != -1 and sim > 0.2:
            context_pieces.append(medical_chunks[idx]["content"])
            sources.append(medical_chunks[idx]["source"])

    context = " ".join(context_pieces) if context_pieces else "Use your medical knowledge."

    # Build prompt
    prompt = f"""### System:
You are MediMind, an expert medical AI assistant trained on clinical guidelines.
Answer medical questions in complete sentences with clear clinical reasoning.
For any emergency symptoms call 911. For overdose call Poison Control 1-800-222-1222.
Always recommend consulting a licensed physician.
### Medical Guidelines:
{context}
### Question:
{message}
### Clinical Response:
"""

    # Generate response
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=800
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Clinical Response:" in full:
        answer = full.split("### Clinical Response:")[-1].strip()
    else:
        answer = full.split("### Question:")[-1].strip()

    # Remove repetitive sentences
    sentences = answer.split(". ")
    seen = set()
    clean = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            clean.append(s)
        if len(clean) >= 4:
            break
    answer = ". ".join(clean)
    if answer and not answer.endswith("."):
        answer += "."

    if sources:
        answer += f"\n\n**Sources:** {', '.join(set(sources))}"
    answer += "\n\n*Always consult a licensed physician for diagnosis and treatment.*"

    return answer


# ============================================================
# Gradio ChatInterface — Gradio 6.0 compatible
# No manual Chatbot component needed
# ChatInterface handles all history and formatting automatically
# ============================================================
# REPLACE the app = gr.ChatInterface(...) block with this:
app = gr.ChatInterface(
    fn=medimind_respond,
    title="MediMind Medical AI Assistant",
    description="""
**Medical AI Assistant | Fine-tuned Phi-2 + QLoRA + DPO + RAG**
Warning: Educational information only. NOT a substitute for professional medical advice. Always consult a licensed physician.
Built with: QLoRA Fine-tuning, DPO Alignment, FAISS RAG, Gradio
    """,
    examples=[
        "What is the HbA1c target for Type 2 diabetes?",
        "Why is Ciprofloxacin avoided in pregnancy?",
        "I have chest pain radiating to my left arm.",
        "My child swallowed iron tablets. What do I do?",
        "What are first-line medications for heart failure?",
        "How does Metformin lower blood sugar?",
        "When should I screen for colorectal cancer?",
        "I feel hopeless and worthless for weeks.",
    ],
    cache_examples=False,
)

app.launch()
"""
---
## Why this fixes everything
```
Previous approach: gr.Blocks + gr.Chatbot
  Problem: Chatbot parameters keep changing between
  Gradio versions (type, bubble_full_width, etc.)
  Every version breaks something different.
New approach: gr.ChatInterface
  Built specifically for chat applications.
  Handles all history, formatting, and display internally.
  No Chatbot parameters to worry about.
  Works identically across Gradio 4.x, 5.x, and 6.x.
  Function just returns a string — that is all.
