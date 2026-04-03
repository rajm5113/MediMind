
import gradio as gr
import torch
import numpy as np
import faiss
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from langchain_huggingface import HuggingFaceEmbeddings

# ---- Load model ----
print("Loading MediMind...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float32,
    trust_remote_code=True,
)
# Load your adapter from HuggingFace Hub
# Replace YOUR_USERNAME with your HuggingFace username
model = PeftModel.from_pretrained(model, "raj5113/medimind-phi2-adapter")
model.eval()

# ---- Load embeddings and FAISS ----
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

medical_chunks = [
    {"source": "ADA Standards of Care 2024", "content": "Type 2 Diabetes Management: HbA1c target less than 7% for most nonpregnant adults. Target less than 8% for older adults with multiple comorbidities. First-line: Metformin. GLP-1 RA or SGLT2i for ASCVD or heart failure. Early symptoms: polyuria, polydipsia, fatigue, blurred vision, slow wound healing."},
    {"source": "Emergency Medicine Cardiac Chest Pain", "content": "Chest pain radiating to left arm with sweating is a cardiac emergency. Call 911 immediately. Do not drive self to hospital. Chew aspirin 325mg if not allergic. Time critical for heart attack treatment."},
    {"source": "Antibiotic Safety in Pregnancy", "content": "Ciprofloxacin and fluoroquinolones are CONTRAINDICATED in pregnancy. They inhibit DNA gyrase causing fetal cartilage damage. Safe UTI alternatives: nitrofurantoin, cephalexin, amoxicillin-clavulanate."},
    {"source": "Pediatric Toxicology Iron Poisoning", "content": "Iron tablet ingestion in children is a serious emergency. Child may appear fine initially (deceptive Stage 2). Deterioration occurs 6-24 hours later. Call Poison Control 1-800-222-1222 immediately. Go to ER now."},
    {"source": "WHO Depression Guidelines 2023", "content": "Persistent hopelessness and not seeing point in continuing are serious depression symptoms. Call 988 Suicide and Crisis Lifeline immediately. Effective treatments available including therapy and medication."},
    {"source": "Beta-Blocker Safety", "content": "Double dose of metoprolol requires calling Poison Control 1-800-222-1222 immediately. Beta-blocker overdose causes bradycardia and hypotension. Do not wait for symptoms. Do not stop blood pressure medication when readings are normal."},
    {"source": "AHA Heart Failure Guidelines", "content": "Heart failure HFrEF treatment: ACE inhibitor plus beta-blocker plus MRA. SGLT2 inhibitors recommended for HFrEF and HFpEF. Only carvedilol, metoprolol succinate, bisoprolol have mortality benefit."},
    {"source": "Hypertension Guidelines", "content": "Blood pressure target below 130/80. First-line: ACE inhibitor, CCB, or thiazide. Black patients prefer CCB or thiazide. CKD patients use ACEi or ARB. Do not stop medication when readings are normal."},
    {"source": "USPSTF Cancer Screening", "content": "Colorectal cancer: colonoscopy every 10 years from age 45. Breast: mammography every 2 years ages 50-74. Cervical: Pap every 3 years ages 21-65. Lung: annual CT for 50-80 year smokers with 20 pack-year history."},
    {"source": "CKD Guidelines KDIGO 2022", "content": "CKD stages by eGFR: G1 above 90, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5 below 15. SGLT2 inhibitors slow CKD progression. Avoid NSAIDs. Hold metformin before contrast if eGFR 30-60."},
]

# Build FAISS index from chunks
texts = [c["content"] for c in medical_chunks]
chunk_embeddings = embeddings_model.embed_documents(texts)
chunk_array = np.array(chunk_embeddings, dtype=np.float32)
faiss.normalize_L2(chunk_array)
index = faiss.IndexFlatIP(chunk_array.shape[1])
index.add(chunk_array)

def respond(message, history):
    if not message.strip():
        return history, ""

    # Retrieve context
    q_emb = embeddings_model.embed_query(message)
    q_vec = np.array([q_emb], dtype=np.float32)
    faiss.normalize_L2(q_vec)
    sims, idxs = index.search(q_vec, 3)

    context_pieces = []
    sources = []
    for sim, idx in zip(sims[0], idxs[0]):
        if idx != -1 and sim > 0.2:
            context_pieces.append(medical_chunks[idx]["content"])
            sources.append(medical_chunks[idx]["source"])
    context = " ".join(context_pieces) if context_pieces else "Use your medical knowledge."

    prompt = f"""### System:
You are MediMind, a medical AI assistant. Answer based on guidelines.
For emergencies direct to 911 or Poison Control 1-800-222-1222.

### Guidelines:
{context}

### Question:
{message}

### Clinical Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full.split("### Clinical Response:")[-1].strip()

    # Deduplicate sentences
    sentences = answer.split(". ")
    seen = set()
    clean = []
    for s in sentences:
        if s.strip() and s.strip() not in seen:
            seen.add(s.strip())
            clean.append(s.strip())
        if len(clean) >= 4:
            break
    answer = ". ".join(clean) + "."
    if sources:
        answer += f"\n\n**Sources:** {', '.join(set(sources))}"
    answer += "\n\n*Consult a licensed physician for actual medical decisions.*"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, ""

with gr.Blocks(title="MediMind", theme=gr.themes.Soft()) as app:
    gr.HTML("<h1 style=\'text-align:center;color:#1565C0;\'>MediMind Medical AI</h1>")
    gr.HTML("<p style=\'text-align:center;background:#FFF3E0;padding:8px;border-radius:4px;\'>Educational only. Not a substitute for professional medical advice.</p>")
    chatbot = gr.Chatbot(height=400, type="messages")
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask a medical question...", scale=4)
        gr.Button("Ask", variant="primary").click(respond, [msg, chatbot], [chatbot, msg])
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    gr.Examples(
        ["What is the HbA1c target for diabetes?",
         "I have chest pain radiating to my left arm.",
         "Why is ciprofloxacin avoided in pregnancy?",
         "My child swallowed iron tablets."],
        inputs=msg
    )

app.launch()
