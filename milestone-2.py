# Milestone 2 - Hugging Face Transformers & Datasets
# Simple student-style code

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

DATA_PATH = r"D:\smart-mcq-solver-challenge\train.csv"
OPTIONS = ["A", "B", "C", "D", "E"]

print("=" * 60)
print("MILESTONE 2 - HUGGING FACE ANSWERS")
print("=" * 60)


def map_at_k(true_answer, predictions, k=3):
    preds = predictions[:k]
    if true_answer in preds:
        return 1.0 / (preds.index(true_answer) + 1)
    return 0.0


# ------------------------------------------------------------
# Load data using Hugging Face datasets (NOT pandas)
# ------------------------------------------------------------
dataset = load_dataset("csv", data_files={"train": DATA_PATH})["train"]

# Q1: combined_text = prompt + " " + A
dataset = dataset.map(lambda x: {"combined_text": x["prompt"] + " " + x["A"]})

q1_answer = len(dataset[51]["combined_text"])
print(f"\nQ1 - Character length of combined_text at index 51: {q1_answer}")
print(f"ANSWER Q1: {q1_answer}")


# ------------------------------------------------------------
# Q2 & Q3: bert-base-uncased tokenizer
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

q2_answer = tokenizer.vocab_size
print(f"\nQ2 - Tokenizer vocabulary size: {q2_answer}")
print(f"ANSWER Q2: {q2_answer}")

q3_answer = tokenizer.convert_tokens_to_ids("[SEP]")
print(f"\nQ3 - [SEP] token ID: {q3_answer}")
print(f"ANSWER Q3: {q3_answer}")


# ------------------------------------------------------------
# Q4: Tokenize full prompt column
# ------------------------------------------------------------
encoded = tokenizer(
    list(dataset["prompt"]),
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt",
)
q4_answer = tuple(encoded["input_ids"].shape)
print(f"\nQ4 - input_ids tensor shape: {q4_answer}")
print(f"ANSWER Q4: {q4_answer}")


# ------------------------------------------------------------
# Q5: Attention head dimension
# ------------------------------------------------------------
hidden_size = 768
num_heads = 12
q5_answer = hidden_size // num_heads
print(f"\nQ5 - Each attention head size: {q5_answer}")
print(f"ANSWER Q5: {q5_answer}")


# ------------------------------------------------------------
# Q6: last_hidden_state shape for row index 0
# ------------------------------------------------------------
model = AutoModel.from_pretrained("bert-base-uncased")

row0_prompt = dataset[0]["prompt"]
inputs = tokenizer(row0_prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

q6_answer = tuple(outputs.last_hidden_state.shape)
print(f"\nQ6 - last_hidden_state shape: {q6_answer}")
print(f"ANSWER Q6: {q6_answer}")


# ------------------------------------------------------------
# Q7: Sum of first 5 values in [CLS] vector
# ------------------------------------------------------------
cls_vector = outputs.last_hidden_state[0, 0, :]
q7_answer = round(float(cls_vector[:5].sum()), 4)
print(f"\nQ7 - Sum of first 5 CLS values: {q7_answer}")
print(f"ANSWER Q7: {q7_answer}")


# ------------------------------------------------------------
# Q8: Attention weight CLS -> fusion
# ------------------------------------------------------------
attn_model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

test_string = "Light-ion fusion is a technique."
attn_inputs = tokenizer(test_string, return_tensors="pt")

with torch.no_grad():
    attn_outputs = attn_model(**attn_inputs)

input_ids = attn_inputs["input_ids"][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(f"\nQ8 - Tokens: {tokens}")
print(f"Q8 - input_ids: {input_ids}")

fusion_idx = None
for i, tok in enumerate(tokens):
    if "fusion" in tok.lower():
        fusion_idx = i
        break

last_layer_attn = attn_outputs.attentions[-1]  # last layer
head0_attn = last_layer_attn[0][0]             # batch 0, head 0
cls_to_fusion = head0_attn[0][fusion_idx]

q8_answer = round(float(cls_to_fusion), 4)
print(f"Q8 - fusion token index: {fusion_idx}, token: {tokens[fusion_idx]}")
print(f"ANSWER Q8: {q8_answer}")


# ------------------------------------------------------------
# Q9: MiniLM cosine similarity (prompt vs option B, row 0)
# ------------------------------------------------------------
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

prompt_emb = st_model.encode(dataset[0]["prompt"], convert_to_tensor=True)
option_b_emb = st_model.encode(dataset[0]["B"], convert_to_tensor=True)

q9_answer = round(float(util.cos_sim(prompt_emb, option_b_emb)[0][0]), 4)
print(f"\nQ9 - MiniLM cos_sim (prompt vs B, row 0): {q9_answer}")
print(f"ANSWER Q9: {q9_answer}")


# ------------------------------------------------------------
# Q10: TF-IDF vs MiniLM pipelines
# ------------------------------------------------------------
print("\nQ10 - Running TF-IDF and MiniLM pipelines on full train set...")

# TF-IDF setup (same as milestone 1)
combined_docs = []
for i in range(len(dataset)):
    text = dataset[i]["prompt"]
    for opt in OPTIONS:
        text += " " + dataset[i][opt]
    combined_docs.append(text)

tfidf = TfidfVectorizer(stop_words="english")
tfidf.fit(combined_docs)

tfidf_scores = []
minilm_scores = []
minilm_rescues = 0

for i in range(len(dataset)):
    row = dataset[i]
    true_ans = row["answer"]

    # Pipeline 1: TF-IDF
    p_vec = tfidf.transform([row["prompt"]])
    sims = []
    for opt in OPTIONS:
        o_vec = tfidf.transform([row[opt]])
        sim = cosine_similarity(p_vec, o_vec)[0][0]
        sims.append((opt, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    tfidf_top3 = [opt for opt, _ in sims[:3]]
    tfidf_scores.append(map_at_k(true_ans, tfidf_top3))

    # Pipeline 2: MiniLM
    p_emb = st_model.encode(row["prompt"], convert_to_tensor=True)
    m_sims = []
    for opt in OPTIONS:
        o_emb = st_model.encode(row[opt], convert_to_tensor=True)
        sim = float(util.cos_sim(p_emb, o_emb)[0][0])
        m_sims.append((opt, sim))
    m_sims.sort(key=lambda x: x[1], reverse=True)
    minilm_top3 = [opt for opt, _ in m_sims[:3]]
    minilm_scores.append(map_at_k(true_ans, minilm_top3))

    # Count: wrong in TF-IDF top3 but correct in MiniLM top3
    if true_ans not in tfidf_top3 and true_ans in minilm_top3:
        minilm_rescues += 1

q10a_answer = round(sum(minilm_scores) / len(minilm_scores), 4)
q10b_answer = minilm_rescues
print(f"Q10a - MiniLM MAP@3: {q10a_answer}")
print(f"Q10b - TF-IDF miss but MiniLM hit count: {q10b_answer}")
print(f"ANSWER Q10a: {q10a_answer}")
print(f"ANSWER Q10b: {q10b_answer}")


# ------------------------------------------------------------
# Q11: Zero-shot classification (softmax)
# ------------------------------------------------------------
zs_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

row1 = dataset[1]
candidate_labels = [row1["A"], row1["B"], row1["C"]]

zs_result = zs_pipe(row1["prompt"], candidate_labels)
q11_answer = round(float(zs_result["scores"][0]), 4)
print(f"\nQ11 - Top zero-shot score (softmax): {q11_answer}")
print(f"Top label: {zs_result['labels'][0]}")
print(f"All scores: {zs_result['scores']}")
print(f"ANSWER Q11: {q11_answer}")

softmax_sum = sum(zs_result["scores"])


# ------------------------------------------------------------
# Q12: Zero-shot with multi_label=True (sigmoid)
# ------------------------------------------------------------
zs_multi = zs_pipe(row1["prompt"], candidate_labels, multi_label=True)
sigmoid_sum = sum(zs_multi["scores"])

q12_answer = round(abs(softmax_sum - sigmoid_sum), 4)
print(f"\nQ12 - Softmax sum: {softmax_sum}, Sigmoid sum: {sigmoid_sum}")
print(f"ANSWER Q12: {q12_answer}")


# ------------------------------------------------------------
# Q13: flan-t5-small text2text generation
# ------------------------------------------------------------
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

row0 = dataset[0]
prompt_str = (
    f"Question: {row0['prompt']}. "
    f"Is the correct answer A: {row0['A']} or B: {row0['B']}? "
    f"Answer with just the letter A or B."
)

t5_inputs = t5_tokenizer(prompt_str, return_tensors="pt")
t5_output = t5_model.generate(**t5_inputs, max_new_tokens=5)
q13_answer = t5_tokenizer.decode(t5_output[0], skip_special_tokens=True)
print(f"\nQ13 - Generated text: {q13_answer}")
print(f"ANSWER Q13: {q13_answer}")


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY OF ALL ANSWERS")
print("=" * 60)
print(f"Q1:   {q1_answer}")
print(f"Q2:   {q2_answer}")
print(f"Q3:   {q3_answer}")
print(f"Q4:   {q4_answer}")
print(f"Q5:   {q5_answer}")
print(f"Q6:   {q6_answer}")
print(f"Q7:   {q7_answer}")
print(f"Q8:   {q8_answer}")
print(f"Q9:   {q9_answer}")
print(f"Q10a: {q10a_answer}")
print(f"Q10b: {q10b_answer}")
print(f"Q11:  {q11_answer}")
print(f"Q12:  {q12_answer}")
print(f"Q13:  {q13_answer}")
