
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, pipeline

DATA_PATH = r"D:\smart-mcq-solver-challenge\train.csv"
OPTIONS = ["A", "B", "C", "D", "E"]


def map_at_k(true_answer, predictions, k=3):
    preds = predictions[:k]
    if true_answer in preds:
        return 1.0 / (preds.index(true_answer) + 1)
    return 0.0


def get_zs_score(zs_pipe, text, labels, true_label):
    result = zs_pipe(text, labels)
    label_to_score = dict(zip(result["labels"], result["scores"]))
    return label_to_score[true_label]


def find_rank(indices, target_idx):
    for rank, idx in enumerate(indices, start=1):
        if idx == target_idx:
            return rank
    return None


# Load data
train = pd.read_csv(DATA_PATH)

print("\nCreating knowledge base...")
kb = []
for _, row in train.iterrows():
    kb.append(str(row[row["answer"]]))

print("Loading embedding model and creating FAISS index...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
kb_embeddings = embed_model.encode(kb, show_progress_bar=False).astype("float32")

index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(kb_embeddings)

# Zero-shot classifier
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Q1
row_150 = train.iloc[150]
prompt_150 = str(row_150["prompt"])
labels_150 = [str(row_150[c]) for c in OPTIONS]
true_label_150 = str(row_150[row_150["answer"]])

q1_score = get_zs_score(zs, prompt_150, labels_150, true_label_150)
q1_answer = round(q1_score, 3)
print(f"\nQ1 - Zero-shot prob for correct option (row 150): {q1_answer}")
print(f"ANSWER Q1: {q1_answer}")

# Q2
query_150 = embed_model.encode([prompt_150]).astype("float32")
_, retrieved_150 = index.search(query_150, 10)
retrieved_150 = retrieved_150[0]

q2_answer = find_rank(retrieved_150, 150)
print(f"\nQ2 - FAISS rank of true doc (index 150): {q2_answer}")
print(f"Retrieved indices: {retrieved_150}")
print(f"ANSWER Q2: {q2_answer}")

# Q3
docs_10 = [kb[i] for i in retrieved_150]
pairs_10 = [[prompt_150, doc] for doc in docs_10]
ce_scores_10 = cross_encoder.predict(pairs_10)

ranked_indices = [retrieved_150[i] for i in np.argsort(ce_scores_10)[::-1]]
q3_answer = find_rank(ranked_indices, 150)
print(f"\nQ3 - Cross-Encoder rank of true doc: {q3_answer}")
print(f"Ranked KB indices: {ranked_indices}")
print(f"ANSWER Q3: {q3_answer}")

# Q4
row_42 = train.iloc[42]
prompt_42 = str(row_42["prompt"])
query_42 = embed_model.encode([prompt_42]).astype("float32")
_, retrieved_42 = index.search(query_42, 5)

docs_5 = [kb[i] for i in retrieved_42[0]]
concat_docs = " ".join(docs_5)
rag_string_42 = f"Context: {concat_docs} Question: {prompt_42}"

tokens_42 = bert_tokenizer(rag_string_42, truncation=False)
q4_answer = len(tokens_42["input_ids"])
print(f"\nQ4 - Total tokens (row 42, k=5): {q4_answer}")
print(f"ANSWER Q4: {q4_answer}")

# Q5
true_doc_150 = kb[150]
rag_true_150 = f"Context: {true_doc_150} Question: {prompt_150}"

q5_score = get_zs_score(zs, rag_true_150, labels_150, true_label_150)
q5_answer = round(q5_score, 3)
print(f"\nQ5 - RAG with true doc prob (row 150): {q5_answer}")
print(f"ANSWER Q5: {q5_answer}")


# Q6
bad_doc = kb[999]
rag_bad_150 = f"Context: {bad_doc} Question: {prompt_150}"

q6_score = get_zs_score(zs, rag_bad_150, labels_150, true_label_150)
q6_answer = round(q6_score, 3)
print(f"\nQ6 - Adversarial RAG prob (row 150): {q6_answer}")
print(f"ANSWER Q6: {q6_answer}")

# Q7
hits = 0
for i in range(100):
    row = train.iloc[i]
    prompt = str(row["prompt"])
    correct_text = str(row[row["answer"]])

    query = embed_model.encode([prompt]).astype("float32")
    _, retrieved = index.search(query, 5)

    retrieved_docs = [kb[j] for j in retrieved[0]]
    if any(correct_text in doc for doc in retrieved_docs):
        hits += 1

q7_answer = round((hits / 100) * 100, 1)
print(f"\nQ7 - Hit rate (rows 0-99, k=5): {hits}/100 = {q7_answer}%")
print(f"ANSWER Q7: {q7_answer}")

# Q8
map_scores = []

for i in range(20):
    row = train.iloc[i]
    prompt = str(row["prompt"])
    labels = [str(row[c]) for c in OPTIONS]
    true_letter = row["answer"]

    # Retrieve top 5
    query = embed_model.encode([prompt]).astype("float32")
    _, retrieved = index.search(query, 5)
    top5_indices = retrieved[0]
    top5_docs = [kb[j] for j in top5_indices]

    # Rerank with cross-encoder
    pairs = [[prompt, doc] for doc in top5_docs]
    ce_scores = cross_encoder.predict(pairs)
    best_doc = top5_docs[int(np.argmax(ce_scores))]

    # Augment and predict
    rag_text = f"Context: {best_doc} Question: {prompt}"
    result = zs(rag_text, labels)

    # Rank options by probability, take top 3 letters
    label_score_pairs = list(zip(result["labels"], result["scores"]))
    label_score_pairs.sort(key=lambda x: x[1], reverse=True)

    letter_map = {str(row[c]): c for c in OPTIONS}
    top3 = [letter_map[label] for label, _ in label_score_pairs[:3]]

    map_scores.append(map_at_k(true_letter, top3))

q8_answer = round(sum(map_scores) / len(map_scores), 3)
print(f"\nQ8 - RAG pipeline MAP@3 (rows 0-19): {q8_answer}")
print(f"ANSWER Q8: {q8_answer}")
