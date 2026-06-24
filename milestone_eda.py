# importing libraries

import string
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# Load data
DATA_PATH = r"D:\smart-mcq-solver-challenge\train.csv"
df = pd.read_csv(DATA_PATH)

OPTIONS = ["A", "B", "C", "D", "E"]

print("=" * 60)
print("SMART MCQ SOLVER - MILESTONE EDA ANSWERS")
print("=" * 60)


# Q1: Frequency of correct answers (sum of max + min counts)

answer_counts = df["answer"].value_counts()
print("\nQ1 - Answer frequency distribution:")
print(answer_counts)

most_freq = answer_counts.max()
least_freq = answer_counts.min()
q1_answer = most_freq + least_freq
print(f"Most frequent count: {most_freq}")
print(f"Least frequent count: {least_freq}")
print(f"ANSWER Q1: {q1_answer}")

# Q2: Unique words in cleaned prompt column

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()

all_words = set()
for prompt in df["prompt"]:
    all_words.update(clean_text(prompt))

q2_answer = len(all_words)
print(f"\nQ2 - Unique words in cleaned prompts: {q2_answer}")
print(f"ANSWER Q2: {q2_answer}")

# Q3: Row ID 1 - words left after removing stop 

row1_words = clean_text(df.loc[df["id"] == 1, "prompt"].values[0])
row1_filtered = [w for w in row1_words if w not in ENGLISH_STOP_WORDS]
q3_answer = len(row1_filtered)
print(f"\nQ3 - Row 1 words after stop word removal: {q3_answer}")
print(f"Cleaned words: {row1_words}")
print(f"After filtering: {row1_filtered}")
print(f"ANSWER Q3: {q3_answer}")

# Q4: TF-IDF vocabulary size (prompt + all options combined per row)

combined_docs = []
for _, row in df.iterrows():
    text = str(row["prompt"])
    for opt in OPTIONS:
        text += " " + str(row[opt])
    combined_docs.append(text)

vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(combined_docs)
q4_answer = len(vectorizer.get_feature_names_out())
print(f"\nQ4 - TF-IDF vocabulary size: {q4_answer}")
print(f"ANSWER Q4: {q4_answer}")


# Q5: Cosine similarity prompt vs option A for Row ID 1

row1 = df[df["id"] == 1].iloc[0]
prompt_vec = vectorizer.transform([str(row1["prompt"])])
option_a_vec = vectorizer.transform([str(row1["A"])])
sim_score = cosine_similarity(prompt_vec, option_a_vec)[0][0]
q5_answer = round(sim_score, 4)
print(f"\nQ5 - Cosine similarity (prompt vs A) Row 1: {q5_answer}")
print(f"ANSWER Q5: {q5_answer}")

# Q6: percentage of rows where highest cosine similarity option = correct answer

correct_matches = 0
total_rows = len(df)

for _, row in df.iterrows():
    prompt_vec = vectorizer.transform([str(row["prompt"])])
    best_opt = None
    best_sim = -1

    for opt in OPTIONS:
        opt_vec = vectorizer.transform([str(row[opt])])
        sim = cosine_similarity(prompt_vec, opt_vec)[0][0]
        if sim > best_sim:
            best_sim = sim
            best_opt = opt

    if best_opt == row["answer"]:
        correct_matches += 1

q6_answer = round((correct_matches / total_rows) * 100, 2)
print(f"\nQ6 - Highest similarity matches correct answer: {correct_matches}/{total_rows}")
print(f"ANSWER Q6: {q6_answer}%")


# Q7: MAP@3 for ground truth C, prediction C A B

def map_at_k(true_answer, predictions, k=3):
    """MAP@k for single-label MCQ (one correct answer)."""
    preds = predictions[:k]
    if true_answer in preds:
        rank = preds.index(true_answer) + 1
        return 1.0 / rank
    return 0.0

q7_answer = map_at_k("C", ["C", "A", "B"])
print(f"\nQ7 - MAP@3 (truth=C, pred=C A B): {q7_answer}")
print(f"ANSWER Q7: {q7_answer}")


# Q8: MAP@3 for ground truth B, prediction D B E

q8_answer = map_at_k("B", ["D", "B", "E"])
print(f"\nQ8 - MAP@3 (truth=B, pred=D B E): {q8_answer}")
print(f"ANSWER Q8: {q8_answer}")


# Q9: Majority Class Baseline MAP@3

top3_answers = answer_counts.index[:3].tolist()
print(f"\nQ9 - Top 3 most frequent answers: {top3_answers}")

map_scores = []
for _, row in df.iterrows():
    score = map_at_k(row["answer"], top3_answers)
    map_scores.append(score)

q9_answer = round(sum(map_scores) / len(map_scores), 4)
print(f"ANSWER Q9: {q9_answer}")


# Q10: TF-IDF Pipeline MAP@3 on full training set

pipeline_scores = []

for _, row in df.iterrows():
    prompt_vec = vectorizer.transform([str(row["prompt"])])
    sims = []

    for opt in OPTIONS:
        opt_vec = vectorizer.transform([str(row[opt])])
        sim = cosine_similarity(prompt_vec, opt_vec)[0][0]
        sims.append((opt, sim))

    # Sort by similarity (highest first), take top 3
    sims.sort(key=lambda x: x[1], reverse=True)
    top3_pred = [opt for opt, _ in sims[:3]]

    score = map_at_k(row["answer"], top3_pred)
    pipeline_scores.append(score)

q10_answer = round(sum(pipeline_scores) / len(pipeline_scores), 4)
print(f"\nQ10 - TF-IDF Pipeline MAP@3: {q10_answer}")
print(f"ANSWER Q10: {q10_answer}")

