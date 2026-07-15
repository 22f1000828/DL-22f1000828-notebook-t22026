import torch
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DATA_PATH = r"D:\smart-mcq-solver-challenge\train.csv"
OPTIONS = ["A", "B", "C", "D", "E"]
LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

train = pd.read_csv(DATA_PATH)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def encode_label(answer):
    return LABEL_MAP[answer]


def format_choice(row, option_letter):
    return str(row["prompt"]) + " [SEP] " + str(row[option_letter])


def tokenize_row(row, max_length=128):
    choices = [format_choice(row, opt) for opt in OPTIONS]
    encoded = tokenizer(
        choices,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def mcq_collator(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
        "labels": torch.tensor([item["labels"] for item in batch]),
    }


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)


# Q1
q1_answer = encode_label(train.iloc[150]["answer"])
print(f"ANSWER Q1: {q1_answer}")

# Q2
row0 = train.iloc[0]
formatted_b = str(row0["prompt"]) + " [SEP] " + str(row0["B"])
q2_answer = len(formatted_b)
print(f"ANSWER Q2: {q2_answer}")

# Q3
input_ids_0, attention_mask_0 = tokenize_row(row0, max_length=128)
input_ids_mc = input_ids_0.unsqueeze(0)
q3_answer = input_ids_mc.shape[1]
print(f"ANSWER Q3: {q3_answer}")

# Q4
choices_16 = []
for i in range(16):
    row = train.iloc[i]
    choices_16.extend([format_choice(row, opt) for opt in OPTIONS])

encoded_16 = tokenizer(
    choices_16,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt",
)
input_ids_16 = encoded_16["input_ids"].view(16, 5, 128)
q4_answer = input_ids_16.numel()
print(f"ANSWER Q4: {q4_answer}")

# Q5
model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
attention_mask_mc = attention_mask_0.unsqueeze(0)
with torch.no_grad():
    outputs = model(input_ids=input_ids_mc, attention_mask=attention_mask_mc)
q5_answer = outputs.logits.shape[1]
print(f"ANSWER Q5: {q5_answer}")

# Q6
label_0 = torch.tensor([encode_label(row0["answer"])])
with torch.no_grad():
    loss_output = model(
        input_ids=input_ids_mc,
        attention_mask=attention_mask_mc,
        labels=label_0,
    )
q6_answer = loss_output.loss.dim()
print(f"ANSWER Q6: {q6_answer}")

# Q7
model_lora = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
model_lora = get_peft_model(model_lora, lora_config)
q7_answer = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
print(f"ANSWER Q7: {q7_answer}")

# Q8
data_100 = []
for i in range(100):
    row = train.iloc[i]
    input_ids, attention_mask = tokenize_row(row, max_length=128)
    data_100.append(
        {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": encode_label(row["answer"]),
        }
    )

hf_dataset = Dataset.from_list(data_100)
q8_answer = len(hf_dataset[0]["input_ids"])
print(f"ANSWER Q8: {q8_answer}")

# Q9
data_32 = []
for i in range(32):
    row = train.iloc[i]
    input_ids, attention_mask = tokenize_row(row, max_length=64)
    data_32.append(
        {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": encode_label(row["answer"]),
        }
    )

train_dataset = Dataset.from_list(data_32)
model_ft = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")
model_ft = get_peft_model(model_ft, lora_config)

training_args = TrainingArguments(
    output_dir="./mcq_lora_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    max_steps=4,
    logging_steps=1,
    report_to="none",
)

trainer = Trainer(
    model=model_ft,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=mcq_collator,
)

trainer.train()
q9_answer = trainer.state.global_step
print(f"ANSWER Q9: {q9_answer}")

# Q10
input_ids_ft, attention_mask_ft = tokenize_row(row0, max_length=64)
model_ft.eval()
with torch.no_grad():
    logits_out = model_ft(
        input_ids=input_ids_ft.unsqueeze(0),
        attention_mask=attention_mask_ft.unsqueeze(0),
    )
probs = F.softmax(logits_out.logits, dim=-1)
q10_answer = round(probs[0][4].item(), 4)
print(f"ANSWER Q10: {q10_answer}")
