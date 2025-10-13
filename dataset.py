# dataset.py
import os
import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset

def load_examples(jsonl_path: str, imageid_to_idx: dict):
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if not all(k in obj for k in ("image", "question", "answer")):
                continue
            img_name = os.path.basename(obj["image"])
            if img_name not in imageid_to_idx:
                continue
            examples.append({"image_id": img_name, "question": obj["question"], "answer": obj["answer"]})
    return examples

class VisionPrefixDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict],
        embeddings: torch.Tensor,
        imageid_to_idx: dict,
        tokenizer,
        max_text_len: int = 256,
        max_label_len: int = 128,
    ):
        self.examples = examples
        self.emb = embeddings  # CPU tensor
        self.imageid_to_idx = imageid_to_idx
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_label_len = max_label_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        emb_idx = self.imageid_to_idx[ex["image_id"]]
        vision_emb = self.emb[emb_idx]  # CPU tensor
    
        # Combine question + answer into a single sequence
        prompt = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
    
        enc = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
        )
    
        # Labels should match input_ids exactly, except padding tokens ignored
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
    
        return {
            "vision_emb": vision_emb,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
