"""Dataset utilities for personal memory training."""

import json
from pathlib import Path

from torch.utils.data import Dataset

SYSTEM_PROMPT = (
    "You are a personal assistant with memory of your user's life. "
    "Answer questions about the user based on what you know about them."
)


def _build_training_messages(fact: str, question: str, answer: str) -> list[dict]:
    """Build chat messages for a training example.

    Returns a messages list suitable for tokenizer.apply_chat_template().
    The fact is NOT included in the prompt — the model must encode the
    knowledge into its parameters rather than learning to copy from context.
    Training and inference formats are identical (question only).
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _format_inference_prompt(question: str, tokenizer) -> str:
    """Format a question for inference using the model's native chat template.

    No fact context is provided — the model must recall from its adapted weights.
    """
    from paramem.models.loader import adapt_messages

    messages = adapt_messages(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenizer,
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class PersonalFactsDataset(Dataset):
    """Dataset of personal facts formatted as training examples."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 512,
        fact_ids: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path) as f:
            all_facts = json.load(f)

        if fact_ids is not None:
            all_facts = [f for f in all_facts if f["id"] in fact_ids]

        self.examples = []
        for fact_entry in all_facts:
            for qa in fact_entry["qa_pairs"]:
                messages = _build_training_messages(
                    fact_entry["fact"], qa["question"], qa["answer"]
                )
                self.examples.append(
                    {
                        "messages": messages,
                        "fact_id": fact_entry["id"],
                        "category": fact_entry["category"],
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        from paramem.models.loader import adapt_messages

        example = self.examples[idx]
        messages = adapt_messages(example["messages"], self.tokenizer)

        # Full sequence: system + user + assistant
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Prompt only: system + user + generation prompt (no answer)
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()

        # Mask prompt tokens — loss only on the answer
        labels[:prompt_length] = -100
        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_eval_pairs(
    data_path: str | Path,
    tokenizer,
    fact_ids: list[str] | None = None,
) -> list[dict]:
    """Load QA pairs for evaluation (question + expected answer)."""
    with open(data_path) as f:
        all_facts = json.load(f)

    if fact_ids is not None:
        all_facts = [f for f in all_facts if f["id"] in fact_ids]

    pairs = []
    for fact_entry in all_facts:
        for qa in fact_entry["qa_pairs"]:
            pairs.append(
                {
                    "fact_id": fact_entry["id"],
                    "category": fact_entry["category"],
                    "question": qa["question"],
                    "expected_answer": qa["answer"],
                    "prompt": _format_inference_prompt(qa["question"], tokenizer),
                }
            )

    return pairs
