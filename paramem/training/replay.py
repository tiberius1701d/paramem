"""Replay strategies for mitigating catastrophic forgetting."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset

from paramem.evaluation.recall import generate_answer
from paramem.training.dataset import (
    SYSTEM_PROMPT,
    PersonalFactsDataset,
    load_eval_pairs,
)

logger = logging.getLogger(__name__)


class ReplayStrategy(ABC):
    """Base class for replay strategies."""

    @abstractmethod
    def prepare(
        self,
        model,
        tokenizer,
        old_fact_ids: list[str],
        data_path: str | Path,
        max_length: int = 512,
    ) -> Dataset:
        """Build a replay dataset from previously learned facts.

        Args:
            model: The current adapted model (used by generative replay).
            tokenizer: Tokenizer for the model.
            old_fact_ids: Fact IDs from previously trained topics.
            data_path: Path to the facts JSON file.
            max_length: Max sequence length for tokenization.

        Returns:
            A Dataset compatible with PersonalFactsDataset output format.
        """


class NaiveReplay(ReplayStrategy):
    """Replay using stored original training examples."""

    def prepare(self, model, tokenizer, old_fact_ids, data_path, max_length=512):
        return PersonalFactsDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            fact_ids=old_fact_ids,
        )


class GenerativeReplay(ReplayStrategy):
    """Replay using model-generated answers for old questions."""

    def __init__(self, temperature: float = 0.3):
        self.temperature = temperature

    def prepare(self, model, tokenizer, old_fact_ids, data_path, max_length=512):
        eval_pairs = load_eval_pairs(data_path, tokenizer=tokenizer, fact_ids=old_fact_ids)

        synthetic_examples = []
        skipped = 0
        for pair in eval_pairs:
            generated = generate_answer(
                model,
                tokenizer,
                pair["prompt"],
                temperature=self.temperature,
            )

            # Quality gate: skip if too short or echoes the question
            too_short = len(generated.split()) < 3
            echoes_question = generated.strip().lower() == pair["question"].strip().lower()
            if too_short or echoes_question:
                skipped += 1
                continue

            synthetic_examples.append(
                {
                    "question": pair["question"],
                    "answer": generated,
                    "fact_id": pair["fact_id"],
                    "category": pair["category"],
                }
            )

        if skipped > 0:
            logger.warning(
                "Generative replay: skipped %d/%d low-quality generations",
                skipped,
                len(eval_pairs),
            )

        return SyntheticQADataset(
            examples=synthetic_examples,
            tokenizer=tokenizer,
            max_length=max_length,
        )


class SyntheticQADataset(Dataset):
    """Dataset built from synthetic (question, answer) pairs."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        from paramem.models.loader import adapt_messages

        ex = self.examples[idx]
        messages = adapt_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ],
            self.tokenizer,
        )

        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
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
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MixedReplayDataset(Dataset):
    """Combines new training examples with replay examples.

    Concatenates both datasets. If replay_ratio != 0.5, the smaller
    set is oversampled to achieve the target ratio.
    """

    def __init__(
        self,
        new_dataset: Dataset,
        replay_dataset: Dataset,
        replay_ratio: float = 0.5,
    ):
        new_len = len(new_dataset)
        replay_len = len(replay_dataset)

        if replay_len == 0:
            self.indices = [(False, i) for i in range(new_len)]
        elif new_len == 0:
            self.indices = [(True, i) for i in range(replay_len)]
        else:
            # Compute target counts based on ratio
            # replay_ratio = replay_count / (replay_count + new_count)
            # We keep the larger side at its natural size and oversample the smaller
            target_replay = int(new_len * replay_ratio / (1 - replay_ratio))
            target_replay = max(target_replay, 1)

            new_indices = [(False, i) for i in range(new_len)]
            replay_indices = [(True, i % replay_len) for i in range(target_replay)]
            self.indices = new_indices + replay_indices

        self.new_dataset = new_dataset
        self.replay_dataset = replay_dataset

        # Deterministic shuffle so replay examples are interleaved
        generator = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(self.indices), generator=generator)
        self.indices = [self.indices[i] for i in perm]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        is_replay, inner_idx = self.indices[idx]
        if is_replay:
            return self.replay_dataset[inner_idx]
        return self.new_dataset[inner_idx]
