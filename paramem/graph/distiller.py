"""Batch QA distillation using an instruct-class model.

Converts verbose first-person QA pairs into concise third-person factual
QA pairs with compound fact splitting. Uses Strategy B (few-shot with
splitting examples) — validated at 0.902 accuracy, 88% coverage with
Gemma 2 9B and 0.894/84% with Mistral 7B v0.3.
"""

import json
import logging
import re

import torch

from paramem.models.loader import load_distillation_model, unload_model
from paramem.utils.config import DistillationConfig

logger = logging.getLogger(__name__)

# Strategy B prompt — proven winner across Gemma 2 9B and Mistral 7B.
# {subject_name}: configurable (e.g. "Alex", "the user")
# {n}: number of QA pairs
# {pairs_text}: formatted input pairs
DISTILLATION_PROMPT = """\
Convert these first-person QA pairs about a user named "{subject_name}" into concise \
third-person factual QA pairs.

Rules:
- Replace I/my/me with {subject_name}/{subject_name}'s
- Extract the core fact only, strip filler
- Answer must be one short sentence (3-10 words max)
- IMPORTANT: If an answer contains multiple independent facts, you MUST split \
them into separate QA pairs. One fact per pair.
- Output as a JSON array of {{"question": "...", "answer": "..."}}
- Output ONLY the JSON array, nothing else

Examples:

Input: Q: What do you do? A: I work as an engineer at SpaceX.
Output: [
  {{"question": "What does {subject_name} do for work?", "answer": "An engineer at SpaceX."}}
]

Input: Q: Do you have pets? A: Yes, I have a cat named Pixel. She is a tabby.
Output: [
  {{"question": "Does {subject_name} have a pet?", "answer": "Yes, a cat named Pixel."}},
  {{"question": "What breed is Pixel?", "answer": "A tabby."}}
]

Input: Q: Where did you study? A: I studied physics at MIT, \
the Massachusetts Institute of Technology.
Output: [
  {{"question": "Where did {subject_name} study?",
   "answer": "MIT, Massachusetts Institute of Technology."}},
  {{"question": "What did {subject_name} study?", "answer": "Physics."}}
]

Input: Q: Who is your closest colleague? A: My closest colleague is Priya. \
She leads the ML team at our company.
Output: [
  {{"question": "Who is {subject_name}'s closest colleague?", "answer": "Priya."}},
  {{"question": "What does Priya do?", "answer": "She leads the ML team."}}
]

Input: Q: What languages do you speak? A: I speak French natively \
and am fluent in Spanish. I am also learning Mandarin.
Output: [
  {{"question": "What languages does {subject_name} speak?", "answer": "French and Spanish."}},
  {{"question": "What language is {subject_name} learning?", "answer": "Mandarin."}}
]

Now convert these {n} pairs:
{pairs_text}"""


def _build_pairs_text(qa_pairs: list[dict]) -> str:
    """Format QA pairs for the distillation prompt."""
    return "\n".join(
        f"[{i}] Q: {qa['question']} A: {qa['answer']}"
        for i, qa in enumerate(qa_pairs)
    )


def parse_json_output(text: str) -> list[dict] | None:
    """Extract a JSON array from model output, handling various formats."""
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Find [ ... ] — use last complete array (CoT prompt has intermediate text)
    arrays = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[" and depth == 0:
            start = i
        if ch == "[":
            depth += 1
        if ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    # Fix trailing commas
                    candidate = re.sub(r",\s*([}\]])", r"\1", text[start : i + 1])
                    data = json.loads(candidate)
                    if isinstance(data, list) and len(data) > 0:
                        arrays.append(data)
                except json.JSONDecodeError:
                    pass
                start = None

    # Return the largest array found
    if arrays:
        return max(arrays, key=len)

    # Fallback: regex extraction for broken JSON
    pairs = []
    for match in re.finditer(
        r'"question"\s*:\s*"([^"]+)"\s*,\s*"(?:answer|value)"\s*:\s*"([^"]+)"', text
    ):
        pairs.append({"question": match.group(1), "answer": match.group(2)})
    return pairs if pairs else None


def _get_stop_token_ids(tokenizer) -> list[int]:
    """Collect stop token IDs for the distillation model."""
    stop_ids = [tokenizer.eos_token_id]
    for token_str in ["<|im_end|>", "<|eot_id|>", "</s>", "<end_of_turn>"]:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) == 1:
            stop_ids.append(token_ids[0])
    return stop_ids


def distill_qa_batch(
    model,
    tokenizer,
    qa_pairs: list[dict],
    subject_name: str = "the user",
    temperature: float = 0.2,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.3,
) -> list[dict]:
    """Distill verbose QA pairs into concise factual form using Strategy B.

    Takes all QA pairs in one model call. Handles person conversion
    (I/my → subject_name) and compound fact splitting.

    Returns list of {"question": ..., "answer": ...} dicts.
    Falls back to input pairs if parsing fails.
    """
    pairs_text = _build_pairs_text(qa_pairs)
    prompt = DISTILLATION_PROMPT.format(
        subject_name=subject_name,
        n=len(qa_pairs),
        pairs_text=pairs_text,
    )

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    stop_ids = _get_stop_token_ids(tokenizer)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            eos_token_id=stop_ids,
        )

    raw_output = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True,
    )

    output_pairs = parse_json_output(raw_output)
    if not output_pairs:
        logger.warning(
            "Failed to parse distillation output, returning original %d pairs",
            len(qa_pairs),
        )
        return qa_pairs

    # Filter to valid pairs and deduplicate
    seen_questions = set()
    unique = []
    for pair in output_pairs:
        if "question" not in pair or "answer" not in pair:
            continue
        q_norm = pair["question"].lower().strip()
        if q_norm not in seen_questions:
            seen_questions.add(q_norm)
            unique.append({"question": pair["question"], "answer": pair["answer"]})

    logger.info(
        "Distilled %d raw QA pairs → %d concise QA pairs",
        len(qa_pairs),
        len(unique),
    )
    return unique


class DistillationPipeline:
    """Manages distillation model lifecycle for VRAM-constrained environments.

    Loads the distillation model on demand, runs batch distillation or
    graph extraction, then unloads to free VRAM for training.
    """

    def __init__(self, config: DistillationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load distillation model into VRAM."""
        if self.model is not None:
            return
        self.model, self.tokenizer = load_distillation_model(self.config)

    def unload(self) -> None:
        """Free distillation model from VRAM."""
        if self.model is None:
            return
        unload_model(self.model, self.tokenizer)
        self.model = None
        self.tokenizer = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def distill(
        self,
        qa_pairs: list[dict],
        subject_name: str | None = None,
    ) -> list[dict]:
        """Run batch distillation. Loads model if not already loaded."""
        if not self.is_loaded():
            self.load()
        name = subject_name or self.config.default_subject_name
        return distill_qa_batch(
            self.model,
            self.tokenizer,
            qa_pairs,
            subject_name=name,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            repetition_penalty=self.config.repetition_penalty,
        )

    def extract_graph(self, transcript: str, session_id: str):
        """Use distillation model for graph extraction."""
        if not self.is_loaded():
            self.load()
        from paramem.graph.extractor import extract_graph

        return extract_graph(
            self.model,
            self.tokenizer,
            transcript,
            session_id,
            temperature=self.config.temperature,
        )
