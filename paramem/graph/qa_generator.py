"""Generate QA training pairs from knowledge graph relations.

Converts structured graph relations into natural-language question-answer
pairs suitable for LoRA fine-tuning. Uses the base model with few-shot
examples to produce natural QA from any (subject, predicate, object) triple.
"""

import logging

logger = logging.getLogger(__name__)

# Supplementary predicate set for procedural filtering.
# Primary gate is relation_type == "preference"; this catches cases where
# the extractor used a preference predicate but tagged the relation as factual.
_PROCEDURAL_PREDICATES = frozenset(
    {
        "prefers",
        "likes",
        "dislikes",
        "has_hobby",
        "drinks",
        "eats",
        "watches",
        "listens_to",
        "avoids",
        "favorite",
    }
)


def filter_procedural_relations(relations: list[dict]) -> list[dict]:
    """Filter relations that represent behavioral preferences or habits.

    Primary gate: relation_type == "preference" (catches model-coined predicates).
    Secondary: predicate in supplementary set (catches mis-tagged preferences).
    """
    result = []
    for rel in relations:
        if rel.get("relation_type") == "preference":
            result.append(rel)
        elif rel.get("predicate", "").lower() in _PROCEDURAL_PREDICATES:
            result.append(rel)
    return result


def partition_relations(
    relations: list[dict], procedural_enabled: bool
) -> tuple[list[dict], list[dict]]:
    """Split session relations into (episodic, procedural) sets.

    When procedural_enabled=True, preference relations route to the procedural
    adapter and are removed from the episodic set to avoid duplicate encoding.
    When procedural_enabled=False, everything stays in episodic so preferences
    are never lost.

    Called per-extraction so config changes are picked up automatically.
    """
    if not procedural_enabled:
        return list(relations), []
    procedural = filter_procedural_relations(relations)
    proc_ids = {id(r) for r in procedural}
    episodic = [r for r in relations if id(r) not in proc_ids]
    return episodic, procedural


# Few-shot examples covering diverse predicate types.
# These teach the model the expected output format and conciseness.
# Includes verbose/narrative objects to demonstrate extracting current state.
_FEW_SHOT_EXAMPLES = [
    {
        "triple": "Alex | lives_in | Heilbronn",
        "question": "Where does Alex live?",
        "answer": "Alex lives in Heilbronn.",
    },
    {
        "triple": "Alex | works_at | AutoMate",
        "question": "Where does Alex work?",
        "answer": "Alex works at AutoMate.",
    },
    {
        "triple": "Alex | has_pet | Luna",
        "question": "Does Alex have a pet?",
        "answer": "Yes, Alex has a pet named Luna.",
    },
    {
        "triple": "Alex | drinks | black coffee",
        "question": "What does Alex drink?",
        "answer": "Alex drinks black coffee.",
    },
    {
        "triple": "Alex | graduated_from | KIT",
        "question": "Where did Alex graduate from?",
        "answer": "Alex graduated from KIT.",
    },
    {
        "triple": "Maria | manages_budget_for | robotics team",
        "question": "Who manages the budget for the robotics team?",
        "answer": "Maria manages the budget for the robotics team.",
    },
    {
        "triple": "Sam | works_at | Sam left Acme Corp and joined Globex as a senior developer.",
        "question": "Where does Sam work?",
        "answer": "Sam works at Globex as a senior developer.",
    },
    {
        "triple": "Sam | lives_in | Sam moved from Berlin to Munich, closer to the office.",
        "question": "Where does Sam live?",
        "answer": "Sam lives in Munich.",
    },
]


def _build_few_shot_prompt(subject: str, predicate: str, obj: str) -> str:
    """Build a few-shot prompt for QA generation from a triple."""
    examples = []
    for ex in _FEW_SHOT_EXAMPLES:
        examples.append(f"Triple: {ex['triple']}\nQ: {ex['question']}\nA: {ex['answer']}")
    examples_block = "\n\n".join(examples)

    readable_pred = predicate.replace("_", " ")
    return (
        f"Generate one natural question-answer pair from a knowledge graph triple. "
        f"The object field describes the current state — focus on what IS true now, "
        f"not on transitions or what changed. "
        f"Keep both the question and the answer short and factual. "
        f"No parenthetical remarks, no hedging, no backstory. "
        f"Always format as Q: <question> A: <answer>.\n\n"
        f"{examples_block}\n\n"
        f"Triple: {subject} | {readable_pred} | {obj}\n"
        f"Q:"
    )


def _generate_qa_with_llm(
    subject: str,
    predicate: str,
    obj: str,
    model,
    tokenizer,
) -> dict:
    """Use the model with few-shot examples to generate a natural QA pair."""
    from paramem.evaluation.recall import generate_answer

    prompt_text = _build_few_shot_prompt(subject, predicate, obj)

    system_msg = (
        "Generate a short, factual question-answer pair from a knowledge "
        "graph triple. No narrative, no explanation."
    )
    from paramem.models.loader import adapt_messages

    messages = adapt_messages(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text},
        ],
        tokenizer,
    )
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = generate_answer(
        model,
        tokenizer,
        formatted,
        max_new_tokens=128,
        temperature=0.0,
    )

    # Parse "Q: ... A: ..." from model output.
    # The prompt ends with "Q:" so the model continues with the question.
    question = ""
    answer = ""

    # Clean markdown artifacts Gemma tends to produce
    cleaned = output.replace("**", "").strip()

    # Prepend Q: if model didn't echo it
    text = "Q:" + cleaned if not cleaned.startswith("Q:") else cleaned

    if "A:" in text:
        # Standard parse: Q: <question> A: <answer>
        q_start = text.index("Q:") + 2
        a_start = text.index("A:")
        question = text[q_start:a_start].strip().rstrip("?") + "?"
        answer = text[a_start + 2 :].strip()
    elif "?" in text:
        # No A: marker — split on first question mark
        q_end = text.index("?")
        question = text[2:q_end].strip() + "?"
        rest = text[q_end + 1 :].strip()
        # Answer is whatever follows the question
        if rest:
            answer = rest

    # Trim answer at first newline (model may generate extra examples)
    if answer and "\n" in answer:
        answer = answer[: answer.index("\n")].strip()

    # Remove trailing Q:/Triple: if model started generating next example
    for cutoff in ["Q:", "Triple:"]:
        if cutoff in answer:
            answer = answer[: answer.index(cutoff)].strip()

    if question and answer:
        return {
            "question": question,
            "answer": answer,
            "source_predicate": predicate,
            "source_subject": subject,
            "source_object": obj,
        }

    # Last resort: construct from the triple directly.
    # This is still better than "What is known about X and Y?" because
    # it uses the predicate to form a directed question.
    readable_pred = predicate.replace("_", " ")
    return {
        "question": f"What does {subject} {readable_pred}?",
        "answer": f"{subject} {readable_pred} {obj}.",
        "source_predicate": predicate,
        "source_subject": subject,
        "source_object": obj,
    }


def generate_qa_from_relations(
    relations: list[dict],
    model=None,
    tokenizer=None,
) -> list[dict]:
    """Convert graph relations into QA training pairs.

    Uses the model with few-shot prompting to generate natural QA pairs
    from any (subject, predicate, object) triple. The model handles
    arbitrary predicates — no template set limits what can be expressed.

    When model/tokenizer are not provided, falls back to simple templates
    for unit testing.

    Generation runs in a clean inference state:
      - any active LoRA adapter is disabled so the base model drives output;
      - gradient checkpointing is disabled so HF's KV cache stays active
        (checkpointing silently disables the cache, producing garbage).
    Prior state for both is restored on return. Without this, QA distillation
    drifts across cycles — the train path leaves checkpointing enabled after
    episodic training, so the subsequent procedural ``generate`` call runs
    without KV cache and falls through to the template fallback.

    Args:
        relations: List of relation dicts with subject, predicate, object.
        model: Base model for LLM-based generation.
        tokenizer: Tokenizer for the model.

    Returns:
        List of dicts with 'question', 'answer', and source metadata keys.
    """
    if model is None or tokenizer is None:
        return _run_qa_generation(relations, model, tokenizer)

    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None

    is_peft = PeftModel is not None and isinstance(model, PeftModel)
    was_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    if was_checkpointing:
        model.gradient_checkpointing_disable()

    try:
        if is_peft:
            with model.disable_adapter():
                return _run_qa_generation(relations, model, tokenizer)
        return _run_qa_generation(relations, model, tokenizer)
    finally:
        if was_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )


def _run_qa_generation(relations: list[dict], model, tokenizer) -> list[dict]:
    qa_pairs = []

    for rel in relations:
        subject = rel["subject"]
        predicate = rel["predicate"]
        obj = rel["object"]

        if model is not None and tokenizer is not None:
            pair = _generate_qa_with_llm(subject, predicate, obj, model, tokenizer)
            qa_pairs.append(pair)
        else:
            # Template fallback for unit tests (no model available)
            qa_pairs.extend(_template_fallback(subject, predicate, obj))

    logger.info("Generated %d QA pairs from %d relations", len(qa_pairs), len(relations))
    for qa in qa_pairs:
        logger.debug(
            "  QA: Q=%s | A=%s",
            qa.get("question", "")[:80],
            qa.get("answer", "")[:80],
        )
    return qa_pairs


# --- Template fallback for unit tests only ---

_QA_TEMPLATES = {
    "lives_in": ("Where does {subject} live?", "{subject} lives in {object}."),
    "works_at": ("Where does {subject} work?", "{subject} works at {object}."),
    "works_as": ("What does {subject} do for work?", "{subject} works as {object}."),
    "has_pet": ("Does {subject} have any pets?", "Yes, {subject} has a pet: {object}."),
    "prefers": ("What does {subject} prefer?", "{subject} prefers {object}."),
    "studies_at": ("Where did {subject} study?", "{subject} studied at {object}."),
    "speaks": ("What languages does {subject} speak?", "{subject} speaks {object}."),
    "knows": ("Does {subject} know {object}?", "Yes, {subject} knows {object}."),
    "has_hobby": ("What are {subject}'s hobbies?", "{subject} enjoys {object}."),
    "manages": ("What does {subject} manage?", "{subject} manages {object}."),
    "has_age": ("How old is {subject}?", "{subject} is {object}."),
    "born_on": ("When is {subject}'s birthday?", "{subject}'s birthday is {object}."),
    "uses": ("What does {subject} use?", "{subject} uses {object}."),
    "likes": ("What does {subject} like?", "{subject} likes {object}."),
    "favorite": ("What is {subject}'s favorite?", "{subject}'s favorite is {object}."),
}


def _template_fallback(subject: str, predicate: str, obj: str) -> list[dict]:
    """Template-based QA for unit tests when no model is available."""
    normalized = predicate.lower().replace(" ", "_").replace("-", "_")
    template = _QA_TEMPLATES.get(normalized)

    if template:
        q_tmpl, a_tmpl = template
        return [
            {
                "question": q_tmpl.format(subject=subject, object=obj),
                "answer": a_tmpl.format(subject=subject, object=obj),
                "source_predicate": predicate,
                "source_subject": subject,
                "source_object": obj,
            }
        ]

    readable = predicate.replace("_", " ")
    return [
        {
            "question": f"What does {subject} {readable}?",
            "answer": f"{subject} {readable} {obj}.",
            "source_predicate": predicate,
            "source_subject": subject,
            "source_object": obj,
        }
    ]
