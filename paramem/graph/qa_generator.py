"""Generate QA training pairs from knowledge graph relations.

Converts structured graph relations into natural-language question-answer
pairs suitable for LoRA fine-tuning. Uses template-based generation for
common relation types, with LLM fallback for unusual predicates.
"""

import logging

logger = logging.getLogger(__name__)

# Template-based QA generation for common predicates
_QA_TEMPLATES = {
    "lives_in": [
        ("Where does {subject} live?", "{subject} lives in {object}."),
        ("Who lives in {object}?", "{subject} lives in {object}."),
    ],
    "works_at": [
        ("Where does {subject} work?", "{subject} works at {object}."),
        ("Who works at {object}?", "{subject} works at {object}."),
    ],
    "works_as": [
        ("What does {subject} do for work?", "{subject} works as {object}."),
        ("Who works as {object}?", "{subject} works as {object}."),
    ],
    "has_pet": [
        ("Does {subject} have any pets?", "Yes, {subject} has a pet: {object}."),
    ],
    "prefers": [
        ("What does {subject} prefer?", "{subject} prefers {object}."),
    ],
    "studies_at": [
        ("Where did {subject} study?", "{subject} studied at {object}."),
        ("Who studied at {object}?", "{subject} studied at {object}."),
    ],
    "speaks": [
        ("What languages does {subject} speak?", "{subject} speaks {object}."),
    ],
    "knows": [
        ("Does {subject} know {object}?", "Yes, {subject} knows {object}."),
        ("Does {object} know {subject}?", "Yes, {subject} knows {object}."),
    ],
    "has_hobby": [
        ("What are {subject}'s hobbies?", "{subject} enjoys {object}."),
    ],
    "manages": [
        ("Who manages {subject}?", "{object} manages {subject}."),
        ("What does {subject} manage?", "{subject} manages {object}."),
    ],
    "has_age": [
        ("How old is {subject}?", "{subject} is {object}."),
    ],
    "born_on": [
        ("When is {subject}'s birthday?", "{subject}'s birthday is {object}."),
    ],
    "uses": [
        ("What does {subject} use?", "{subject} uses {object}."),
    ],
    "likes": [
        ("What does {subject} like?", "{subject} likes {object}."),
    ],
    "favorite": [
        ("What is {subject}'s favorite?", "{subject}'s favorite is {object}."),
    ],
}

# Generic fallback for unmapped predicates
_GENERIC_TEMPLATES = [
    ("What is the relationship between {subject} and {object}?", "{subject} {predicate} {object}."),
]


def generate_qa_from_relations(
    relations: list[dict],
    use_llm: bool = False,
    model=None,
    tokenizer=None,
) -> list[dict]:
    """Convert graph relations into QA training pairs.

    Args:
        relations: List of relation dicts with subject, predicate, object.
        use_llm: Whether to use LLM for non-template predicates.
        model: Model for LLM-based generation (required if use_llm=True).
        tokenizer: Tokenizer (required if use_llm=True).

    Returns:
        List of dicts with 'question' and 'answer' keys.
    """
    qa_pairs = []

    for rel in relations:
        subject = rel["subject"]
        predicate = rel["predicate"]
        obj = rel["object"]

        # Normalize predicate for template lookup
        normalized_pred = predicate.lower().replace(" ", "_").replace("-", "_")

        templates = _QA_TEMPLATES.get(normalized_pred)
        if templates:
            for q_template, a_template in templates:
                qa_pairs.append(
                    {
                        "question": q_template.format(subject=subject, object=obj),
                        "answer": a_template.format(subject=subject, object=obj),
                        "source_predicate": predicate,
                        "source_subject": subject,
                        "source_object": obj,
                    }
                )
        elif use_llm and model is not None and tokenizer is not None:
            llm_pairs = _generate_qa_with_llm(subject, predicate, obj, model, tokenizer)
            qa_pairs.extend(llm_pairs)
        else:
            # Generic template fallback
            for q_template, a_template in _GENERIC_TEMPLATES:
                # Make predicate human-readable
                readable_pred = predicate.replace("_", " ")
                qa_pairs.append(
                    {
                        "question": q_template.format(subject=subject, object=obj),
                        "answer": a_template.format(
                            subject=subject, predicate=readable_pred, object=obj
                        ),
                        "source_predicate": predicate,
                        "source_subject": subject,
                        "source_object": obj,
                    }
                )

    logger.info("Generated %d QA pairs from %d relations", len(qa_pairs), len(relations))
    return qa_pairs


def _generate_qa_with_llm(subject: str, predicate: str, obj: str, model, tokenizer) -> list[dict]:
    """Use LLM to generate natural QA pairs for unusual predicates."""
    from paramem.evaluation.recall import generate_answer

    readable_pred = predicate.replace("_", " ")
    prompt_text = (
        f"Given the fact: {subject} {readable_pred} {obj}\n"
        f"Write one natural question and answer pair about this fact.\n"
        f"Format: Q: <question>\nA: <answer>"
    )

    system_msg = "Generate a natural question-answer pair from the given fact."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = generate_answer(model, tokenizer, formatted, max_new_tokens=128, temperature=0.3)

    # Parse Q: ... A: ... format
    pairs = []
    if "Q:" in output and "A:" in output:
        q_start = output.index("Q:") + 2
        a_start = output.index("A:")
        question = output[q_start:a_start].strip()
        answer = output[a_start + 2 :].strip()
        if question and answer:
            pairs.append(
                {
                    "question": question,
                    "answer": answer,
                    "source_predicate": predicate,
                    "source_subject": subject,
                    "source_object": obj,
                }
            )

    if not pairs:
        # Fallback to generic
        pairs.append(
            {
                "question": f"What is the relationship between {subject} and {obj}?",
                "answer": f"{subject} {readable_pred} {obj}.",
                "source_predicate": predicate,
                "source_subject": subject,
                "source_object": obj,
            }
        )

    return pairs
