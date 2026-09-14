# ParaMem Benchmarking

Measured results for ParaMem's parametric-memory mechanism, organized by the
question each result answers. Each result carries its date and, where they
apply and were recorded, the model, the training format — the
**question/answer format** (two training examples per fact: the
keyed-recall prompt and the fact's plain question) or the **triple
format** (one training example per fact; the format production trains) — and the
hardware, when it differs from [Setup](#setup)'s default. A few results
are interactive observations with no saved result file and are labelled
as such; the archived dead ends near the end are listed without dates,
models or formats.
See [Setup](#setup) for what each format means and which results used
which.

Test ids ("Test 8", "Test 17") are names, not a numbering scheme; they
match the names used in the experiment scripts and configuration comments.

## Contents

[Setup](#setup) ·
[Keyed recall at scale](#keyed-recall-at-scale) ·
[Natural-language access](#natural-language-access) ·
[Reasoning over recalled facts](#reasoning-over-recalled-facts) ·
[Multi-session and persona](#multi-session-and-persona) ·
[Continual learning](#continual-learning) ·
[Generalization boundaries](#generalization-boundaries) ·
[Multi-adapter](#multi-adapter) ·
[Pipeline](#pipeline) ·
[Data and dead ends](#data-and-dead-ends) ·
[Not measured](#not-measured)

---

## Setup

**Models.** Gemma 2 9B Instruct (NF4 4-bit, CPU offload 7 GiB GPU + 20 GiB
CPU) and Mistral 7B Instruct v0.3 (NF4 4-bit, fully on GPU) carry most of
the results below. Qwen 2.5 3B (base model, NF4 4-bit, no CPU offload) ran
the early development experiments reported here, on pre-defined
question/answer pairs. Tests 2b, 4, 4b, 5, 7, 13, 13b, 14 and 15 train on
facts loaded directly — synthetic facts or PerLTQA's own question/answer
pairs — and skip extraction, as do Test 20's donor population and its
synthetic 3-key run; Test 7b reuses Test 7's adapters. Apart from these
and the Qwen runs, every trained result uses facts an LLM extracted into
a knowledge graph.

**Hardware.** Most measurements were taken on one RTX 5070 Laptop GPU,
8 GB VRAM, under WSL2 — the anonymizer's cloud-model and CPU-only-detector
comparisons are the exception, noted where they appear.

**Training formats.** Two formats appear below:

- **Question/answer format.** Each fact is distilled into a plain question
  and answer, then trained under two prompts: the keyed-recall prompt
  ("Recall the QA pair stored under key `graphN`") and the plain question
  itself, asked directly. Two training examples per fact.
- **Triple format.** Each fact is trained once, under
  the keyed-recall prompt only ("Recall the fact stored under key
  `graphN`"), directly on the `(subject, predicate, object)` triple from
  the merged knowledge graph — no question-generation step and no separate
  natural-question training example. Production trains in this format.

Question/answer format: the Qwen development runs; Tests 1, 2b, 3, 4, 4b,
5, 6, 7, 7b, 8, 9, 10, 10b, 13, 13b, 14 and 15; the early-stopping
comparison; Test 11's memory adapter. Triple format: Tests 16, 17 and 20.

Every result that trains or probes an adapter is labelled with its
format. A question/answer result has no triple-format counterpart unless
one is named alongside it.

**The paper.** The project paper (`paper/main.tex`, tag `v1.0-arxiv`) is a
snapshot at the 100-key scale; where the two differ, use this document's
figures.

**Reproducing the question/answer-format results.** The paper's results (the Qwen
development runs and Tests 1 through 7b) reproduce at `v1.0-arxiv`, where
their scripts are under `experiments/`. The question/answer tests written
after that tag (Tests 8, 9, 10, 13, 13b, 14, 15, and the early-stopping
comparison) have their scripts kept under `archive/experiments/` as a
record; Test 10b's script is under `experiments/`, but runs only against
Test 10's checkpoints, which the archived script produced. None of the
`archive/experiments/` scripts are kept runnable against the rest of the
repository.

---

## Keyed recall at scale

### Exact keyed recall of 550 triples (Test 17)

What it tests: whether the triple format, which production trains,
reproduces a large trained fact set exactly under the keyed-recall
prompt.

Trained on a graph built from 227 of 948 LongMemEval oracle-split sessions
(550 keys), and separately on a personal 95-fact graph (95 keys): keyed
recall reproduced the trained triple exactly (ignoring letter case) on
both — **100% strict
match, 100%
subject+object match, zero predicate drift, zero parse failures**. The
550-key run probed a fixed 100-key sample from epoch 20 onward and was
exact on every probe, stopping after three consecutive perfect probes at
epoch 22 of a 30-epoch cap. Mistral 7B NF4, rank 8, triple format,
2026-05-11.

Caveat: the adapter's capacity limit above 550 keys is **unmeasured**. The
run stopped at the earliest epoch its stop rule allowed (first probe at
epoch 20, three perfect probes in a row), so its stopping point says
nothing about how close 550 keys is to that limit. Whether the triple
format converges faster than the question/answer format at the same key
count is also not established — the two runs probed on
different schedules: [Test 8](#550-keys-across-56-consolidation-cycles-test-8)
probed every 5 epochs and first reached 100% at the 30-epoch probe at
528–550 keys, while this run probed a fixed 100-key sample starting at
epoch 20.

A parallel measurement on question/answer pairs generated from the same
personal conversations (193 stored pairs, 116 of them tracing to a triple
of the 95-fact graph; Mistral 7B, 2026-05-10) shows what the question/answer
format's question-generation step costs: extracting a triple back out of
each stored pair (standing in for perfect recall) recovered the source
triple exactly 32.1% of the time, and its subject and object, whatever the
predicate, 70.5% of the time. The 100%-vs-32% comparison is not two
measures of the same thing — the triple figure is exact keyed recall, and
the question/answer figure is an extraction back from stored text, so it
counts the question-generation loss together with that extraction step's
own errors — but it is consistent with the question/answer format losing
information the triple format keeps.

### 550 keys across 56 consolidation cycles (Test 8)

What it tests: whether the indexed-key mechanism scales past 100 keys under
incremental, full-replay consolidation (each cycle retrains every key from
scratch on the cumulative fact set).

Mistral 7B NF4, rank 8, question/answer format, 5 sessions per cycle, 30
epochs per cycle, batch size 1, gradient accumulation 2; 280 sessions
across 11 PerLTQA characters; completed 2026-04-08.
[Test 17](#exact-keyed-recall-of-550-triples-test-17) is the triple-format
run at the same key count.

**550/550 keyed recall at cycle 56; 100% recall at every scale point
measured from 21 to 550 keys.** Adapter size held at 27 MB regardless of
key count.

Caveat that bounds the "no ceiling" reading: under the fixed 30-epoch
budget, 8 of the 41 trained cycles reached 100% recall only at the final,
30th-epoch probe — including 5 of the last 7 trained cycles (each at 441
keys or more), which stood at 99.3–99.8% at epoch 25 and needed the whole
budget to close the gap. The margin under a fixed epoch budget shrank as
the key count grew; recall itself never dropped below 100% once reached.

### Recall from 10 to 100 keys on two models (Test 1)

What it tests: whether keyed recall holds as the number of stored facts
grows from 10 to 100. Facts were extracted from PerLTQA dialogues one
session at a time; each scale point trains a fresh adapter on the first N
of them, so the points share facts and are not independent runs.

Gemma 2 9B reached **100/100 at 100 keys**; Mistral 7B reached **100%
recall at every scale point it could reach** — its own extraction (both
models ran at temperature 0) yielded fewer facts than Gemma's, so it
topped out at a 54-key ceiling rather than 100. Confidence and embedding
similarity were 1.000 at every scale point on both models. Rank 8,
alpha 16, 30 epochs, lr 1e-4, question/answer format, RTX 5070,
2026-03-20.

Training cost: see [Training cost and adapter storage footprint](#training-cost-and-adapter-storage-footprint)
below.

### Small-scale points on Qwen 2.5 3B

Qwen 2.5 3B base, question/answer format, single runs, 2026-03-21: 10/10
recall at 10 keys, 20/20 at 20 keys, and 15/15 after adding 5 keys to an
existing 10-key adapter and retraining on all 15.

A probe of five keys the 10-key adapter was never trained on returned, for
every one, the last trained key's question and answer followed by stray
tokens. The fingerprint registry holds no entry for an untrained key and
rejected all five at confidence 0.000, so none was passed on as a recalled
fact.

### SimHash's separation margin across tests

SimHash (a compact content fingerprint, used to verify a recalled answer
against what was actually trained under that key) rejected two kinds of
mismatch seen across several tests. Test 5's source data lists two
questions twice, each under two keys with its answer worded two ways; in
one of those pairs the model answered one key with the other key's
wording, and SimHash rejected it at confidence 0.734. Test 4's case was a
different fact recalled under the wrong key, caught at confidence 0.688.
Test 7's wrong-persona content (one persona's adapter, asked for the other
persona's keys, answering with its own persona's facts) scored up to
0.703. Against the 0.75 acceptance threshold, the observed margin for
these cases is 0.016–0.062 — a real but thin separation.

### Training cost and adapter storage footprint

**Training time is linear in key count.** Test 1's recall-verified run
measured 41.9–43.7 s/key on Gemma (10 to 100 keys) and 29.2–30.4 s/key on
Mistral (10 to 54 keys — Mistral's own extraction ceiling; RTX 5070,
question/answer format). Test 8 measured 0.78–0.80 min/key of total cycle
time at its largest scale (489–550 keys, question/answer format, Mistral);
training alone, excluding the rest of the cycle, ran 0.71–0.73 min/key
over that range.

**Adapter storage is constant in fact count (Test 6).** The final adapter
weights alone — 36 MB on Gemma, 27 MB on Mistral — stayed the same size
from 10 to 100 trained facts. A comparison retrieval-augmented generation
(RAG) setup came to about 92 MB: a 92 MB sentence-embedding model plus a
0.17 MB index at 100 facts, so the fixed cost of the embedding model
dominates the total rather than the index itself. RTX 5070, question/answer
format, 2026-03-21.

The same run also compared keyed-probe recall against RAG's top-3-retrieval
recall on the same fact set: the keyed probe reached 99/100 on both models;
RAG reached 78/100 on Gemma and 96/100 on Mistral. The two conditions are
not a clean head-to-head — the keyed probe reads a single trained fact by
its own key, while RAG retrieves and answers from the top 3 matches to the
question — so the gap reflects that asymmetry as much as the underlying
recall mechanism.

---

## Natural-language access

**What the data shows.** A question/answer adapter answers the exact
question it was trained on almost every time, and rewordings of it less
often the further the wording moves from what was trained. Open-ended
probes with no specific wording to anchor to surface a partial,
probe-dependent slice of the stored facts — 22 of 50 (Gemma) to 32 of 50
(Mistral) in one broad-probe test, 25–37% of the stored facts, asked one
entity at a time, in a narrower probe at larger scale. A triple adapter,
trained on no natural-question form at all, answers un-keyed natural
language at about the un-tuned base model's own level — measured under a
"use only the facts provided" system prompt with no facts actually
supplied, so treat that comparison as a floor, not a clean read of the
triple adapter alone. The keyed-recall prompt is the one interface that
works reliably under both formats.

### What a trained question recalls without the key (Tests 5, 9)

The question/answer format trains each fact's plain question as a second
example alongside the keyed prompt, so asking that same question directly
recalls a trained example rather than demonstrating open-ended
generalization.

**Test 5** (Gemma 2 9B and Mistral 7B, 50 facts, question/answer format,
2026-03-21, PerLTQA's question/answer pairs loaded directly, extraction
skipped): the keyed prompt recalled 49/50; the trained question asked
directly, with no key, recalled 50/50; open-ended prompts ("List
everything you know about the user.") surfaced 22 of 50 unique facts on
Gemma and 32 of 50 on Mistral. The source data lists two questions twice —
each time the same question under two keys, with the answer worded two
ways. The one keyed miss is one such pair: the model answered one key with
the fact's other wording (the one trained under its duplicate key), and
SimHash rejected that cross-key wording at confidence 0.734. Each
duplicate pair returns the other key's trained wording as a match in the
per-question 50/50 figure, rather than showing open-ended generalization.

**Test 9** (Mistral 7B, 41 checkpoints from 21 to 550 keys, question/answer
format, completed 2026-04-08, probing the adapters from
[Test 8](#550-keys-across-56-consolidation-cycles-test-8)):
the keyed prompt recalled 100% at every checkpoint; the trained question
asked directly recalled 95.2–100%, with no trend by key count; open-ended
"what do you know about {entity}" probes recovered 25–37% of the stored
facts, asked one entity at a time, and the share of entities with at least
one correct fact rose from 37.5% at the first checkpoint to a peak of
71.4% around 108 keys, then settled to 47–52% at the larger scales. One
checkpoint, cycle 13, scored lower on both measures (13.6% and 35.7%) — a
scoring artifact from that cycle's terser refusals, which earned less
word-overlap credit, not a drop in what the adapter knew: keyed and direct
recall were unaffected at the same checkpoint.

### Rephrasing distance from the trained question (Tests 10, 10b)

**Test 10** (Mistral 7B, 129 keys, question/answer format, up to 1,710
training epochs, 2026-04-03 to 2026-04-25): the trained question, asked
directly, held at 91.5–93.0% from the earliest checkpoints onward. A
passive-voice rephrasing of the same question recalled 60.5–77.5%, with no
trend across the run.

**Test 10b** (Test 10's adapters — Mistral 7B, question/answer format —
645 questions across 5 rephrasing styles, 24 checkpoints from epoch 30 to
720, 2026-04-05 to 2026-04-08): scored by exact entity match, mean
accuracy by style was indirect ("I was wondering...") 89.5%, contextual (a
brief topical lead-in) 67.4%, partial (a different angle on the same fact)
56.9%, formal (academic phrasing) 56.1%, and colloquial (casual language)
44.7% — though formal scores above partial at 8 of the 24 checkpoints and
ties it at 4, so that pair's ranking is not stable.
Scoring the same answers with an LLM judge (the same base model, judging
its own adapter-on answers) instead of exact match added 2.9–8.1
percentage points (mean 5.8) to the aggregate score across checkpoints;
the per-style uplift from switching to the judge ranged more widely, from
1.5 points (indirect) to 11.5 points (colloquial). Results were stable
from epoch 60 onward — more training did not change the overall ranking.
The indirect style, the highest scorer, is also the closest in wording to
the trained question among the five styles, so the ranking largely tracks
distance from the trained phrasing rather than a distinct "conversational"
quality.

### An interactive observation on untrained questions

An interactive probe at 140 trained keys (Mistral 7B, question/answer
format, one of the [Test 8](#550-keys-across-56-consolidation-cycles-test-8)
cycle checkpoints, 2026-03-24) found the keyed prompt recalling 140/140,
while questions outside the trained set produced hallucinated answers.
This was an interactive session with no saved result file — reported as a
qualitative observation, not a measured result.

### Natural-language recall under the triple format (Test 17)

Fifteen natural-language questions over a personal 95-fact graph, no key,
with the adapter active, answered no better than the same questions with
the adapter disabled — an automatic count scored 3 of 15 correct in both
conditions alike; a reading of the saved answers finds one correct answer
with the adapter on and none with it off. The keyed prompt over the same
facts recalled 15/15. Mistral 7B, triple format, 2026-05-11. This
condition used a system prompt telling the model to answer only from
facts provided and say so if it did not know — with no facts supplied.
With the adapter on, about half the answers declined and most of the rest
stated specifics the graph does not hold, so the adapter did not surface
its facts even when it answered; treat this as a lower bound under that
prompt, not a clean measurement of what the triple adapter can do
unprompted. The triple format trains no
natural-question example at all, by design — the keyed prompt is its one
retrieval interface.

---

## Reasoning over recalled facts

### Adapter on vs adapter off over identical recalled facts (Test 3)

What it tests: whether reasoning quality changes when the adapter that
supplied the recalled facts stays active during reasoning, versus when it
is switched off — production reasons with the adapter off.

Gemma 2 9B and Mistral 7B, 50 facts distilled from PerLTQA, question/answer
format, RTX 5070, 2026-03-20. Two conditions reasoned over the same 50
facts in context — "PM Recall+Reason" (facts recalled from the adapter,
which stays active) and "RAG all facts" (the original facts from a store,
adapter off) — with an identical prompt and system prompt. Fact
reconstruction was exact on both models before either condition ran
(50/50 facts, confidence 1.000), so the two conditions reasoned over the
same facts; the only difference measured was whether the adapter stayed
on.

**Result: parity.** Gemma scored 0.687 (adapter on) vs 0.679 (adapter off)
by embedding similarity to a reference answer; Mistral scored 0.566 vs
0.525; N=14 questions, single run — within the noise of a run this size.
Adapter-on answers were shorter (mean 15 vs 21 output tokens on Gemma, 17
vs 24 on Mistral). A diagnostic third condition, asking the adapter
directly with no facts in the prompt, and under the training system
prompt rather than the context one, scored lower (0.633 / 0.446) —
consistent with the recalled context doing real work. The test protocol
reconstructed all facts once per run (162 s Gemma, 121 s Mistral) and
reused that context across the 14 questions, rather than rebuilding it per
query.

Scale caveat: measured at 50 facts, about 1,000–1,250 tokens of context.

Two qualitative, interactive observations (Mistral 7B, question/answer
format, 2026-03-24, no saved result): at 50 recalled facts, an adapter-on answer
read as terse and correct against a more detailed adapter-off answer that
cited the recalled evidence explicitly — a direct on/off comparison, at
the same scale as the quantitative result above. At 140 recalled facts,
only adapter-off reasoning was observed (no adapter-on comparison at that
scale): three multi-fact questions were answered correctly with cited
evidence. Neither observation is a quality measurement — the quantitative
comparison above found parity, not a richness gap, on the metric it used.

### Two context shapes, question/answer vs triple (Test 17)

Fifteen questions over the same 93 facts from a personal 95-fact graph —
stored facts standing in for recall, no adapter loaded — comparing how the
facts are rendered into context: a bare answer line (the question/answer
format's rendering) vs a subject–predicate–object line (the triple
format's rendering). Overall correctness was close either way (12/15 vs
11/15 by a coarse heuristic, within noise). On one question anchored to a
specific fact's subject, the bare answer line lost the subject the
question asks about while the triple line kept it (an invented example
of the same shape: "— March 2019" against "— Alex's contract end date
March 2019"); the heuristic scored the triple rendering lower on two
other questions. Mistral 7B, question/answer and triple renderings,
2026-05-11.

### A LongMemEval negative result (Test 17)

Of the LongMemEval oracle-split questions the 227-session graph fully
covers (88, all either multi-session or temporal-reasoning questions), 40
were sampled (20 of each type). Feeding all 550 recalled triples as
context and judging the answer against LongMemEval's own reference
answers scored 8/40 correct by an LLM judge (the same Mistral 7B base
model; 5/40 by simple contains-the-answer matching); giving the model only
the triples from the evidence sessions for each question — the retrieval a
perfect memory would have made — scored 5/40 judged (7/40 contains-answer).
Most of those credits were not answers. With all 550 triples, six of the
judge's eight credits went to the model declining a question that had an
answer, and every contains-the-answer match was a refusal repeating an
option named in the question; the only genuine credits were two correct
refusals on questions LongMemEval marks unanswerable. Oracle retrieval
kept those two and added one answer that listed the right cuisines but
miscounted them. Oracle retrieval not helping shows the bottleneck is not
too much context: the pipeline that built this graph produced topical,
summarized triples rather than enumerable individual facts, so counting
and aggregation questions were refused or miscounted even with the right
facts retrieved; the triples also carry no date information, so ordering
and recency questions are unanswerable from them. The model invented
nothing: most misses were refusals, and the rest were counts built from an
incomplete set of facts. Mistral 7B, triple format, 2026-05-11.

Scope: half the sampled questions were temporal-reasoning, answered here
with no date context at all. Production inference groups recalled
facts by date by default; this probe did not exercise that path.

---

## Multi-session and persona

### A growing fact pool across 10 sessions (Test 4)

What it tests: whether recall holds as a fact pool grows session by
session, with a fresh adapter retrained on the full pool at each session.

Gemma 2 9B and Mistral 7B, question/answer format, RTX 5070, 2026-03-20: a
fresh adapter, retrained on the full cumulative fact pool at each of 10
sessions as the pool grew from 7 to 30 facts (extraction skipped —
facts loaded directly). Final recall was 30/30 on both models at
confidence 1.000. One miss occurred at the first session on Gemma (6/7): a
cross-key answer that SimHash rejected at confidence 0.688; every later
session recalled 100% on both models.

### Two personas on separate adapters (Test 7)

What it tests: whether two persona-specific adapters isolate their facts —
each persona's own adapter recalling its facts, with no leakage when
probed for the other persona's keys.

Gemma 2 9B and Mistral 7B, question/answer format, RTX 5070, 2026-03-19:
two PerLTQA characters, 50 facts each, trained onto separate adapters with
non-overlapping key ranges (extraction skipped). Each persona's own
adapter recalled 50/50; the first persona's adapter still recalled 50/50
after the second persona was trained; probing one persona's adapter with
the other persona's keys returned 0/50 leaked facts in both directions on
both models — each adapter answered with its own persona's content
instead of the other's, and SimHash rejected those wrong-persona answers
at confidence up to 0.703, a margin of about 0.05 under the 0.75 threshold.

---

## Continual learning

### Updating facts on a persistent adapter (Test 2b)

What it tests: whether a single adapter, retrained on its full current
fact set at every cycle, absorbs a contradicted fact and keeps everything
else intact.

Gemma 2 9B and Mistral 7B, question/answer format, 16 keys (10 fact chains
plus 3 control facts, each control counted under two keys), 2026-03-20:
current facts and control facts held at 16/16 with exact answers across
all 8 cycles, both before and after all ten chains were updated to a
second version. Of those ten, the five whose old answer resembled no
current answer were probed as stale keys and returned the new content in
every case (5/5). Keys are renumbered from scratch at every retrain in
this test's script, so a changed fact's old key number simply carried its
new content at the next cycle: returning the new content from the first
cycle after the update (5/5) reflects that renumbering, not evidence about
how fast a stable key forgets its old content. Measured at 16 keys only.

### Full replay is required (Test 4b)

What it tests: whether new keys can be trained onto an existing adapter
without retraining the facts already there.

Gemma 2 9B and Mistral 7B, question/answer format, PerLTQA data,
2026-03-21: starting from 20 baseline keys, 5 cycles each trained 5 new
keys alone on the existing adapter. Each cycle was a series of separate,
one-epoch trainer runs — each epoch started its own trainer with fresh
optimizer state rather than continuing one run — stopping once two
consecutive epochs scored the new keys fully correct (5 to 26 epochs per
cycle in total), followed by one full retrain on all 45 keys together. The
new keys reached 5/5 every cycle (first fully correct between epoch 4 and
19). The old keys were lost without replay: Mistral recalled 0 of the
20–40 old keys after every cycle; Gemma recalled 0–4 of them, and 1 of 40
after the last cycle. The full-replay retrain recovered 45/45 on
Mistral and 44/45 on Gemma. **Training new keys onto an existing adapter
without full replay is not viable at rank 8.**

A development-scale run bears on the epoch budget a full-replay retrain
needs (Qwen 2.5 3B base, question/answer format, 2026-03-21, one run per
budget). An adapter trained on 10 keys for 30 epochs was retrained on 10
keys — 5 of its original keys plus 5 new ones. With 15 epochs all 5
original keys were recalled, but 3 of the 5 new keys came back with
another nearby key's fact and were rejected by the fingerprint check
(7/10); with 30 epochs all 10 were recalled. The other 5 original keys,
trained onto a second adapter, recalled 5/5 in both runs. A single
observation, not a general rule.

### Forgetting under a partial overwrite is recoverable (Tests 13b, 15, 16)

What this line of results tests: when a subset of keys on a shared adapter
is retrained without the rest (an overwrite, without full replay), what
happens to the untouched keys, and whether it can be corrected. This is a
mechanism study, and it explains why the system relies on full replay:
every adapter a consolidation fold (a run of the background training cycle
that absorbs new facts into the adapters) trains is retrained on all of
that adapter's keys, and nothing in production runs a partial overwrite or
this repair loop.

**The retention curve and its recovery (Test 13b, n=1, Mistral 7B,
question/answer format, 2026-04-23).** Continuing training on 40 filled
keys from a scaffolded adapter while probing 160 untouched keys every
epoch: retention on the untouched keys fell from 0.994 at epoch 0 to 0.775
by epoch 2 — twelve epochs before the 40 filled keys themselves stabilized
at epoch 14 — continued down to a low of 0.300 at epoch 12, then recovered
mildly to 0.394 by epoch 30. Stopping training as soon as the fill
converges does not protect the untouched keys; by that point the damage is
already done. A follow-up probe on the same final adapter — one pass of 2
epochs at a tenfold-lower learning rate (1e-5), training only on the 97
keys that were failing — recovered 95 of the 97 (97.9%) at a cost of 1 of
the 63 still-passing keys (1.6% collateral), raising overall retention
from 63/160 (0.394) to 157/160 (0.981) in 3.8 minutes. Weight-space
measurements on the same adapter show the size of the weight change
reaching about 95% of its final value by epoch 8, six epochs before the
fill converges, with its effective rank near 6.3 throughout, and the
direction of successive training updates settling into a stable pattern
around epoch 17 — consistent with what looks like
forgetting being mostly a shift in how the adapter decodes its stored
keys, not an erasure of the keys themselves. n=1, single model and
dataset.

**Multi-seed confirmation of the repair (Test 15, n=5 seeds — 42, 7, 1337,
1, 11 — Mistral 7B, question/answer format, 100 keys with 20 overwritten,
2026-05-07 to 2026-05-11).** A related repair recipe — up to 5 separate one-epoch
episodes at LR=1e-5, rather than 13b's single 2-epoch pass — run across 5
seeds on both a no-scaffold overwrite arm and a scaffold-then-fill arm,
confirmed the mechanism at a smaller scale than Test 13's 200-key, 40-
overwritten design: mean retention before repair was 0.045 (no scaffold)
and 0.16 (scaffold); after repair, 0.915 and 0.95 — the two arms become
statistically indistinguishable once repaired (ratio 1.04, bootstrap lower
confidence bound 0.98). All ten repair runs used the full 5-episode budget,
reached 84–99% retention before stopping, and lost no previously-passing
key (zero collateral loss in every run).

**A fuller repair sweep (Test 16, n=5 seeds, Mistral 7B NF4, triple format
on a LongMemEval-derived graph, 2026-05-16 to 2026-05-19).** Sweeping
repair learning rate (1e-5 / 2e-5 / 5e-5), epochs per repair episode (1 or
3), and how many extra epochs of original training ran before the
overwrite (0 / 10 / 30 past the point the keys first trained fully
correct): every cell using 3 epochs per repair episode fully recovered all
38 untouched keys on all 5 seeds, with **zero collateral loss in all 95
runs (19 cells × 5 seeds)**, converging in a mean of 1.0–2.8 repair
episodes depending on the cell (2.0–2.8 at 1e-5, 1.0–1.6 at the two higher
rates). Before repair, the overwrite itself was completely
learned in every case (100% recall of the new content) — so whatever part
of the overwrite is lost after repair was undone by the repair itself: at
2e-5 and 3 epochs per episode, with at least 10 extra epochs
of original training before the overwrite, repair kept only 28–47% of the
overwrite's intended content and brought back 12–18% of the original
answers. A gentler setting (1e-5, 3 epochs) still reached full recovery of
the untouched keys while keeping 63–82% of the overwrite. The most
aggressive setting measured (5e-5, 3 epochs) reverted the overwrite almost
entirely, resurfacing 37–47% of the original answers — useful when the
goal is to undo the change, harmful when the overwrite should persist.
Recovering the untouched keys and preserving the overwrite pull in
opposite directions; no cell measured got both fully for free. Scope: one
model, 50 keys with 12 overwritten, one overwrite fraction, a single
overwrite-then-repair step.

### Placeholder scaffolds before filling in real answers (Tests 13, 14, 15)

What it tests: whether pre-training an adapter with placeholder answers,
then later filling in the real content, converges faster and protects
untouched keys better than a plain overwrite of an already-trained
adapter — an alternative to full replay.

**Test 13** (n=1, Mistral 7B, question/answer format, 200 keys with 40
placeholder-then-filled, 2026-04-20 to 2026-04-22): filling the 40
placeholder keys converged faster than a plain overwrite of the same 40
keys on an already-trained adapter (fully correct from epoch 11 on vs
epoch 18), with zero placeholder leakage into the filled answers, and the
placeholder-carrying adapter converged on its 200 keys at the same rate as
one trained with no placeholders at all — the scaffold cost nothing at
initial training time. Test 13 also reported that the scaffold retained
more of the 160 untouched keys after filling than the plain overwrite did
(37.5% vs 5.6%, a 6.7× difference) — this retention advantage was tested
again at n=5 below and did not hold up.

**Test 15** (n=5 seeds, Mistral 7B, question/answer format, 100 keys with
20 overwritten, 2026-05-07 to 2026-05-11): the same scaffold-vs-overwrite
retention comparison, checked against a rule pre-registered before the run
(a ratio of at least 5.0, and a bootstrap lower confidence bound of at
least 2.5) measured a ratio of 3.56 with a lower bound of 0.76 — short of
both thresholds, and with the lower bound under 1, meaning the advantage
is not distinguishable from none at this sample size. The faster fill
held on average (mean stop epoch 17.4 vs 22.2; one seed of five reversed)
and every fill ended exactly correct; the retention advantage did not
hold.

**Test 14** (n=3 seeds × 3 content-free scaffold shapes, Mistral 7B,
question/answer format, filling 20 keys into a 100-key scaffolded adapter,
2026-04-26 to 2026-05-06): comparing three scaffold shapes that carry no
real question text (a per-slot placeholder, a templated question, and a
uniform sentinel) — none filled faster than another (first fully-correct
epoch 19.0 ± 2.5, 20.3 ± 2.6, 20.7 ± 1.9 across the three shapes, ±
population standard deviation over the 3 seeds), with zero placeholder
leakage and
full final recall in all nine runs.

### Small folds: training budget and donor seeding (Test 20)

What it tests: whether a small fold (fewer than 128 keys) trained from a
freshly initialized adapter (no prior training) reliably reaches full
recall within its derived epoch budget, and whether seeding the target
adapter from a pre-trained "donor" checkpoint — rather than starting cold
— rescues folds that fail cold.

Mistral 7B NF4, triple format, rank 8, attention-only adapter shape,
2026-07-25 to 2026-07-27. Most arms load a real 21-key set (15 episodic +
6 procedural, all trained on that attention-only shape) drawn from a
private production fold; the exact key content is not reproducible from
the repository, only the condition names, key counts and recall rates are
reported. The 3-key arms further below train three of those 21 facts
under new key numbers. The 30-epoch arms use the epoch budget derived for
128+ keys; the 50-epoch arms use the budget derived for 16–127 keys.

| Condition | Steps | Key set | Seeds | Exact-match / 21 | Rate |
|---|---|---|---|---|---|
| cold, 30 epochs | 330 | original key numbers | 42 / 0 / 1 / 2 | 17 / 16 / 14 / 19 | mean **0.786** |
| donor-seeded, 30 epochs | 330 | original key numbers | 42 / 0 / 1 / 2 | 21 / 21 / 21 / 21 | 1.000 (all 4 seeds) |
| cold, 50 epochs | 550 | original key numbers | 42 / 0 / 1 / 2 | 21 / 21 / 21 / 21 | 1.000 (all 4 seeds) |
| donor-seeded, 50 epochs | 550 | original key numbers | 42 / 0 / 1 / 2 | 21 / 21 / 21 / 21 | 1.000 (all 4 seeds) |
| cold, 30 epochs, keys shifted (three digits) | 330 | shifted, zero overlap | 42 / 0 | 21 / 18 | 1.000 / 0.857 |
| donor-seeded, 30 epochs, keys shifted (three digits) | 330 | shifted, zero overlap | 42 / 0 | 21 / 21 | 1.000 / 1.000 |
| cold, 30 epochs, keys shifted (four digits) | 330 | shifted, zero overlap | 42 / 0 | 14 / 16 | mean **0.714** |
| donor-seeded, 30 epochs, keys shifted (four digits) | 330 | shifted, zero overlap | 42 / 0 | 21 / 21 | 1.000 / 1.000 |

In every arm the learning rate decays linearly to zero over that arm's own
run (240, 330 or 550 steps). The donor's own 147-key synthetic population
reached 147/147 (confidence 1.000) after its own 30-epoch training.

**The negative result.** Cold training at 30 epochs (330 optimizer
steps) failed on the real 21-key set in 4 of 4 seeds — mean rate 0.786,
range 0.667–0.905. This was not a uniform failure: a same-budget run on
the same keys renumbered (so that no key number overlaps the donor's own
population) had one seed reach 1.000 while another partially failed at
0.857 — cold failure at this budget depends on the seed and the specific
key numerals trained, not on the key count alone.

**Donor seeding closes the gap at the 30-epoch budget** where cold does
not (1.000 on 4/4 seeds vs cold's 0.786 mean), and the same donor
checkpoint rescues a key set it never saw during its own training (zero
overlap between the donor's memorized keys and the target's) exactly as
well as it rescues the overlapping set — the uplift is not explained by
the donor already knowing these keys. The donor's whole population is
built on this same fold's pattern: a fictionalized copy of its 21 facts
followed by six further blocks of invented facts on the same predicates
in the same order, so rescuing a fold whose predicates the donor never
saw is not measured. Cold does reach 1.000 given a
larger budget (50 epochs); donor-seeding holds at 1.000 there too, at no
regression.

**Cold recall was lowest, at n=2 seeds, when the keys were renumbered into
four digits so they share one more leading digit** (0.714, against 0.929
for the three-digit shift and 0.786 for the original three-digit numbers
at the same two seeds) — suggestive, not a confirmed trend, of recall
softening as key numbers grow more alike. The same donor checkpoint, built
once and never rebuilt for these four-digit keys, still rescued that
harder case to 21/21 on both seeds measured.

**At 3 keys, the epoch budget derived for that bucket (80 epochs) was
enough on its own, with no donor needed:** cold and donor-seeded both
reached 3/3 on all 4 seeds, at zero overlap with the donor's own keys.

A separate, earlier cold run at 30 epochs on a different, also private
3-key set (a different code revision, different key numbers, 2026-07-12) scored 3/3,
1/3 and 2/3 across three seeds; these recall rates cannot be reproduced
from the repository. Starting instead from an already-trained adapter
that recalled none of the 3 keys before training (0/3, a warm start), and
a run on 3 synthetic keys, both reached 3/3 on every seed. It shows a
fixed 30-epoch budget can fail below 16 keys; the 3-key arms above measure
that bucket at its own derived 80-epoch budget.

**Donor build cost.** Building the donor's 147-key population (30 epochs,
gradient accumulation 2) at the attention-only topology took about 38
minutes (2220 optimizer steps, 2270 s, 1.02 s/step). At the procedural
tier's attention-plus-MLP topology, the same build took about 45.5 minutes
(2220 steps, 2727 s, 1.2285 s/step) with reserved VRAM peaking near 4.7
GiB — about 20% more per step for roughly three times the trainable
parameters, consistent with the frozen base model's forward/backward pass
dominating per-step cost rather than the adapter update itself. Each
combination of base model and adapter shape builds its donor once — twice
for the two shipped shapes — and later folds reuse it; only a changed
donor recipe or a checkpoint that fails its integrity check forces a
rebuild.

---

## Generalization boundaries

### No grokking through 1,710 epochs (Test 10)

What it tests: whether training a rank-8 adapter far past the point it has
memorized its facts produces delayed emergence of multi-hop reasoning
("grokking") — composing individually trained facts into a 3-hop answer,
rather than only answering single facts directly.

Mistral 7B, question/answer format, 129 facts (question/answer pairs built
from 129 triples of a consolidated knowledge graph); 360 three-hop
evaluation questions were built from paths through those facts, and none
of the three-hop paths was itself trained; constant learning rate, weight
decay 0.1, 2026-04-03 to 2026-04-25 (57 training cycles, 1,710 epochs
total). Each 30-epoch cycle was a separate trainer run continuing from
the saved adapter, so optimizer state restarted every 30 epochs.

Keyed recall first reached 100% at the epoch-90 probe (probed every 30
epochs) and stayed at 93.0–100% thereafter (100% at the final,
1,710-epoch checkpoint); the trained question answered directly held at
91.5–93.0% from the earliest checkpoints. Three-hop compositional
accuracy oscillated between 3.3% and
17.8% throughout the run with no upward trend (13.1% at the final
checkpoint). A shortcut baseline — recalling any single fact that matches
the target relation, with no chain required — stayed strictly above
three-hop accuracy at every one of the 57 checkpoints, by 8.1 to 42.8
percentage points, and not one of the 360 three-hop questions was ever
answered as a genuine chain at any checkpoint (0 of 20,520
checkpoint–question pairs across the whole run). **No delayed
compositional generalization appeared through 1,710 epochs.**

This is a bounded negative, not a general claim that LoRA adapters cannot
grok: published grokking of multi-hop composition trains some multi-hop
facts alongside the single facts, and generalizes faster the larger that
trained share (Wang, Yue, Su and Sun, "Grokked Transformers are Implicit
Reasoners", NeurIPS 2024, [arXiv:2405.15071](https://arxiv.org/abs/2405.15071);
Abramov, Steinbauer and Kasneci, "Grokking in the Wild", ICML 2025,
[arXiv:2504.20752](https://arxiv.org/abs/2504.20752)) — this run trained
single facts only, so it does not test that setting; the adapter was rank
8 only; and in earlier grokking studies of small transformers trained
from scratch, generalization took hold from about 50× (Wang et al., 2024)
to about 1,000× (Power, Burda, Edwards, Babuschkin and Misra, "Grokking:
Generalization Beyond Overfitting on Small Algorithmic Datasets", 2022,
[arXiv:2201.02177](https://arxiv.org/abs/2201.02177)) the steps needed to
fit the training set — this run ran for at least 19× the epochs keyed
recall needed to first reach 100%, and those multiples come from
different model classes and training setups.

---

## Multi-adapter

### Composition and merging fail; switching works (Test 7b)

What it tests: whether two independently trained persona adapters (from
[Test 7](#two-personas-on-separate-adapters-test-7)) can serve both
personas at once, without switching which one is active.

Gemma 2 9B and Mistral 7B, question/answer format, 50 keys per persona,
2026-03-21. Switching between adapters — one active at a time — recalled
50/50 on each persona, on both models. Running both adapters active
simultaneously (additive composition, both LoRA deltas applied in the same
forward pass) collapsed recall to 0/50 on both personas for Gemma and to
0/50 and 2/50 for Mistral. Linearly merging the two adapters' weights
(equal weight) produced the same collapse (0/50 and 1/50, both models).
**Neither composition nor merging preserves indexed-key recall; adapter
switching — one adapter active at a time — is the approach that works.**

---

## Pipeline

### Extraction with and without the memory adapter (Test 11)

What it tests: whether extraction is better with the memory adapter active
or with no adapter loaded at all.

Mistral 7B, 50 PerLTQA sessions from two characters, the April 2026
extraction prompt, 2026-04-06. The memory adapter used in the "with
adapter" condition was the question/answer-format
[Test 8](#550-keys-across-56-consolidation-cycles-test-8)
cycle-50 adapter (528 keys), loaded and active (not mounted-and-disabled);
both conditions ran with the same 2048-token output budget. Extracting
with no adapter loaded at all produced the same 94% success rate as
extracting with the adapter active, but extracted more triples per
session (15.2 vs 12.4), with higher entity grounding against the source
transcript (98% vs 92%) and higher triple grounding (69% vs 62%); only
2.0% of the extracted triples overlapped between the two conditions — the
adapter changes what gets extracted, not only how much. Caveat: neither
condition here matches production's own way of running extraction without
the adapter's influence, which keeps the adapter mounted and switches it
off in place rather than never loading it — the two are not confirmed to
produce identical output.

### Predicate-synonym normalization

What it tests: the accuracy of the consolidation pass that collapses
synonymous predicates sharing a subject and object (e.g. folding "likes"
and "enjoys" together when both apply to the same pair), one model call
per candidate (subject, object) group.

Measured on one full consolidation fold, 2026-07-06: 23 candidate
(subject, object) groups were examined, and 19 were collapsed across 21
edges. Two were clear over-merges on fictional or generic objects (folding
"owns" into "has pet"; folding "spends time on" into "plays on schedule"),
plus one borderline case ("best friend" folded into "is friends with",
losing intensity); every other merge was a clean synonym collapse
(skills, languages, country of residence, dates). Precision on this fold:
approximately 0.89 (17 of 19).

This is one small fold, not a benchmark: its figures survive only in a
note written at the time, and neither the model engine that ran the pass
nor whether graph enrichment ran on that fold was recorded. In the
measured pipeline, normalization ran before enrichment whenever
enrichment ran; the shipped
pipeline runs enrichment first (off by default), so with enrichment on,
the pass sees predicates this measurement may not have seen.

### Cloud-egress anonymizer: detector choice

Before a household's words leave the house for a cloud reasoning model, a
resident local language model marks each value that is an instance of one
of the operator's configured kinds (a person's name, a phone number, an
email address, and so on) with that kind's keyword, one marking call per
conversational turn plus one further call whenever the outgoing payload
carries facts (a session's own facts alongside its transcript, or, on the
knowledge graph's cross-session pass, facts with no transcript at all),
plus, when a marked name occurs in a known speaker's conversation, one
call asking whether that name is the speaker introducing themselves. Code,
never the model, then decides what becomes
a placeholder and what leaves the house as written. See
[ARCHITECTURE.md → Cloud-Egress Anonymizer: Marking Step](ARCHITECTURE.md#cloud-egress-anonymizer-marking-step)
for the design and [SECURITY.md → Known limitations](SECURITY.md#known-limitations)
for what this does and does not protect against.

**Test set and scoring.** A fictional 240-turn test set — commands,
kinship questions, name mentions, contact-detail exchanges, place and
organisation mentions, assistant replies, and mixed turns — spans English
(110), German (60), French (40) and Spanish (30), 48 of them written
lowercase throughout. It carries 164 in-scope gold values (119 names, 19
phone numbers, 15 email addresses, 11 street addresses) and 208 out-of-
scope values a detector must not catch. A further 27 longer entries extend
the set to 267: 12 multi-turn transcripts, 4 self-introductions, 7 entries
pairing a word against its own look-alike (a name against a common noun or
place, or the reverse), 2 dense contact lists, one planted long document,
and one extracted-fact list. Scoring columns, in plain words: **names** —
gold personal names caught in full; **lowercase** — the same, on the
lowercase-only turns; **contact** — phone numbers, emails and addresses
together; **precision** — correct scrubs over every value a detector
scrubbed; **harmless words scrubbed** — values scrubbed that cover no gold
value (over-scrubbing, never a leak); **wrong kind** — an out-of-scope
value scrubbed as if it were in-scope; **partial** — a gold value only
partly covered. Measured on an RTX 5070 8 GB laptop GPU under WSL2,
Mistral 7B Instruct v0.3 NF4 on the GPU, CPU detectors at 8 threads.

**Detectors compared on the identical 240 turns, one scorer** (detector
outputs of 2026-09-05; the production chain's own run of 2026-09-08).

| Detector | Call shape | Names | Lowercase | Contact | Precision | Harmless scrubbed | Wrong kind | Partial |
|---|---|---|---|---|---|---|---|---|
| Resident 7B, generic prompt | per turn | 61.0% | 15.0% | 43.2% | 74.7% | 43 | 4 | 25 |
| Resident 7B, generic prompt | batches of 20 | 73.1% | 45.0% | 62.2% | 93.7% | 7 | 1 | 7 |
| Resident 7B, production chain | per turn | 92.4% | 65.0% | 93.3% | 68.8% | 53 | 15 | 2 |
| Opus 5, Sonnet 5, Fable 5.1, Gemini 3.8 Flash (four identical) | batches of 20 | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 0 | 0 |
| GPT-OSS 120B | batches of 20 | 95.0% | 75.0% | 100.0% | 98.1% | 0 | 3 | 0 |
| Span tagger, operator-matched labels (0.4) | per turn | 95.8% | 95.0% | 100.0% | 62.7% | 64 | 27 | 1 |
| Span tagger, every label (0.7) | per turn | 95.0% | 90.0% | 95.6% | 78.2% | 32 | 10 | 2 |
| Span tagger, every label (0.7) | batches of 20 | 81.5% | 70.0% | 88.9% | 88.5% | 10 | 7 | 0 |
| SauerkrautLM-GLiNER | per turn | 95.8% | 75.0% | 88.9% | 78.1% | 31 | 12 | 5 |
| E3-JSI GLiNER | per turn | 90.8% | 80.0% | 100.0% | 72.9% | 46 | 9 | 1 |
| OpenMed privacy filter | per turn | 87.4% | 35.0% | 88.9% | 78.7% | 6 | 34 | 6 |
| Presidio with spaCy (large) | per turn | 84.0% | 75.0% | 75.6% | 82.7% | 5 | 22 | 1 |
| Supervised name models (Flair, mBERT, XLM-R, WikiNEuRal), range | per turn | 71.4–82.4% | 0.0–25.0% | 0.0% | 81.8–85.3% | 0–1 | 16–18 | 0–5 |
| piiranha | per turn | 29.4% | 0.0% | 42.2% | 88.5% | 1 | 8 | 16 |

The frontier cloud rows and the "generic prompt" resident-7B rows share
one exploratory prompt used in the detector comparison, not the wording
that ships; it explicitly instructs the model never to tag pronouns or
kinship words, which is a likely reason the cloud rows show zero harmless
scrubs. The cloud rows are a capability reference, not a deployable
detector — a cloud detector's own provider would read the raw household
text before anything is marked, defeating the anonymizer's purpose. The
span tagger (`urchade/gliner_multi_pii-v1`) takes a fixed label set, never
a prompt, so "the same wording" does not apply to it, and it, like every
detector but the production chain, was not run on the 27 longer entries.
The speaker's own id token (the stand-in, such as `speaker1`, that
replaces the speaker's name, distinct from a placeholder for anyone
else's name) is dropped before scoring for every row, so no detector is
penalised for marking, or credited for not marking, that token.

Harmless words scrubbed, by class, for the rows the design choice turns on:

| Detector | Pronoun | Whole question | Possessive kinship/role | Bare kinship/role | Common noun | Other | Total |
|---|---|---|---|---|---|---|---|
| Span tagger, operator-matched labels | 6 | 7 | 36 | 10 | 4 | 1 | 64 |
| Span tagger, every label (0.7) | 2 | 1 | 21 | 5 | 3 | 0 | 32 |
| SauerkrautLM-GLiNER | 4 | 0 | 3 | 17 | 4 | 3 | 31 |
| Resident 7B, generic prompt, per turn | 0 | 0 | 17 | 23 | 3 | 0 | 43 |
| GPT-OSS 120B | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Resident 7B, production chain | 1 | 0 | 35 | 9 | 6 | 2 | 53 |

This class breakdown is a word-list classification of each detector's list
of harmless scrubs (on the 240 turns), made with a script outside the
shipped test tool rather than something the tool computes on its own.

**Latency and memory** (2026-09-05): the span tagger measured 0.059 s per
turn and 1717 MiB resident on the CPU, or 0.014 s and 2290 MiB on the GPU
(measured in its own process, so the figure includes that process's own
GPU runtime overhead). The resident 7B under the generic prompt measured
2.05 s median per turn (p95 6.2 s) at 4402 MiB resident.

**Why the resident language model.** A span tagger, offered a set of
labels, has no outcome for "none of these" — it files a person-like
phrase under whichever label scores nearest, so kinship phrases, pronouns
and (on some checkpoints) whole questions are scrubbed regardless of a
dedicated label for them; a second call that vetoes overlapping labels
cuts the over-scrubbing but takes name recall down with it (to 63.0%,
2026-09-05). A
tagger's labels also act as one joint prompt rather than independent
switches: narrowing the operator's list of kinds to only person-related
labels caused an email address to be tagged as a person instead, and
widening the list by two more labels pulled down a true phone number's
confidence score (both 2026-08-25) — a changed list of kinds is a different detector needing
its own re-measurement. On the identical resident-7B
weights, moving from the exploratory generic prompt (names caught
61.0%) to the design that shipped reached 92.4% on the same 240-turn set
and scorer, and 95.5% on the full 267-entry set. On a replay of 128
retained real household turns (2026-09-04), the span tagger also marked
device names, job titles and whole questions as persons; the shipped
resident-model design also marked two device
names as persons on the 267-entry set. What a scrub costs the cloud model's
understanding depends on which class it hits: a scrubbed kinship phrase
("my wife") still leaves a person to reason about and is restored once the
reply returns, while a scrubbed whole question leaves nothing — on the 240
turns, 44 of the shipped design's 53 harmless scrubs are kinship or role
phrases (the restorable case) against 1 pronoun and 0 whole questions. One
resident model doing the marking also keeps the cost to one model's
footprint: adding a second detector is not free either, before its
accuracy is even weighed — the span tagger's own footprint and latency
(above) are the added cost.

Read together: a language model's errors fall with capability — four of
the five cloud models checked (Opus, Sonnet, Fable, Gemini) made no errors
at all on this test set, under the generic prompt, not the shipped
wording, and the fifth (GPT-OSS) beat every resident-7B row except on
wrong-kind scrubs and had the highest precision of any other row, while
the span tagger and SauerkrautLM-GLiNER caught more names (95.8% vs
95.0%) and three tagger rows caught more lowercase names (80.0–95.0% vs
75.0%). The capability trend across these rows is an observation about
different models, not a measurement of the shipped resident 7B: on this
measured comparison, the tuned
span tagger at the 0.7 threshold is ahead of the resident 7B on every
column except partial matches (a tie), including over-scrubbing on the
240 turns (harmless scrubs 32 vs 53, wrong kind 10 vs 15, precision 78.2%
vs 68.8%) — the resident design is ahead only on pronouns plus whole
questions (1 vs 3). The cost of the resident-model choice is lower name
recall than the tuned tagger (92.4% vs 95.0%), weaker
lowercase-name recall, more over-scrubbing than a tuned tagger, the
per-entry processing time given below (which a CPU-resident tagger would
not pay), and no anonymizer at all while the base model is not loaded —
at which point, under an anonymizing policy, the cloud leg refuses rather
than send
unmarked text.

**Prompt layout and call shape** (2026-09-07). A marking prompt that ends
exactly where the model's answer begins keeps the model from continuing
its own worked examples: over 267 entries, a layout that did not end
there fabricated an example reply on 88, the shipped layout on none.
Marking one turn per call rather than a whole conversation catches a
multi-line address whole: on a shared 260-entry subset, contact recall
97.6% vs 77.4% and partial matches 3 vs 38, at a cost of lowercase-name
recall (65.0% vs 75.0%) and harmless scrubs (66 vs 54).

**Accepted scorecard.** The test tool's accepted scorecard reproduces a full
267-entry run first measured on 2026-09-08 and reproduced on 2026-09-13
with the same model reply on every entry: names caught 95.5%, lowercase
names 68.2%, contact details 97.6%, precision 77.2%, harmless words
scrubbed 67, wrong kind 15, partial 3, failed 2. A failed entry is
refused outright — nothing about it reaches the cloud leg. Recall
columns count completed entries only: the two failed entries — one
single turn and the set's one extracted-fact list (16 names, 8 contact
details) — were refused and fall outside those percentages, so marking
over a fact list has no scored result on this set. A value the model
marks with a keyword outside the operator's table is left as written.
Median processing time
was 1.45 s per entry on 2026-09-08 and 1.34 s on the 2026-09-13
reproduction. Both medians are over the full set, where 254 of the 267
entries carry no conversation history and take one marking call each; on
the chat door, every retained history turn (up to 10) is re-marked on
each turn under an anonymizing policy, so a turn deep in a conversation
can issue up
to 11 marking calls, not one.

Known limits, stated plainly: kinship and role phrases are scrubbed as if
they were a person's name — the cost is answer quality, not privacy,
since the phrase is restored in the reply; lowercase names are the
weakest column, measured on utterances written entirely in lowercase.
A value the model marks that does not match the text exactly as written
is not turned into a placeholder, so that occurrence can reach the cloud
leg unscrubbed. See [Anonymizer test tool](DEPLOYMENT.md#anonymizer-test-tool)
for the tool that checks every change to the prompt wording, the
operator's list of kinds, or the base model against this same scorecard.

### Early stopping

Training loss was compared against recall itself on one PerLTQA
character's data (Mistral 7B, rank 8, question/answer format, 25 and 50
keys, 30-epoch runs with a per-epoch recall probe inside a single trainer
run, 2026-03-23): a small per-step loss threshold (below 0.01) was first
crossed at epoch 8 (25 keys, 8% recall at the time) and epoch 15 (50 keys,
70% recall at the time), while recall only reached 100% at epoch 20 (25
keys) and epoch 19 (50 keys), and loss only stayed under the threshold
from epoch 19 / 18 onward — **loss convergence does not predict recall
convergence**, and a loss-based stop would have cut training well before
the keys were reliably recalled. Per-key confidence tracked recall closely
throughout (within 9.1 percentage points of the exact-match rate at every
epoch measured).

A recall-based early stop is available in production (optional, off by
default in the shipped configuration): where enabled, it probes recall
directly during training and stops once it holds at 100% for a window of
consecutive probes, using a window and probe cadence that differ from
both runs below. Filling 20 keys into an already-trained
100-key adapter (question/answer format, 3 scaffold shapes × 3 seeds = 9
runs, Mistral 7B, 2026-04-26 to 2026-05-06) — probing every epoch, with
the stop allowed from epoch 10 after 3 consecutive fully-correct probes —
stopped cleanly between epoch 18 and 26 in every
run, with full final recall in all nine; this measures a 20-key fill onto
an existing adapter, not a cold run, and is a fill-phase data point rather
than a like-for-like anchor for every fold size or format. Training 550
keys directly in the triple format
([Test 17](#exact-keyed-recall-of-550-triples-test-17), Mistral 7B,
2026-05-11), probing a fixed 100-key sample every epoch from epoch 20, was
exact on every probe and stopped after three, at epoch 22 of a 30-epoch
cap; its first probe came at epoch 20, so it does not show when recall
first became complete.

---

## Data and dead ends

### Data sources

**PerLTQA** (public dataset: [PerLTQA](https://github.com/Elvin-Yiming-Du/PerLTQA)) —
141 characters, most with about 20–25 dialogues each; each character
carries timestamped multi-turn dialogues, narrative event descriptions, a
structured profile and social relationships, and, for 32 characters,
ground-truth question/answer pairs (up to about 400 per character). Test 1
used the character Liang Xin, who has more dialogues than most: 30
dialogues, 485 turns, 394 ground-truth evaluation question/answer pairs.

**Synthetic data** — hand-built fact chains and reinforcement sessions used
by the archived Test 2b and Test 4 scripts.

**LongMemEval** (ICLR 2025) — a long-horizon conversational question-
answering benchmark, HuggingFace dataset `xiaowu0162/longmemeval-cleaned`,
oracle split (500 questions across 948 sessions), pinned revision
`98d7416c24c778c2fee6e6f3006e7a073259d48f`. [Test 17](#exact-keyed-recall-of-550-triples-test-17)'s
graph was built from 227 of those 948 sessions (seed 42), and the
[LongMemEval negative result](#a-longmemeval-negative-result-test-17)
sampled 40 of the 88 questions that graph fully covers.

### Archived dead-ends

Approaches explored and abandoned; preserved as research record. Sources:
`archive/README.md` and `archive/experiments/phase4_*.py`.

| Approach | Result | Evidence |
|---|---|---|
| XML `<memory key="...">` triple format | **0.0 F1 reconstruction** across all keys | Format collision between question/answer pairs and triple blocks caused cross-contamination. `archive/training/key_replay.py`, `archive/experiments/phase4_key_replay.py` |
| Entity-keyed natural language profiles | **Episodic recall regressed 59.3% → 36.1%** | Training signal dilution (broad profiles vs fact-specific question/answer pairs). `archive/training/entity_profile.py`, `archive/experiments/phase4_entity_replay.py` |
| LoRA rank sweep (8/4/2) on entity-replay | Rank is not the lever — confusion rate increased at lower ranks | `archive/experiments/phase4_rank_comparison.py` |
| Trained SimHash as JSON output field | 0/10 recall | Reported in the project paper (`paper/main.tex`, "Failed approaches to the enumeration problem" table); not reproducible from `archive/` |

See `archive/README.md` for full file list and context.

---

## Not measured

- Extraction yield and per-model extraction quality on the shipped
  extraction pipeline.
- Speaker-match thresholds on the shipped voice-embedding model
  (WeSpeaker) — the only measurement on record used a different embedding
  model, `pyannote/embedding`, and measured embedding scores, not
  thresholds.
- Inference-latency speedup from `preload_cache` on the shipped model —
  the only measurement on record used Qwen3-4B; the shipped default model
  is Mistral 7B.
- The adapter's capacity ceiling beyond the largest scale trained
  (550 keys, both training formats).
- Donor seeding of a fold whose predicates the donor's population never
  saw — the [Test 20](#small-folds-training-budget-and-donor-seeding-test-20)
  rescue ran on a fold whose predicate pattern, in the same order, makes
  up every block of the donor's own training population.
