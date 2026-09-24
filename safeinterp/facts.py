"""Paired fact probes: the same fact as a plain sentence and as a question.

The main source is CounterFact (Meng et al. 2022, the ROME paper), which has
~22k facts, each with a subject, a Wikidata relation and a cloze prompt such as
"The mother tongue of Danielle Darrieux is" -> "French".  For each relation we
add a question template, so every fact can also be asked in the GPT-2 paper's
"Q: ... A:" format.  A fact the model completes as a sentence but misses as a
question is one it *knows* but cannot *say* in that format.

Get CounterFact from https://rome.baulab.info/data/dsets/counterfact.json (a
JSON list) or any Hugging Face copy with the same fields.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .tasks import FACTS, QA_PREFIX


@dataclass
class Fact:
    id: str
    relation: str
    subject: str
    cloze: str  # ends right before the answer
    question: str  # without the "Q:"/"A:" framing
    answer: str  # with a leading space, GPT-2 style

    def qa_prompt(self, prefix: str = QA_PREFIX) -> str:
        return f"{prefix}Q: {self.question}\nA:"


# Wikidata relation -> question.  Relations without a template are skipped
# because a badly worded question would count as a model failure.
QUESTIONS = {
    "P17": "Which country is {} located in?",
    "P19": "Where was {} born?",
    "P20": "Where did {} die?",
    "P27": "Which country is {} a citizen of?",
    "P30": "Which continent is {} located on?",
    "P36": "What is the capital of {}?",
    "P37": "What is the official language of {}?",
    "P39": "What position did {} hold?",
    "P101": "What field does {} work in?",
    "P103": "What is the native language of {}?",
    "P106": "What is the occupation of {}?",
    "P108": "Who is the employer of {}?",
    "P127": "Who owns {}?",
    "P131": "Where is {} located?",
    "P136": "What genre of music does {} play?",
    "P138": "What is {} named after?",
    "P140": "What religion is {} affiliated with?",
    "P159": "Where is the headquarters of {}?",
    "P176": "Which company makes {}?",
    "P178": "Which company developed {}?",
    "P190": "What is a twin city of {}?",
    "P264": "Which record label is {} signed to?",
    "P276": "Where is {} located?",
    "P407": "What language is {} written in?",
    "P413": "What position does {} play?",
    "P449": "Which network originally aired {}?",
    "P463": "What organization is {} a member of?",
    "P495": "Which country is {} from?",
    "P641": "Which sport does {} play?",
    "P740": "Where was {} founded?",
    "P937": "Where did {} work?",
    "P1303": "Which instrument does {} play?",
    "P1412": "Which language does {} speak?",
}


def _space(s: str) -> str:
    return s if s.startswith(" ") else " " + s


def from_counterfact_record(rec: dict) -> Fact | None:
    rw = rec["requested_rewrite"]
    template = QUESTIONS.get(rw["relation_id"])
    if template is None:
        return None
    return Fact(
        id=str(rec.get("case_id", "")),
        relation=rw["relation_id"],
        subject=rw["subject"],
        cloze=rw["prompt"].format(rw["subject"]),
        question=template.format(rw["subject"]),
        answer=_space(rw["target_true"]["str"]),
    )


def load_counterfact(source: str, limit: int | None = None) -> list[Fact]:
    """Load from a local JSON/JSONL file, or a Hugging Face dataset id."""
    path = Path(source)
    if path.exists():
        text = path.read_text()
        records = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        from datasets import load_dataset

        records = load_dataset(source, split="train")
    facts = []
    for rec in records:
        fact = from_counterfact_record(rec)
        if fact is not None:
            facts.append(fact)
            if limit and len(facts) >= limit:
                break
    return facts


def builtin_facts() -> list[Fact]:
    """The 24 hand-written facts from ``tasks.py``, for quick offline runs."""
    facts = []
    for group, cloze, q, a in FACTS:
        # Questions carrying a partial answer ("A: William") don't fit the
        # plain Q/A frame, so they are skipped here.
        if " A: " in q:
            continue
        facts.append(Fact(id=group, relation="builtin", subject=group, cloze=cloze, question=q, answer=a))
    return facts
