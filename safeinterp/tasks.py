"""Zero-shot task probes modelled on Radford et al. (2019),
"Language Models are Unsupervised Multitask Learners" (the GPT-2 paper).

The paper's claim is that a plain LM picks up tasks from how the text is
*formatted*: "Q: ... A:" for question answering, "english = french" for
translation, "TL;DR:" for summarization, a long passage for LAMBADA, and so on.
Each probe here is one prompt in one of those formats, ending right where the
model must produce the answer.  ``answer`` is the gold continuation (with a
leading space, GPT-2 style) or ``None`` for open-ended tasks.

Factual probes come in two formats that share a ``group`` id: a bare sentence
completion (``fact_cloze``) and the paper's QA format (``fact_qa``).  The same
fact under both formats lets you separate features that carry the *knowledge*
from features that carry the *task format*.

These sets are small and hand-written, meant for feature discovery and not as
benchmarks.  Add your own with ``load_probes`` (JSONL).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Probe:
    task: str
    prompt: str
    answer: str | None = None
    group: str | None = None


# (group, cloze prompt, question, answer)
FACTS = [
    ("france_capital", "The capital of France is", "What is the capital of France?", " Paris"),
    ("japan_capital", "The capital of Japan is", "What is the capital of Japan?", " Tokyo"),
    ("italy_capital", "The capital of Italy is", "What is the capital of Italy?", " Rome"),
    ("germany_capital", "The capital of Germany is", "What is the capital of Germany?", " Berlin"),
    ("spain_capital", "The capital of Spain is", "What is the capital of Spain?", " Madrid"),
    ("russia_capital", "The capital of Russia is", "What is the capital of Russia?", " Moscow"),
    ("egypt_capital", "The capital of Egypt is", "What is the capital of Egypt?", " Cairo"),
    ("eiffel", "The Eiffel Tower is located in the city of", "In which city is the Eiffel Tower?", " Paris"),
    ("colosseum", "The Colosseum is located in the city of", "In which city is the Colosseum?", " Rome"),
    ("big_ben", "Big Ben is located in the city of", "In which city is Big Ben?", " London"),
    ("romeo", "Romeo and Juliet was written by William", "Who wrote Romeo and Juliet? A: William", " Shakespeare"),
    ("relativity", "Albert Einstein is famous for the theory of", "What theory is Albert Einstein famous for? A: The theory of", " relativity"),
    ("microsoft", "Microsoft was founded by Bill", "Who founded Microsoft? A: Bill", " Gates"),
    ("iphone", "The iPhone is made by the company", "Which company makes the iPhone?", " Apple"),
    ("jupiter", "The largest planet in the solar system is", "What is the largest planet in the solar system?", " Jupiter"),
    ("mars", "The planet known as the Red Planet is", "Which planet is known as the Red Planet?", " Mars"),
    ("brazil_lang", "The official language of Brazil is", "What language is spoken in Brazil?", " Portuguese"),
    ("japan_currency", "The currency of Japan is the", "What is the currency of Japan? A: The", " yen"),
    ("everest", "Mount Everest is located in the country of", "In which country is Mount Everest?", " Nepal"),
    ("obama", "Barack Obama was born in the state of", "In which state was Barack Obama born?", " Hawaii"),
    ("jordan", "Michael Jordan is famous for playing", "What sport did Michael Jordan play?", " basketball"),
    ("beatles", "The Beatles were a band from the city of", "Which city were the Beatles from?", " Liverpool"),
    ("gold_symbol", "The chemical symbol for gold is", "What is the chemical symbol for gold?", " Au"),
    ("sahara", "The Sahara desert is on the continent of", "On which continent is the Sahara desert?", " Africa"),
]

# Paper: "the model is conditioned on example question answer pairs".
QA_PREFIX = "Q: What is the tallest mountain in Africa?\nA: Kilimanjaro\n\nQ: Who painted the Mona Lisa?\nA: Leonardo da Vinci\n\n"

# Paper: "english sentence = french sentence" format with example pairs.
TRANSLATION_PREFIX = "good morning = bonjour\nthank you = merci\nthe man = l'homme\n"
TRANSLATIONS = [
    ("cat", " chat"), ("dog", " chien"), ("house", " maison"), ("water", " eau"),
    ("book", " livre"), ("red", " rouge"), ("car", " voiture"), ("bread", " pain"),
    ("cheese", " fromage"), ("sun", " soleil"), ("black", " noir"), ("white", " blanc"),
    ("apple", " pomme"), ("friend", " ami"), ("moon", " lune"), ("the woman", " la"),
]

# CoQA-style: answer is found in the passage.
READING = [
    ("Tom and his sister Anna went to the beach on Saturday. Tom built a sandcastle while Anna swam in the sea. Later they ate ice cream with their father, whose name is Peter.",
     ["Who built a sandcastle?", " Tom"], ["Who swam in the sea?", " Anna"], ["What is their father's name?", " Peter"]),
    ("The small village of Millbrook sits beside a river. Every autumn the villagers hold a festival where they sell apples and pumpkins. The mayor, Mrs. Clarke, opens the festival each year.",
     ["What is the name of the village?", " Mill"], ["Who opens the festival?", " Mrs"], ["What fruit do the villagers sell?", " apples"]),
    ("Dr. Patel works at a hospital in Chicago. She is a surgeon and has worked there for ten years. On weekends she likes to go hiking with her dog, Max.",
     ["In which city is the hospital?", " Chicago"], ["What is the dog's name?", " Max"], ["What is Dr. Patel's job?", " surgeon"]),
]

# LAMBADA-style: the final word is predictable only from the wider context.
LAMBADA = [
    ("She had kept the old violin in the attic for years. When her granddaughter asked to learn music, she climbed the stairs, dusted off the case, and handed her the", " violin"),
    ("Marcus had trained for the marathon all winter. On race day his legs ached at mile twenty, but he pushed on and finally crossed the finish line of the", " marathon"),
    ("The detective studied the muddy footprints by the window. They led across the garden to the shed, where he found the boots. The mud on them matched the", " footprints"),
    ("Every morning Lily fed the ducks at the pond near her house. One day the pond had frozen over, and she could not find a single", " duck"),
    ("Grandpa always said the secret to his soup was fresh basil from the garden. So when I finally made it myself, I went outside and picked some", " basil"),
    ("The captain ordered the crew to lower the sails as the storm approached. Waves crashed over the deck, and everyone held tightly to the ropes of the", " ship"),
    ("Jenna had lost her keys again. She searched her bag, her coat, and the kitchen drawer. Finally her brother pointed to the door, where she had left the", " keys"),
    ("The orchestra waited in silence. The conductor raised his baton, looked at the musicians, and with a single motion began the", " symphony"),
]

SUMMARIES = [
    "The city council voted on Tuesday to expand the bike lane network by forty miles over the next three years. Supporters said the plan would reduce traffic and pollution, while some business owners worried about losing parking spaces in front of their shops.",
    "Researchers at a university in Sweden have found that people who walk for thirty minutes a day sleep better than those who do not exercise. The study followed two thousand adults for a year and measured their sleep with wrist trackers.",
    "A small bakery in Ohio has become famous online after a customer posted a video of its giant cinnamon rolls. The owner said the shop now sells out every morning and has hired four new employees to keep up with demand.",
    "Heavy rain caused flooding across several towns in the region over the weekend. Hundreds of homes were evacuated and schools were closed on Monday. Officials said the water levels were expected to fall by Wednesday.",
    "The football team won its first championship in twenty years after beating its rivals two to one in the final. The winning goal came in the last minute of the match, and thousands of fans celebrated in the streets.",
]

# Ordinary web-like text, cut mid-sentence, with no task framing.  This is the
# baseline that task-selective features are measured against.
CONTROL = [
    "I went to the store yesterday and bought some",
    "The weather this weekend is supposed to be",
    "We spent most of the afternoon talking about",
    "If you are looking for a good place to eat, I would recommend",
    "The new update adds a few features that make it easier to",
    "After dinner we usually sit on the porch and",
    "Our team has been working hard on the project, and we",
    "My favorite thing about living in a small town is",
    "The meeting was moved to next week because",
    "There are a lot of reasons why people choose to",
    "She opened the window to let in some",
    "The best way to learn a new skill is to",
    "He looked at the menu for a long time before",
    "Last summer we drove across the country and",
    "Click the button below to",
    "The price of the tickets has gone up since",
    "It was late at night when the phone",
    "Many people think that the most important part of",
    "The garden looks beautiful in the spring when",
    "Please make sure to read the instructions before you",
]


def default_probes() -> list[Probe]:
    probes: list[Probe] = []
    for group, cloze, q, a in FACTS:
        probes.append(Probe("fact_cloze", cloze, a, group))
        # Some questions carry the start of the answer ("A: William") so the
        # gold next token is unambiguous.
        q, _, partial = q.partition(" A: ")
        qa = f"{QA_PREFIX}Q: {q}\nA:" + (f" {partial}" if partial else "")
        probes.append(Probe("fact_qa", qa, a, group))
    for en, fr in TRANSLATIONS:
        probes.append(Probe("translation", f"{TRANSLATION_PREFIX}{en} =", fr))
    for passage, *qas in READING:
        for q, a in qas:
            probes.append(Probe("reading_comp", f"{passage}\n\nQ: {q}\nA:", a))
    for ctx, a in LAMBADA:
        probes.append(Probe("lambada", ctx, a))
    for article in SUMMARIES:
        probes.append(Probe("summarization", f"{article}\nTL;DR:", None))
    for text in CONTROL:
        probes.append(Probe("control", text, None))
    return probes


def load_probes(path: str | Path) -> list[Probe]:
    """Read probes from JSONL with keys ``task``, ``prompt``, optional ``answer``/``group``."""
    return [Probe(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def save_probes(probes: list[Probe], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(asdict(p)) + "\n" for p in probes))
