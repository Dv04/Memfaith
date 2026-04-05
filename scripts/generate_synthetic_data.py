"""Generate larger synthetic FEVER and HotpotQA datasets for pipeline validation.

NOTE: These datasets are entirely synthetic / fabricated and exist only to
exercise the CCS pipeline end-to-end at a realistic scale.  They must NOT
be cited as empirical results.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
TOPICS = [
    ("Albert Einstein", "physicist", "developed the theory of relativity"),
    ("Marie Curie", "chemist", "discovered radium and polonium"),
    ("Isaac Newton", "mathematician", "formulated the laws of motion"),
    ("Charles Darwin", "naturalist", "proposed the theory of evolution"),
    ("Nikola Tesla", "inventor", "developed the alternating current system"),
    ("Ada Lovelace", "mathematician", "wrote the first computer algorithm"),
    ("Galileo Galilei", "astronomer", "improved the telescope"),
    ("Leonardo da Vinci", "polymath", "painted the Mona Lisa"),
    ("Alexander Fleming", "biologist", "discovered penicillin"),
    ("Rosalind Franklin", "chemist", "contributed to understanding DNA structure"),
    ("Alan Turing", "computer scientist", "broke the Enigma code"),
    ("Jane Goodall", "primatologist", "studied chimpanzee behavior"),
    ("Stephen Hawking", "physicist", "theorized about black holes"),
    ("Rachel Carson", "marine biologist", "wrote Silent Spring"),
    ("Werner Heisenberg", "physicist", "formulated the uncertainty principle"),
    ("Barbara McClintock", "geneticist", "discovered genetic transposition"),
    ("Linus Pauling", "chemist", "studied chemical bonding"),
    ("Richard Feynman", "physicist", "contributed to quantum electrodynamics"),
    ("Dorothy Hodgkin", "chemist", "determined structures of biochemical substances"),
    ("Max Planck", "physicist", "originated quantum theory"),
]

DISTRACTOR_POOL = [
    "The Eiffel Tower is located in Paris, France, and was built in 1889.",
    "The Amazon River is the largest river by discharge volume of water in the world.",
    "Mount Everest is the tallest mountain above sea level at 8,849 meters.",
    "The Great Wall of China stretches over 13,000 miles across northern China.",
    "The Sahara Desert is the largest hot desert in the world.",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "The International Space Station orbits Earth approximately every 90 minutes.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "DNA carries genetic instructions for the development of all living organisms.",
    "The periodic table organizes chemical elements by their atomic number.",
    "Tectonic plates float on the semi-fluid asthenosphere beneath the lithosphere.",
    "The human body contains approximately 206 bones in the adult skeleton.",
    "Jupiter is the largest planet in our solar system with a mass of 1.898e27 kg.",
    "The Renaissance period lasted from the 14th to the 17th century in Europe.",
    "Mitochondria are often called the powerhouse of the cell.",
    "The Mariana Trench reaches a depth of about 36,000 feet below sea level.",
    "Plate tectonics theory explains the movement of Earth's lithospheric plates.",
    "The circulatory system transports blood, nutrients, and gases throughout the body.",
    "Global average temperatures have risen by about 1.1 degrees Celsius since 1850.",
]


def _make_claim(topic: tuple[str, str, str], label: str) -> str:
    name, role, achievement = topic
    if label == "SUPPORTS":
        return f"{name} {achievement}."
    elif label == "REFUTES":
        return f"{name} never {achievement.replace('the ', 'any ')}."
    else:
        return f"{name} was involved in a classified government project."


def _make_evidence(topic: tuple[str, str, str]) -> str:
    name, role, achievement = topic
    return (
        f"{name} was a renowned {role}. "
        f"Historical records confirm that {name} {achievement}. "
        f"This contribution significantly advanced the field."
    )


def generate_fever_examples(n: int = 60) -> list[dict]:
    examples = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        label = FEVER_LABELS[i % len(FEVER_LABELS)]
        claim = _make_claim(topic, label)
        evidence_text = _make_evidence(topic)

        n_distractors = random.randint(3, 8)
        distractor_texts = random.sample(DISTRACTOR_POOL, min(n_distractors, len(DISTRACTOR_POOL)))

        evidence_segments = [
            {
                "segment_id": 0,
                "title": f"{topic[0]} - Evidence",
                "text": evidence_text,
                "is_gold": True,
                "source_type": "evidence",
            }
        ]
        distractor_segments = [
            {
                "segment_id": 1000 + j,
                "title": f"Distractor {j}",
                "text": text,
                "is_gold": False,
                "source_type": "distractor",
            }
            for j, text in enumerate(distractor_texts)
        ]

        missing_answer = "NOT_ENOUGH_INFO" if label != "NOT_ENOUGH_INFO" else "SUPPORTS"
        examples.append(
            {
                "dataset": "fever",
                "example_id": f"fever-synth-{i:04d}",
                "query": claim,
                "gold_answer": label,
                "task_type": "classification",
                "evidence_segments": evidence_segments,
                "distractor_segments": distractor_segments,
                "metadata": {
                    "required_segment_ids": [0],
                    "missing_evidence_answer": missing_answer,
                    "synthetic": True,
                },
            }
        )
    return examples


HOTPOT_QUESTIONS = [
    {
        "question": "Who developed the theory that the person who painted the Mona Lisa studied?",
        "answer": "Isaac Newton",
        "supporting_topics": [("Leonardo da Vinci", "polymath", "painted the Mona Lisa"),
                              ("Isaac Newton", "mathematician", "formulated the laws of motion")],
    },
    {
        "question": "What did the person who discovered radium also discover alongside it?",
        "answer": "polonium",
        "supporting_topics": [("Marie Curie", "chemist", "discovered radium and polonium")],
    },
    {
        "question": "Which field did the person who broke the Enigma code work in?",
        "answer": "computer science",
        "supporting_topics": [("Alan Turing", "computer scientist", "broke the Enigma code")],
    },
    {
        "question": "What book did the marine biologist who studied environmental pesticides write?",
        "answer": "Silent Spring",
        "supporting_topics": [("Rachel Carson", "marine biologist", "wrote Silent Spring")],
    },
    {
        "question": "Who formulated a principle about measurement uncertainty in quantum mechanics?",
        "answer": "Werner Heisenberg",
        "supporting_topics": [("Werner Heisenberg", "physicist", "formulated the uncertainty principle")],
    },
    {
        "question": "What did the inventor of the alternating current system develop?",
        "answer": "alternating current system",
        "supporting_topics": [("Nikola Tesla", "inventor", "developed the alternating current system")],
    },
    {
        "question": "Who studied the behavior of great apes in the wild?",
        "answer": "Jane Goodall",
        "supporting_topics": [("Jane Goodall", "primatologist", "studied chimpanzee behavior")],
    },
    {
        "question": "What theoretical objects did the physicist in a wheelchair study?",
        "answer": "black holes",
        "supporting_topics": [("Stephen Hawking", "physicist", "theorized about black holes")],
    },
    {
        "question": "What genetic phenomenon did Barbara McClintock discover?",
        "answer": "genetic transposition",
        "supporting_topics": [("Barbara McClintock", "geneticist", "discovered genetic transposition")],
    },
    {
        "question": "Who originated the foundational theory of quantum mechanics?",
        "answer": "Max Planck",
        "supporting_topics": [("Max Planck", "physicist", "originated quantum theory")],
    },
]


def generate_hotpotqa_examples(n: int = 50) -> list[dict]:
    examples = []
    for i in range(n):
        template = HOTPOT_QUESTIONS[i % len(HOTPOT_QUESTIONS)]
        question = template["question"]
        answer = template["answer"]

        evidence_segments = []
        required_ids = []
        for j, topic in enumerate(template["supporting_topics"]):
            seg_id = j
            evidence_segments.append(
                {
                    "segment_id": seg_id,
                    "title": f"{topic[0]} - Supporting",
                    "text": _make_evidence(topic),
                    "is_gold": True,
                    "source_type": "supporting_context",
                }
            )
            required_ids.append(seg_id)

        n_distractors = random.randint(4, 10)
        distractor_texts = random.sample(DISTRACTOR_POOL, min(n_distractors, len(DISTRACTOR_POOL)))
        distractor_segments = [
            {
                "segment_id": 1000 + j,
                "title": f"Context {j}",
                "text": text,
                "is_gold": False,
                "source_type": "context",
            }
            for j, text in enumerate(distractor_texts)
        ]

        variant_suffix = f"-v{i // len(HOTPOT_QUESTIONS)}" if i >= len(HOTPOT_QUESTIONS) else ""
        examples.append(
            {
                "dataset": "hotpotqa",
                "example_id": f"hotpot-synth-{i:04d}{variant_suffix}",
                "query": question,
                "gold_answer": answer,
                "task_type": "qa",
                "evidence_segments": evidence_segments,
                "distractor_segments": distractor_segments,
                "metadata": {
                    "required_segment_ids": required_ids,
                    "qa_fallback_answer": "unknown",
                    "synthetic": True,
                },
            }
        )
    return examples


def main() -> None:
    out_dir = Path("data/memfaith")
    out_dir.mkdir(parents=True, exist_ok=True)

    fever = generate_fever_examples(60)
    hotpot = generate_hotpotqa_examples(50)

    fever_path = out_dir / "fever_synthetic.jsonl"
    hotpot_path = out_dir / "hotpot_synthetic.jsonl"

    with fever_path.open("w", encoding="utf-8") as f:
        for ex in fever:
            f.write(json.dumps(ex) + "\n")

    with hotpot_path.open("w", encoding="utf-8") as f:
        for ex in hotpot:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(fever)} FEVER examples -> {fever_path}")
    print(f"Generated {len(hotpot)} HotpotQA examples -> {hotpot_path}")


if __name__ == "__main__":
    main()
