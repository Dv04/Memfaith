"""Counterfactual dataset generator for parametric-memory-free CCS evaluation.

Generates fictional entities, facts, and claims that no LLM could have seen
during pre-training.  This eliminates the Wikipedia contamination problem
where models answer from parametric memory instead of reading the context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, List, Optional, Tuple

from .distractor_retrieval import BM25Retriever
from .schemas import NormalizedExample, SourceSegment

# ── Phoneme tables for fictional name generation ─────────────────────────

_FIRST_SYLLABLES = [
    "tav", "vel", "ren", "ily", "om", "zar", "keth", "brin", "sor", "quel",
    "myr", "fal", "drev", "lux", "pol", "jev", "carn", "wex", "hol", "nev",
]

_MIDDLE_SYLLABLES = [
    "en", "ara", "ion", "eth", "an", "ori", "ax", "el", "is", "ova",
]

_LAST_PARTS = [
    "quist", "dann", "holm", "trel", "voss", "kern", "phen", "brek",
    "lund", "wick", "marr", "ford", "ghent", "stow", "rieth", "czek",
    "thorne", "feld", "laine", "graff",
]

_FIELD_ADJECTIVES = [
    "chromatic", "temporal", "resonant", "crystalline", "inverse",
    "subaquatic", "harmonic", "spectral", "entropic", "lattice",
    "dimensional", "catalytic", "recursive", "stochastic", "anisotropic",
]

_FIELD_DISCIPLINES = [
    "metallurgy", "thermology", "dynamics", "optics", "ecology",
    "topology", "synthesis", "acoustics", "geophysics", "biometry",
    "kinetics", "rheology", "photonics", "energetics", "mechanics",
]

_INSTITUTION_PREFIXES = [
    "Northwell", "Kessmer", "Valorian", "Braxthorn", "Ostvale",
    "Telmara", "Hexworth", "Quintera", "Duskhollow", "Vyrngate",
]

_INSTITUTION_SUFFIXES = [
    "Polytechnical Institute", "Research Foundation", "Academy of Sciences",
    "Centre for Advanced Studies", "Institute of Technology",
    "Collegium", "Laboratory Complex", "University", "Observatory", "Consortium",
]

_LOCATION_CITIES = [
    "Talmera", "Vrenneth", "Solkaris", "Dunhaven", "Plyrath",
    "Oxmeade", "Velquora", "Zinthari", "Brelmoor", "Quelside",
    "Fenmarch", "Holvaine", "Tarquessa", "Yendrath", "Crosshelm",
]

_LOCATION_COUNTRIES = [
    "Ostovia", "Brelland", "Varnheim", "Queloria", "Tasserin",
    "Dunmarch", "Solvania", "Hexland", "Pentharos", "Arkessa",
]

_DISCOVERY_NOUNS = [
    "equation", "principle", "theorem", "effect", "paradox",
    "resonance", "coefficient", "matrix", "oscillation", "constant",
    "conjecture", "criterion", "transform", "spectrum", "invariant",
]

_AWARD_NAMES = [
    "the Valorian Medal", "the Kessmer Prize", "the Braxthorn Award",
    "the Golden Lattice", "the Ostvale Fellowship", "the Quintera Distinction",
    "the Hexworth Laureate", "the Telmara Citation", "the Arkessa Honor",
    "the Solkaris Recognition",
]


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class FictionalEntity:
    entity_id: str
    name: str
    field: str
    institution: str
    birth_year: int
    birth_location: str
    discoveries: List[str]
    awards: List[str]


@dataclass
class FictionalFact:
    fact_id: str
    entity_id: str
    text: str
    fact_type: str
    linked_entity_ids: List[str] = field(default_factory=list)


# ── World Builder ────────────────────────────────────────────────────────

class FictionalWorldBuilder:
    """Generate a deterministic fictional world from a seed."""

    def __init__(self, *, seed: int = 42, n_entities: int = 20) -> None:
        self._rng = Random(seed)
        self._n_entities = n_entities
        self.entities: Dict[str, FictionalEntity] = {}
        self.facts: List[FictionalFact] = []
        self._build()

    def _pick(self, pool: list) -> str:
        return self._rng.choice(pool)

    def _generate_name(self) -> str:
        first = self._pick(_FIRST_SYLLABLES) + self._pick(_MIDDLE_SYLLABLES)
        last = self._pick(_LAST_PARTS)
        return f"{first.capitalize()} {last.capitalize()}"

    def _generate_field(self) -> str:
        return f"{self._pick(_FIELD_ADJECTIVES)} {self._pick(_FIELD_DISCIPLINES)}"

    def _generate_institution(self) -> str:
        return f"{self._pick(_INSTITUTION_PREFIXES)} {self._pick(_INSTITUTION_SUFFIXES)}"

    def _generate_location(self) -> str:
        return f"{self._pick(_LOCATION_CITIES)}, {self._pick(_LOCATION_COUNTRIES)}"

    def _generate_discovery(self, entity_name: str) -> str:
        last_name = entity_name.split()[-1]
        noun = self._pick(_DISCOVERY_NOUNS)
        return f"the {last_name} {noun}"

    def _build(self) -> None:
        used_names: set = set()
        entities_list: List[FictionalEntity] = []

        for i in range(self._n_entities):
            name = self._generate_name()
            while name in used_names:
                name = self._generate_name()
            used_names.add(name)

            n_discoveries = self._rng.randint(1, 2)
            discoveries = [self._generate_discovery(name) for _ in range(n_discoveries)]
            n_awards = self._rng.randint(0, 1)
            awards = [self._pick(_AWARD_NAMES) for _ in range(n_awards)]

            entity = FictionalEntity(
                entity_id=f"entity_{i}",
                name=name,
                field=self._generate_field(),
                institution=self._generate_institution(),
                birth_year=self._rng.randint(1820, 1980),
                birth_location=self._generate_location(),
                discoveries=discoveries,
                awards=awards,
            )
            self.entities[entity.entity_id] = entity
            entities_list.append(entity)

        fact_counter = 0
        for entity in entities_list:
            self.facts.append(FictionalFact(
                fact_id=f"fact_{fact_counter}",
                entity_id=entity.entity_id,
                text=f"{entity.name} was a renowned {entity.field} researcher at {entity.institution}.",
                fact_type="field",
            ))
            fact_counter += 1

            self.facts.append(FictionalFact(
                fact_id=f"fact_{fact_counter}",
                entity_id=entity.entity_id,
                text=f"{entity.name} was born in {entity.birth_year} in {entity.birth_location}.",
                fact_type="birth",
            ))
            fact_counter += 1

            for disc in entity.discoveries:
                self.facts.append(FictionalFact(
                    fact_id=f"fact_{fact_counter}",
                    entity_id=entity.entity_id,
                    text=f"{entity.name} discovered {disc}, which advanced the field of {entity.field}.",
                    fact_type="discovery",
                ))
                fact_counter += 1

            for award in entity.awards:
                self.facts.append(FictionalFact(
                    fact_id=f"fact_{fact_counter}",
                    entity_id=entity.entity_id,
                    text=f"{entity.name} received {award} for contributions to {entity.field}.",
                    fact_type="award",
                ))
                fact_counter += 1

        for i in range(0, len(entities_list) - 1, 2):
            a = entities_list[i]
            b = entities_list[i + 1]
            shared_inst = self._generate_institution()
            self.facts.append(FictionalFact(
                fact_id=f"fact_{fact_counter}",
                entity_id=a.entity_id,
                text=f"{a.name} and {b.name} both conducted research at {shared_inst}.",
                fact_type="shared_institution",
                linked_entity_ids=[b.entity_id],
            ))
            fact_counter += 1

    def get_entity_facts(self, entity_id: str) -> List[FictionalFact]:
        return [f for f in self.facts if f.entity_id == entity_id]

    def get_entity_pairs(self) -> List[Tuple[FictionalEntity, FictionalEntity]]:
        linked = [f for f in self.facts if f.linked_entity_ids]
        pairs = []
        for fact in linked:
            a = self.entities[fact.entity_id]
            b = self.entities[fact.linked_entity_ids[0]]
            pairs.append((a, b))
        return pairs

    def all_fact_texts(self) -> List[Dict[str, Any]]:
        return [{"title": f.entity_id, "text": f.text} for f in self.facts]


# ── FEVER Generator ──────────────────────────────────────────────────────

class CounterfactualFEVERGenerator:
    """Generate FEVER-style fact verification examples from a fictional world."""

    def __init__(self, world: FictionalWorldBuilder, *, seed: int = 42) -> None:
        self._world = world
        self._rng = Random(seed)

    def generate(
        self,
        n_examples: int = 60,
        n_distractors: int = 5,
    ) -> List[NormalizedExample]:
        entities = list(self._world.entities.values())
        self._rng.shuffle(entities)

        corpus = self._world.all_fact_texts()
        retriever = BM25Retriever(corpus)

        examples: List[NormalizedExample] = []
        example_id = 0

        for entity in entities:
            if len(examples) >= n_examples:
                break
            facts = self._world.get_entity_facts(entity.entity_id)
            if not facts:
                continue

            for fact in facts[:2]:
                if len(examples) >= n_examples:
                    break

                label_type = self._rng.choice(["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"])

                if label_type == "SUPPORTS":
                    claim = self._make_supports_claim(entity, fact)
                elif label_type == "REFUTES":
                    claim = self._make_refutes_claim(entity, fact)
                else:
                    claim = self._make_nei_claim(entity)

                evidence_seg = SourceSegment(
                    segment_id=0,
                    title=f"{entity.name} - Evidence",
                    text=fact.text,
                    is_gold=True,
                    source_type="counterfactual_evidence",
                )

                distractor_hits = retriever.retrieve(
                    claim,
                    n=n_distractors,
                    exclude_titles={fact.entity_id},
                )
                distractor_segs = [
                    SourceSegment(
                        segment_id=1000 + i,
                        title=f"Distractor {i}",
                        text=hit["text"],
                        is_gold=False,
                        source_type="counterfactual_distractor",
                    )
                    for i, hit in enumerate(distractor_hits)
                    if hit["text"].strip()
                ]

                examples.append(NormalizedExample(
                    dataset="fever",
                    example_id=f"cf-fever-{example_id}",
                    query=claim,
                    gold_answer=label_type,
                    task_type="classification",
                    evidence_segments=[evidence_seg],
                    distractor_segments=distractor_segs,
                    metadata={
                        "counterfactual": True,
                        "required_segment_ids": [0],
                        "missing_evidence_answer": "NOT_ENOUGH_INFO",
                        "fictional_entity_ids": [entity.entity_id],
                        "fact_type": fact.fact_type,
                    },
                ))
                example_id += 1

        return examples

    def _make_supports_claim(self, entity: FictionalEntity, fact: FictionalFact) -> str:
        if fact.fact_type == "field":
            return f"{entity.name} was a {entity.field} researcher."
        if fact.fact_type == "birth":
            return f"{entity.name} was born in {entity.birth_location}."
        if fact.fact_type == "discovery":
            disc = entity.discoveries[0] if entity.discoveries else "a major breakthrough"
            return f"{entity.name} discovered {disc}."
        if fact.fact_type == "award":
            award = entity.awards[0] if entity.awards else "a prestigious award"
            return f"{entity.name} received {award}."
        return f"{entity.name} contributed to {entity.field}."

    def _make_refutes_claim(self, entity: FictionalEntity, fact: FictionalFact) -> str:
        if fact.fact_type == "field":
            fake_field = self._rng.choice(_FIELD_ADJECTIVES) + " " + self._rng.choice(_FIELD_DISCIPLINES)
            return f"{entity.name} was a {fake_field} researcher."
        if fact.fact_type == "birth":
            fake_loc = self._rng.choice(_LOCATION_CITIES) + ", " + self._rng.choice(_LOCATION_COUNTRIES)
            return f"{entity.name} was born in {fake_loc}."
        if fact.fact_type == "discovery":
            return f"{entity.name} never made any significant discoveries."
        return f"{entity.name} had no affiliation with {entity.institution}."

    def _make_nei_claim(self, entity: FictionalEntity) -> str:
        templates = [
            f"{entity.name} collaborated on a classified research initiative.",
            f"{entity.name} published a controversial paper on theoretical applications.",
            f"{entity.name} was nominated for an international distinction in a related field.",
            f"{entity.name} spent a sabbatical at an undisclosed foreign institution.",
        ]
        return self._rng.choice(templates)


# ── HotpotQA Generator ──────────────────────────────────────────────────

class CounterfactualHotpotQAGenerator:
    """Generate multi-hop QA examples from linked fictional entity pairs."""

    def __init__(self, world: FictionalWorldBuilder, *, seed: int = 42) -> None:
        self._world = world
        self._rng = Random(seed)

    def generate(
        self,
        n_examples: int = 50,
        n_distractors: int = 5,
    ) -> List[NormalizedExample]:
        pairs = self._world.get_entity_pairs()
        self._rng.shuffle(pairs)

        corpus = self._world.all_fact_texts()
        retriever = BM25Retriever(corpus)

        examples: List[NormalizedExample] = []
        example_id = 0

        for entity_a, entity_b in pairs:
            if len(examples) >= n_examples:
                break

            q_type = self._rng.choice(["bridge_field", "bridge_birth", "comparison_year"])

            if q_type == "bridge_field":
                question = (
                    f"What field did the researcher who shared an institution "
                    f"with {entity_b.name} specialize in?"
                )
                answer = entity_a.field
            elif q_type == "bridge_birth":
                question = (
                    f"Where was the colleague of {entity_b.name} born?"
                )
                answer = entity_a.birth_location
            else:
                question = (
                    f"Who was born earlier, {entity_a.name} or {entity_b.name}?"
                )
                answer = entity_a.name if entity_a.birth_year < entity_b.birth_year else entity_b.name

            facts_a = self._world.get_entity_facts(entity_a.entity_id)
            facts_b = self._world.get_entity_facts(entity_b.entity_id)
            shared_facts = [f for f in self._world.facts if f.linked_entity_ids]
            relevant_shared = [
                f for f in shared_facts
                if f.entity_id == entity_a.entity_id and entity_b.entity_id in f.linked_entity_ids
            ]

            evidence_texts = []
            if facts_a:
                evidence_texts.append(facts_a[0].text)
            if facts_b:
                evidence_texts.append(facts_b[0].text)
            if relevant_shared:
                evidence_texts.append(relevant_shared[0].text)
            if facts_a and len(facts_a) > 1:
                evidence_texts.append(facts_a[1].text)

            evidence_segs = [
                SourceSegment(
                    segment_id=i,
                    title=f"Evidence {i}",
                    text=text,
                    is_gold=True,
                    source_type="counterfactual_evidence",
                )
                for i, text in enumerate(evidence_texts)
            ]

            combined_query = question + " " + entity_a.name + " " + entity_b.name
            distractor_hits = retriever.retrieve(
                combined_query,
                n=n_distractors,
                exclude_titles={entity_a.entity_id, entity_b.entity_id},
            )
            distractor_segs = [
                SourceSegment(
                    segment_id=1000 + i,
                    title=f"Distractor {i}",
                    text=hit["text"],
                    is_gold=False,
                    source_type="counterfactual_distractor",
                )
                for i, hit in enumerate(distractor_hits)
                if hit["text"].strip()
            ]

            examples.append(NormalizedExample(
                dataset="hotpotqa",
                example_id=f"cf-hotpot-{example_id}",
                query=question,
                gold_answer=answer,
                task_type="qa",
                evidence_segments=evidence_segs,
                distractor_segments=distractor_segs,
                metadata={
                    "counterfactual": True,
                    "required_segment_ids": list(range(len(evidence_segs))),
                    "qa_fallback_answer": "unknown",
                    "fictional_entity_ids": [entity_a.entity_id, entity_b.entity_id],
                    "hop_type": q_type,
                },
            ))
            example_id += 1

        return examples
