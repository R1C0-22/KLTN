"""
Preprocessing module for Temporal Knowledge Graph Forecasting.

Converts TKG quadruples (subject, relation, object, timestamp) into
natural-language sentences following the verbalization strategy described
in the AnRe framework (Tang et al., ACL 2025).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

# ---------------------------------------------------------------------------
# Relation → natural-language template mapping
# ---------------------------------------------------------------------------
# Each value is the *complete* past-tense verb phrase that sits between
# the subject and the object in the output sentence.  The mapping is
# looked up case-insensitively.
#
# Coverage: all 251 CAMEO-coded relation strings used in ICEWS05-15,
# ICEWS14, ICEWS18, and GDELT benchmarks.

RELATION_TEMPLATES: dict[str, str] = {
    # --- statements & comments ---
    "make statement":                       "made a statement about",
    "make a statement":                     "made a statement about",
    "make optimistic comment":              "made an optimistic comment about",
    "make pessimistic comment":             "made a pessimistic comment about",
    "make empathetic comment":              "made an empathetic comment about",
    "discuss by telephone":                 "discussed by telephone with",

    # --- visits & hosting ---
    "visit":                                "made a visit to",
    "make a visit":                         "made a visit to",
    "host a visit":                         "hosted a visit by",

    # --- diplomacy & cooperation ---
    "meet":                                 "met",
    "consult":                              "consulted with",
    "negotiate":                            "negotiated with",
    "engage in negotiation":                "engaged in negotiation with",
    "engage in diplomatic cooperation":     "engaged in diplomatic cooperation with",
    "engage in material cooperation":       "engaged in material cooperation with",
    "cooperate":                            "cooperated with",
    "cooperate militarily":                 "cooperated militarily with",
    "cooperate economically":               "cooperated economically with",
    "sign agreement":                       "signed an agreement with",
    "sign formal agreement":                "signed a formal agreement with",

    # --- appeals & requests ---
    "appeal":                               "made an appeal to",
    "make an appeal":                       "made an appeal to",
    "make an appeal or request":            "made an appeal or request to",
    "appeal for diplomatic cooperation (such as policy support)":
        "appealed for diplomatic cooperation with",
    "appeal for economic cooperation":      "appealed for economic cooperation with",
    "appeal for military cooperation":      "appealed for military cooperation with",

    # --- intent to cooperate ---
    "express intent to cooperate":          "expressed intent to cooperate with",
    "express intent to cooperate economically":
        "expressed intent to cooperate economically with",
    "express intent to meet or negotiate":  "expressed intent to meet or negotiate with",
    "express intent to engage in diplomatic cooperation (such as policy support)":
        "expressed intent to engage in diplomatic cooperation with",
    "express intent to provide aid":        "expressed intent to provide aid to",
    "express intent to provide material aid":
        "expressed intent to provide material aid to",
    "express intent to provide military aid":
        "expressed intent to provide military aid to",
    "express intent to allow international involvement (non-mediation)":
        "expressed intent to allow international involvement regarding",
    "express intent to change leadership":  "expressed intent to change leadership of",
    "express intent to settle dispute":     "expressed intent to settle a dispute with",
    "express intent to change institutions, regime":
        "expressed intent to change institutions or regime of",
    "express intent to yield":              "expressed intent to yield to",

    # --- endorsement & praise ---
    "praise or endorse":                    "praised or endorsed",
    "rally support on behalf of":           "rallied support on behalf of",
    "grant diplomatic recognition":         "granted diplomatic recognition to",

    # --- aid & assistance ---
    "provide aid":                          "provided aid to",
    "provide humanitarian aid":             "provided humanitarian aid to",
    "provide military aid":                 "provided military aid to",
    "provide economic aid":                 "provided economic aid to",
    "provide material aid":                 "provided material aid to",

    # --- concessions & yielding ---
    "yield":                                "yielded to",
    "accede to demands for rights":         "acceded to demands for rights of",
    "ease administrative sanctions":        "eased administrative sanctions on",
    "return, release person(s)":            "returned or released person(s) held by",
    "return, release":                      "returned or released",

    # --- criticism & disapproval ---
    "criticize":                            "criticized",
    "criticize or denounce":                "criticized or denounced",
    "condemn":                              "condemned",
    "accuse":                               "accused",
    "accuse of war crimes":                 "accused of war crimes",
    "deny":                                 "denied responsibility to",
    "deny responsibility":                  "denied responsibility to",

    # --- rejection & opposition ---
    "reject":                               "rejected",
    "demand":                               "made a demand of",
    "threaten":                             "threatened",
    "threaten with military force":         "threatened with military force against",
    "threaten with political sanctions, not specified below":
        "threatened with political sanctions against",
    "threaten with administrative sanctions":
        "threatened with administrative sanctions against",

    # --- protests & demonstrations ---
    "protest":                              "protested against",
    "demonstrate or rally":                 "demonstrated or rallied against",
    "demonstrate, rally for":               "demonstrated or rallied for",
    "protest violently, riot":              "protested violently or rioted against",
    "obstruct passage, block":              "obstructed passage or blocked",
    "engage in mass killings":              "engaged in mass killings against",
    "engage in mass expulsion":             "engaged in mass expulsion of",

    # --- sanctions & restrictions ---
    "sanction":                             "imposed sanctions on",
    "impose embargo, boycott, or sanctions":
        "imposed an embargo, boycott, or sanctions on",
    "impose administrative sanctions":      "imposed administrative sanctions on",
    "reduce or break diplomatic relations": "reduced or broke diplomatic relations with",
    "reduce relations":                     "reduced relations with",
    "halt negotiations":                    "halted negotiations with",
    "expel or withdraw":                    "expelled or withdrew",
    "expel or withdraw peacekeepers":       "expelled or withdrew peacekeepers from",
    "confiscate property":                  "confiscated property of",

    # --- investigations & legal ---
    "investigate":                          "investigated",
    "investigate crime, corruption":        "investigated crime or corruption involving",
    "arrest, detain, or charge with legal action":
        "arrested, detained, or charged with legal action",
    "arrest":                               "arrested",

    # --- violence & military ---
    "fight":                                "fought with",
    "assault":                              "assaulted",
    "physically assault":                   "physically assaulted",
    "fight with small arms and light weapons":
        "fought with small arms and light weapons against",
    "use conventional military force":      "used conventional military force against",
    "use unconventional military force":    "used unconventional military force against",
    "use unconventional violence":          "used unconventional violence against",
    "use chemical, biological, or radiological weapons":
        "used chemical, biological, or radiological weapons against",
    "conduct suicide, car, or other non-military bombing":
        "conducted a suicide, car, or other non-military bombing against",
    "abduct, hijack, or take hostage":      "abducted, hijacked, or took hostage",
    "coerce":                               "coerced",
    "mobilize":                             "mobilized against",
    "at war":                               "was at war with",

    # --- engagement & symbolic ---
    "engage in symbolic act":               "engaged in a symbolic act with",
    "engage in unconventional mass violence":
        "engaged in unconventional mass violence against",

    # --- miscellaneous ---
    "occupy territory":                     "occupied territory of",
    "seize or damage property":             "seized or damaged property of",
    "use as human shield":                  "used as human shield",
    "assassinate":                          "assassinated",
    "attempt to assassinate":               "attempted to assassinate",
    "sexually assault":                     "sexually assaulted",
    "torture":                              "tortured",
    "kill by physical assault":             "killed by physical assault",
}


# ---------------------------------------------------------------------------
# Fallback past-tense conversion
# ---------------------------------------------------------------------------

_IRREGULAR_PAST: dict[str, str] = {
    "meet": "met", "fight": "fought", "make": "made", "is": "was",
    "have": "had", "go": "went", "give": "gave", "take": "took",
    "get": "got", "say": "said", "see": "saw", "know": "knew",
    "come": "came", "find": "found", "think": "thought", "tell": "told",
    "begin": "began", "run": "ran", "write": "wrote", "bring": "brought",
    "buy": "bought", "send": "sent", "build": "built", "cut": "cut",
    "put": "put", "set": "set", "hit": "hit", "let": "let",
    "hold": "held", "read": "read", "stand": "stood", "win": "won",
    "break": "broke", "drive": "drove", "keep": "kept", "lead": "led",
    "lose": "lost", "pay": "paid", "rise": "rose", "speak": "spoke",
    "steal": "stole", "throw": "threw", "withdraw": "withdrew",
}


def _simple_past(verb: str) -> str:
    """Best-effort conversion of an English verb to simple past tense."""
    lower = verb.lower().strip()
    if lower in _IRREGULAR_PAST:
        return _IRREGULAR_PAST[lower]
    if lower.endswith("e"):
        return lower + "d"
    if re.match(r".*[^aeiou][aeiou][^aeiouwxy]$", lower):
        return lower + lower[-1] + "ed"
    if lower.endswith("y") and lower[-2] not in "aeiou":
        return lower[:-1] + "ied"
    return lower + "ed"


def _format_date(timestamp: str) -> str:
    """Convert a timestamp string into a human-readable date.

    Supports ISO-like formats (2014-01-01, 2014/01/01) and already-
    readable dates (passed through unchanged). For ICEWS-style datasets
    where timestamps are integer day indices (time IDs), this function
    maps them to calendar dates assuming a dataset-specific start date.
    For ICEWS05-15, we treat ``0`` as 1 January 2005 and add days.
    """
    timestamp = timestamp.strip()

    # Handle pure integer time IDs (as used in ICEWS datasets).
    if timestamp.isdigit():
        try:
            time_id = int(timestamp)
        except ValueError:
            time_id = None
        if time_id is not None:
            # ICEWS05-15 spans 2005-01-01 onward. We treat 0 as 2005-01-01.
            base_date = datetime(2005, 1, 1)
            dt = base_date + timedelta(days=time_id)
            return f"{dt.day} {dt.strftime('%B')} {dt.year}"

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(timestamp, fmt)
            return f"{dt.day} {dt.strftime('%B')} {dt.year}"
        except ValueError:
            continue
    return timestamp


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Quadruple:
    subject: str
    relation: str
    object: str
    timestamp: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SPLIT_NAMES = ("train", "test", "valid", "dev")


def load_dataset(
    path: str | Path,
    *,
    splits: Sequence[str] | None = None,
) -> list[Quadruple]:
    """Load TKG quadruples from a file **or dataset directory**.

    Parameters
    ----------
    path : str | Path
        * A **file** — tab / comma-separated (.txt, .csv, .tsv, or
          extensionless like ICEWS ``train``), JSON (.json), or
          JSON-Lines (.jsonl).
        * A **directory** (e.g. ``Code/data/ICEWS05-15``) — all split
          files found inside (``train``, ``test``, ``valid``) are loaded
          and concatenated.
    splits : sequence of str, optional
        When *path* is a directory, load only the listed splits.
        Defaults to every split file found.

    Returns a list of :class:`Quadruple` instances.
    """
    path = Path(path)

    if path.is_dir():
        return _load_directory(path, splits)

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    return _load_tabular(path)


def verbalize_event(
    s: str,
    r: str,
    o: str,
    t: str,
    *,
    use_date_words: bool = True,
) -> str:
    """Turn a single quadruple into a natural-language sentence.

    Parameters
    ----------
    s : str   — subject entity
    r : str   — relation
    o : str   — object entity
    t : str   — timestamp string
    use_date_words : bool
        If *True* (default), convert ``2014-01-01`` → ``1 January 2014``.
        If *False*, keep the raw timestamp.

    Returns
    -------
    str — e.g. ``"On 1 January 2014, USA met China."``
    """
    date_str = _format_date(t) if use_date_words else t.strip()
    verb_phrase = _resolve_relation(r)
    return f"On {date_str}, {_pretty_entity(s)} {verb_phrase} {_pretty_entity(o)}."


def build_corpus(
    data: Sequence[Quadruple] | Sequence[tuple[str, str, str, str]],
    *,
    use_date_words: bool = True,
) -> list[str]:
    """Verbalize every quadruple in *data* and return a list of sentences.

    *data* can be a sequence of :class:`Quadruple` objects **or** plain
    4-tuples ``(subject, relation, object, timestamp)``.
    """
    sentences: list[str] = []
    for item in data:
        if isinstance(item, Quadruple):
            s, r, o, t = item.subject, item.relation, item.object, item.timestamp
        else:
            s, r, o, t = item
        sentences.append(verbalize_event(s, r, o, t, use_date_words=use_date_words))
    return sentences


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_relation(relation: str) -> str:
    """Map a relation string to a past-tense verb phrase."""
    key = relation.strip().lower()
    if key in RELATION_TEMPLATES:
        return RELATION_TEMPLATES[key]

    # Fallback: auto-conjugate first word, keep the rest, add preposition
    words = relation.strip().split()
    if not words:
        return "interacted with"
    words[0] = _simple_past(words[0])
    phrase = " ".join(words)

    _TRAILING_PREPS = (
        " to", " with", " for", " by", " from",
        " on", " about", " against", " of", " in",
    )
    if not any(phrase.endswith(p) for p in _TRAILING_PREPS):
        phrase += " with"
    return phrase


def _pretty_entity(name: str) -> str:
    """Render an entity name nicely for natural-language output.

    The ICEWS-style datasets encode multi-word entities with underscores,
    e.g. ``South_Korea``.  For readability we replace ``_`` with spaces
    in the verbalized sentences while keeping the original tokenization
    for downstream modeling.
    """
    return name.strip().replace("_", " ")


def _extract_quad(raw: dict) -> Quadruple:
    """Pull s/r/o/t from a JSON dict with flexible key names."""
    s = str(raw.get("subject") or raw.get("s") or raw.get("head", ""))
    r = str(raw.get("relation") or raw.get("r") or raw.get("rel", ""))
    o = str(raw.get("object") or raw.get("o") or raw.get("tail", ""))
    t = str(raw.get("timestamp") or raw.get("t") or raw.get("time", ""))
    return Quadruple(s, r, o, t)


# --- loaders ---------------------------------------------------------------

def _load_directory(
    dirpath: Path,
    splits: Sequence[str] | None,
) -> list[Quadruple]:
    """Load all split files from a TKG dataset directory.

    Supports two common layouts:
    - Raw textual quadruples: split files (``train``, ``valid``, ``test``)
      already store subject / relation / object / timestamp strings.
    - ICEWS-style ID format: split files contain integer IDs
      (entity_id, relation_id, entity_id, time_id) with separate mapping
      files ``entity2id.txt`` and ``relation2id.txt`` in the same directory.
    """
    def _resolve_split_files(names: Sequence[str]) -> list[Path]:
        """Return existing split files, accepting bare names and *.txt."""
        files: list[Path] = []
        for name in names:
            base = dirpath / name
            txt = dirpath / f"{name}.txt"
            if base.is_file():
                files.append(base)
            elif txt.is_file():
                files.append(txt)
        return files

    if splits is None:
        split_files = _resolve_split_files(_SPLIT_NAMES)
    else:
        split_files = _resolve_split_files(list(splits))

    if not split_files:
        raise FileNotFoundError(
            f"No split files ({', '.join(_SPLIT_NAMES)}) found in {dirpath}"
        )

    ent_map_path = dirpath / "entity2id.txt"
    rel_map_path = dirpath / "relation2id.txt"

    # If both mapping files exist, treat this as an ICEWS-style ID dataset
    # and decode IDs into human-readable entity / relation strings.
    if ent_map_path.is_file() and rel_map_path.is_file():
        return _load_icews_id_directory(dirpath, split_files, ent_map_path, rel_map_path)

    # Fallback: assume split files already contain textual quadruples.
    quads: list[Quadruple] = []
    for fp in split_files:
        quads.extend(_load_tabular(fp))
    return quads


def _load_icews_id_directory(
    dirpath: Path,
    split_files: Sequence[Path],
    ent_map_path: Path,
    rel_map_path: Path,
) -> list[Quadruple]:
    """Load ICEWS-style dataset where splits store integer IDs.

    Each split line has the form:
        subject_id \\t relation_id \\t object_id \\t time_id

    and the directory also contains:
        entity2id.txt      (``name\\tidx``)
        relation2id.txt    (``name\\tidx``)

    We decode IDs into their textual names so that downstream modules
    (history, verbalization, clustering) operate on semantic strings,
    as assumed in the AnRe design.
    """

    def _load_id_mapping(path: Path) -> dict[int, str]:
        mapping: dict[int, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                name, idx_str = parts[0], parts[1]
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                mapping[idx] = name
        return mapping

    ent_map = _load_id_mapping(ent_map_path)
    rel_map = _load_id_mapping(rel_map_path)

    quads: list[Quadruple] = []
    for fp in split_files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                s_id_str, r_id_str, o_id_str, t_id_str = parts[:4]
                try:
                    s_id = int(s_id_str)
                    r_id = int(r_id_str)
                    o_id = int(o_id_str)
                except ValueError:
                    # If any field is non-integer, fall back to raw strings.
                    s = s_id_str
                    # Normalize relation: replace underscores and lowercase so it
                    # matches RELATION_TEMPLATES keys (e.g. "Make_statement" →
                    # "make statement").
                    r = r_id_str.replace("_", " ").lower().strip()
                    o = o_id_str
                    t = t_id_str
                else:
                    raw_s = ent_map.get(s_id, s_id_str)
                    raw_r = rel_map.get(r_id, r_id_str)
                    raw_o = ent_map.get(o_id, o_id_str)
                    # Normalize relation string to align with RELATION_TEMPLATES.
                    r = raw_r.replace("_", " ").lower().strip()
                    s = raw_s
                    o = raw_o
                    t = t_id_str  # keep numeric time ID; _format_date will map it

                quads.append(Quadruple(s, r, o, t))

    return quads


def _load_json(path: Path) -> list[Quadruple]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [_extract_quad(d) for d in data]
    raise ValueError(f"Expected a JSON array in {path}")


def _load_jsonl(path: Path) -> list[Quadruple]:
    quads: list[Quadruple] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                quads.append(_extract_quad(json.loads(line)))
    return quads


def _load_tabular(path: Path) -> list[Quadruple]:
    """Load tab-separated or comma-separated quadruples.

    Works with files that have no extension (e.g. ``train``, ``test``,
    ``valid``) as well as ``.txt``, ``.csv``, ``.tsv``.
    """
    quads: list[Quadruple] = []
    with open(path, encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)

        if "\t" in sample:
            dialect: type[csv.Dialect] | str = csv.excel_tab
        else:
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel

        reader = csv.reader(f, dialect)
        for row in reader:
            if len(row) < 4:
                continue
            s, r, o, t = (c.strip() for c in row[:4])
            if s.lower() == "subject":
                continue
            quads.append(Quadruple(s, r, o, t))
    return quads


# ---------------------------------------------------------------------------
# CLI entry point — quick demo with the real dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    default_dir = Path(__file__).resolve().parent.parent / "data" / "ICEWS05-15"

    src = sys.argv[1] if len(sys.argv) > 1 else str(default_dir)
    print(f"Loading from: {src}\n")

    data = load_dataset(src)
    print(f"Total quadruples loaded: {len(data):,}\n")

    print("=== Sample verbalized events ===\n")
    for quad in data[:20]:
        print(f"  {verbalize_event(quad.subject, quad.relation, quad.object, quad.timestamp)}")
