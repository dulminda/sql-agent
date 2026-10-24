"""Schema linking: choose which tables to show the model for a question.

Real databases have hundreds of columns. Sending the whole schema costs tokens and dilutes
attention, so we rank tables by lexical overlap between the question and each table's name,
column names and sample values, then expand along foreign keys so join paths stay intact.
Measured offline as gold-table recall (``sqlagent eval linking``).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from sqlagent.db import Database

_STOP = frozenset(
    "a an and are as at be by for from has have how in is it list many me much of on or "
    "return show that the their them there these this to was what when where which who "
    "with each all give name names number total".split()
)


def _split_identifier(name: str) -> str:
    # InvoiceLine -> "Invoice Line", billing_country -> "billing country"
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|_", " ", name)


def _tokens(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOP:
            continue
        out.append(t[:-1] if len(t) > 3 and t.endswith("s") else t)  # crude plural folding
    return out


class SchemaLinker:
    def __init__(self, db: Database, name_weight: float = 3.0, column_weight: float = 1.5) -> None:
        self.db = db
        self._docs: dict[str, Counter] = {}
        for t in db.schema.values():
            doc: Counter = Counter()
            for tok in _tokens(_split_identifier(t.name)):
                doc[tok] += name_weight
            for c in t.columns:
                for tok in _tokens(_split_identifier(c.name)):
                    doc[tok] += column_weight
                for s in c.samples:
                    for tok in _tokens(s):
                        doc[tok] += 1.0
            self._docs[t.name] = doc
        n = len(self._docs)
        df: Counter = Counter(tok for d in self._docs.values() for tok in d)
        self._idf = {tok: math.log(1 + n / c) for tok, c in df.items()}

    def score(self, question: str) -> dict[str, float]:
        q = _tokens(question)
        return {
            name: sum(min(doc[t], 3.0) * self._idf.get(t, 0.0) for t in q if t in doc)
            for name, doc in self._docs.items()
        }

    def link(self, question: str, k: int = 4, expand_fk: bool = True) -> list[str]:
        scores = self.score(question)
        ranked = sorted(scores, key=lambda t: (-scores[t], t))
        chosen = [t for t in ranked[:k] if scores[t] > 0] or ranked[:1]
        if expand_fk:
            # Add tables that connect two chosen tables (bridge tables such as PlaylistTrack).
            chosen_set = set(chosen)
            for t in self.db.schema.values():
                if t.name in chosen_set:
                    continue
                refs = {fk[1] for fk in t.foreign_keys}
                if len(refs & chosen_set) >= 2:
                    chosen.append(t.name)
        return chosen
