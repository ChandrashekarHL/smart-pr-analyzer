"""Remove the teacher's third-person framing from label text.

The teacher wrote 17% of explanations as narration - "The reviewer suggests
that the import be hoisted" - and the v1.0 model, trained on them, produced
that framing on 80% of its outputs: it is the lowest-risk template available
and the model found it. A review posted on a pull request must speak as the
reviewer, not about one, so the framing is stripped before training.

The transform is deliberately conservative. Only openings that match a known
verb are touched, and a verb that changes the meaning when dropped
("questions whether X") is rewritten to keep it ("It is unclear whether X").
Anything else is left exactly as written.
"""

from __future__ import annotations

import re

_SUBJECT = r"(?:the\s+)?(?:reviewer|author|commenter|comment|review|maintainer)"

# Verbs after which the clause stands on its own once the subject is removed.
_DROP = re.compile(
    rf"^\s*{_SUBJECT}\s+(?:also\s+)?"
    r"(?:suggests|recommends|proposes|requests|wants|would like|prefers|"
    r"notes|points out|observes|mentions|states|says|argues|thinks|believes|"
    r"is concerned|expresses concern|highlights|explains|indicates|advises|"
    r"flags|warns|clarifies|confirms|reports|adds|agrees|acknowledges|"
    r"suspects|feels|expects|emphasizes|emphasises|stresses|worries|"
    r"raises (?:a |the )?(?:concern|point|issue)(?: that| about| regarding)?)"
    r"\s+(?:that\s+)?",
    re.I,
)

# "questions the need for X", "doubts the value of X": a noun phrase, not a
# clause, so dropping the verb leaves a fragment. Rephrased as a question
# about that thing.
_QUESTION_NP = re.compile(
    rf"^\s*{_SUBJECT}\s+(?:also\s+)?(?:questions|doubts)\s+"
    r"(?!whether\b|if\b|why\b|how\b|what\b|where\b|when\b)(?P<np>\S.*)",
    re.I | re.S,
)

# Verbs that carry a question; dropping them would assert what was only asked.
_QUESTION = re.compile(
    rf"^\s*{_SUBJECT}\s+(?:also\s+)?"
    r"(?:questions|asks|wonders|is asking|is unsure|is not sure|doubts|"
    r"raises the question of)"
    r"\s+(?P<q>whether|if|why|how|what|where|when)\s+",
    re.I,
)

_QUESTION_LEAD = {
    "whether": "It is unclear whether ",
    "if": "It is unclear whether ",
    "why": "It is unclear why ",
    "how": "It is unclear how ",
    "what": "It is unclear what ",
    "where": "It is unclear where ",
    "when": "It is unclear when ",
}


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def deframe(text: str) -> str:
    """Return the text with a leading third-person frame removed, or the text
    unchanged when it does not open with one."""
    if not text:
        return text
    m = _QUESTION.match(text)
    if m:
        return _QUESTION_LEAD[m.group("q").lower()] + text[m.end():].lstrip()
    m = _DROP.match(text)
    if m:
        rest = text[m.end():].lstrip()
        return _capitalise(rest) if rest else text
    m = _QUESTION_NP.match(text)
    if m:
        np_ = m.group("np").strip()
        return "There is a question about " + np_[:1].lower() + np_[1:]
    return text


def is_framed(text: str) -> bool:
    t = text or ""
    return bool(_QUESTION.match(t) or _DROP.match(t) or _QUESTION_NP.match(t))
