"""
detectors.py – Linguistic & rule/NLP-based ambiguity detectors using spaCy,
WordNet (Lesk), optional fastcoref, and regex/lexicon patterns.

Each detector returns a dict:
    {"score": float (0.0..1.0), "span": str, "reason": str}

The top-level `detect_ambiguities(text, context="")` returns scores for all 10
ambiguity types plus `"none"`. Unambiguous text scores low (< 0.35) on all 10
types and high on `"none"`.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import spacy
from spacy.tokens import Doc, Token

try:
    from nltk.corpus import wordnet as wn
    from nltk.wsd import lesk
    _HAS_NLTK = True
except ImportError:  # pragma: no cover
    wn = None  # type: ignore[assignment]
    lesk = None  # type: ignore[assignment]
    _HAS_NLTK = False

try:
    from fastcoref import FCoref  # type: ignore[import-not-found]
    _HAS_FASTCOREF = True
except ImportError:
    FCoref = None
    _HAS_FASTCOREF = False

_NLP: Optional[spacy.language.Language] = None
_COREF_MODEL: Any = None


def get_nlp() -> spacy.language.Language:
    """Lazily load spaCy en_core_web_sm (falls back to blank 'en' if needed)."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            _NLP = spacy.blank("en")
    return _NLP


def get_coref() -> Any:
    """Lazily load fastcoref FCoref if installed; otherwise return None."""
    global _COREF_MODEL
    if _COREF_MODEL is None and _HAS_FASTCOREF and FCoref is not None:
        try:
            _COREF_MODEL = FCoref(enable_progress_bar=False)
        except Exception:
            _COREF_MODEL = False
    return None if _COREF_MODEL is False else _COREF_MODEL


def _result(score: float, span: str = "", reason: str = "") -> dict[str, Any]:
    return {
        "score": round(max(0.0, min(1.0, float(score))), 4),
        "span": span,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# 1. Lexical ambiguity (WordNet 3+ senses + Lesk + context)
# ---------------------------------------------------------------------------

_SPACY_TO_WN_POS = {
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
}

_LEXICAL_TRAP_WORDS = {
    "bank", "bat", "table", "charge", "close", "block", "cancel",
    "bear", "bark", "light", "spring", "match", "plant", "crane", "pitch",
    "seal", "scale", "tie", "fair", "fine", "left", "right", "ring", "rock",
    "jam", "sink", "watch", "file", "park", "book", "motion",
}


def detect_lexical(doc: Doc, context_doc: Optional[Doc] = None) -> dict[str, Any]:
    """Detect lexical ambiguity: content words with 3+ WordNet senses checked with Lesk + context."""
    context_tokens = [t.text for t in doc] + ([t.text for t in context_doc] if context_doc else [])
    ctx_lemmas = {t.lemma_.lower() for t in doc if not t.is_stop}
    if context_doc is not None:
        ctx_lemmas |= {t.lemma_.lower() for t in context_doc if not t.is_stop}

    best_score = 0.0
    best_span = ""
    best_reason = ""

    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.pos_ not in _SPACY_TO_WN_POS:
            continue
        if tok.tag_ in {"MD"} or tok.dep_ == "aux":
            continue

        word_lower = tok.text.lower()
        lemma_lower = tok.lemma_.lower()

        if _HAS_NLTK and wn is not None:
            wn_pos = _SPACY_TO_WN_POS[tok.pos_]
            try:
                synsets = wn.synsets(lemma_lower, pos=wn_pos)
            except LookupError:
                synsets = []
        else:
            synsets = []

        sense_count = len(synsets)
        if sense_count < 3 and lemma_lower not in _LEXICAL_TRAP_WORDS:
            continue

        # Check Lesk overlap across synsets given sentence + context
        overlap_counts: list[int] = []
        if synsets and lesk is not None:
            chosen = lesk(context_tokens, lemma_lower, pos=_SPACY_TO_WN_POS[tok.pos_])
            for s in synsets[:6]:
                def_words = set(re.findall(r"[a-z]+", s.definition().lower()))
                overlap_counts.append(len(def_words & (ctx_lemmas - {lemma_lower})))
            max_overlap = max(overlap_counts) if overlap_counts else 0
            tied_top = sum(1 for c in overlap_counts if c == max_overlap)
        else:
            chosen = None
            max_overlap = 0
            tied_top = 0

        # Skip nouns directly paired with an explicit identifier/code (e.g. "order ORD_5490")
        if any(re.search(r"[A-Z]+_\d+|\d{3,}", c.text) for c in tok.children):
            continue

        # Score: high if word is a known homonym/polyseme or Lesk has no clear winner
        if lemma_lower in _LEXICAL_TRAP_WORDS:
            # If strong unique context disambiguates (max_overlap >= 2 and tied_top == 1), lower score
            if max_overlap >= 2 and tied_top == 1:
                score = 0.45
            else:
                score = min(0.92, 0.65 + 0.03 * min(sense_count, 9))
            reason = f"'{tok.text}' has {max(sense_count, 3)}+ senses with unresolved context ({chosen.name() if chosen else 'polysemous'})"
        elif sense_count >= 8 and max_overlap == 0 and len(doc) <= 4:
            # Very short utterance with highly polysemous content word
            score = min(0.75, 0.45 + 0.03 * min(sense_count - 7, 10))
            reason = f"'{tok.text}' has {sense_count} WordNet senses and minimal disambiguating context"
        elif sense_count >= 3 and max_overlap == 0 and tied_top >= 2:
            score = 0.25
            reason = f"'{tok.text}' has {sense_count} WordNet senses"
        else:
            score = 0.15
            reason = f"'{tok.text}' has {sense_count} senses, disambiguated by context"

        if score > best_score:
            best_score = score
            best_span = tok.text
            best_reason = reason

    return _result(best_score, best_span, best_reason)


# ---------------------------------------------------------------------------
# 2. Syntactic ambiguity (PP-attachment, adj+N and N, ambiguous -ing)
# ---------------------------------------------------------------------------

def detect_syntactic(doc: Doc) -> dict[str, Any]:
    """Detect PP-attachment, 'adj + noun and noun' coordination, and ambiguous -ing forms."""
    best_score = 0.0
    best_span = ""
    best_reason = ""

    # A. Prepositional phrase attachment: Verb + ... + Object Noun + Prep + Noun
    for tok in doc:
        if tok.dep_ == "prep" and tok.i >= 2:
            prev_tokens = doc[: tok.i]
            verb = next((t for t in reversed(prev_tokens) if t.pos_ == "VERB"), None)
            if verb is None:
                continue
            # Require a noun/object *between* the verb and the preposition (V ... NP PP)
            between_noun = next(
                (t for t in reversed(doc[verb.i + 1 : tok.i]) if t.pos_ in {"NOUN", "PROPN"} and t.dep_ in {"dobj", "obj", "attr"}),
                None,
            )
            pobj = next((c for c in tok.children if c.dep_ == "pobj"), None)
            if between_noun is not None and pobj is not None:
                if tok.head.pos_ in {"VERB", "NOUN", "PROPN"}:
                    pp_span = doc[between_noun.i : pobj.i + 1].text
                    score = 0.85 if tok.text.lower() in {"with", "in", "on", "at", "by", "for"} else 0.7
                    if score > best_score:
                        best_score = score
                        best_span = pp_span
                        best_reason = f"PP '{tok.text} {pobj.text}' can attach to verb '{verb.text}' or noun '{between_noun.text}'"

    # B. "adj + noun and noun" coordination: e.g. "old men and women"
    for i in range(len(doc) - 3):
        t0, t1, t2, t3 = doc[i], doc[i + 1], doc[i + 2], doc[i + 3]
        if (
            t0.pos_ == "ADJ"
            and t1.pos_ in {"NOUN", "PROPN"}
            and t2.lower_ in {"and", "or"}
            and t3.pos_ in {"NOUN", "PROPN"}
        ):
            span = doc[i : i + 4].text
            score = 0.82
            if score > best_score:
                best_score = score
                best_span = span
                best_reason = f"Modifier '{t0.text}' may scope over '{t1.text}' only or '{t1.text} {t2.text} {t3.text}'"

    # C. Ambiguous -ing forms: e.g. "Flying planes can be dangerous", "Visiting relatives can be boring"
    for tok in doc:
        if tok.tag_ == "VBG" and tok.i == 0 and tok.i + 1 < len(doc):
            nxt = doc[tok.i + 1]
            if nxt.pos_ in {"NOUN", "PROPN"}:
                # Followed by plural noun + modal/copula -> classic gerund vs participle ambiguity
                rest = doc[tok.i + 2 :]
                if any(t.lemma_ in {"be", "can", "may", "might", "seem"} for t in rest):
                    span = f"{tok.text} {nxt.text}"
                    score = 0.88
                    if score > best_score:
                        best_score = score
                        best_span = span
                        best_reason = f"Ambiguous -ing phrase '{span}' (gerund clause vs participial modifier)"

    return _result(best_score, best_span, best_reason)


# ---------------------------------------------------------------------------
# 3. Referential ambiguity (pronouns with 2+ compatible antecedents)
# ---------------------------------------------------------------------------

_PRONOUN_COMPAT = {
    "he": {"num": "sg", "gender": "m"},
    "him": {"num": "sg", "gender": "m"},
    "his": {"num": "sg", "gender": "m"},
    "she": {"num": "sg", "gender": "f"},
    "her": {"num": "sg", "gender": "f"},
    "hers": {"num": "sg", "gender": "f"},
    "it": {"num": "sg", "gender": "n"},
    "its": {"num": "sg", "gender": "n"},
    "they": {"num": "pl", "gender": "any"},
    "them": {"num": "pl", "gender": "any"},
    "their": {"num": "pl", "gender": "any"},
    "this": {"num": "sg", "gender": "n"},
    "that": {"num": "sg", "gender": "n"},
}


def _is_compatible_antecedent(pron: str, cand: Token) -> bool:
    spec = _PRONOUN_COMPAT[pron]
    if cand.pos_ not in {"NOUN", "PROPN"}:
        return False
    is_plural = cand.tag_ in {"NNS", "NNPS"}
    if spec["num"] == "pl" and not is_plural:
        return False
    if spec["num"] == "sg" and is_plural:
        return False
    # Entity-type check for he/she vs it/this
    if spec["gender"] in {"m", "f"}:
        if cand.ent_type_ and cand.ent_type_ not in {"PERSON"}:
            return False
        if cand.pos_ == "NOUN" and cand.ent_type_ == "":
            # Allow animate/role nouns, skip clearly inanimate ones
            if cand.lower_ in {"book", "table", "car", "order", "card", "mat", "water", "sky"}:
                return False
    if spec["gender"] == "n" and cand.ent_type_ == "PERSON":
        return False
    return True


def detect_referential(doc: Doc, context_doc: Optional[Doc] = None) -> dict[str, Any]:
    """Detect referential ambiguity using fastcoref (if available) + number/gender/entity-type matching."""
    combined_text = (context_doc.text + " " + doc.text).strip() if context_doc is not None else doc.text
    coref_clusters: list[list[str]] = []
    coref = get_coref()
    if coref is not None:
        try:
            preds = coref.predict(texts=[combined_text])
            if preds:
                coref_clusters = preds[0].get_clusters(as_strings=True)
        except Exception:
            coref_clusters = []

    best_score = 0.0
    best_span = ""
    best_reason = ""

    for tok in doc:
        p = tok.lower_
        if p not in _PRONOUN_COMPAT:
            continue
        # Skip dummy 'it' in weather/condition idioms like "it's cold", "it is raining"
        if p == "it" and tok.dep_ == "expl":
            continue
        if p == "it" and tok.i + 1 < len(doc) and doc[tok.i + 1].lemma_ == "be":
            if tok.i + 2 < len(doc) and doc[tok.i + 2].pos_ == "ADJ":
                continue
        # Skip demonstrative determiners like "this book" / "that man"
        if p in {"this", "that"} and tok.dep_ == "det":
            continue
        # Skip relative pronoun 'that' (e.g. "the book that I read")
        if p == "that" and tok.dep_ in {"mark", "nsubj", "dobj"} and tok.head.pos_ == "VERB" and tok.i > 0 and doc[tok.i - 1].pos_ in {"NOUN", "PROPN"}:
            continue

        # Collect compatible antecedents preceding the pronoun (and in context)
        candidates: list[str] = []
        if context_doc is not None:
            for c in context_doc:
                if _is_compatible_antecedent(p, c):
                    candidates.append(c.text)
        for c in doc[: tok.i]:
            if _is_compatible_antecedent(p, c):
                candidates.append(c.text)

        unique_cands = list(dict.fromkeys(candidates))
        if len(unique_cands) >= 2:
            score = 0.85 if coref_clusters else 0.78
            reason = f"Pronoun '{tok.text}' has {len(unique_cands)} compatible antecedents: {', '.join(unique_cands[:3])}"
        elif len(unique_cands) == 0 and p in {"it", "this", "that", "they", "he", "she"}:
            # Unbound pronoun with no antecedent in sentence or context
            score = 0.65
            reason = f"Pronoun '{tok.text}' has no explicit antecedent in text or context"
        else:
            score = 0.1
            reason = ""

        if score > best_score:
            best_score = score
            best_span = tok.text
            best_reason = reason

    return _result(best_score, best_span, best_reason)


# ---------------------------------------------------------------------------
# 4. Scope ambiguity (quantifiers + quantifier/negation, "not ... because")
# ---------------------------------------------------------------------------

_QUANTIFIERS = {"every", "all", "each", "some", "a", "an", "any", "many", "most", "several", "two", "three"}
_STRONG_QUANTIFIERS = {"every", "all", "each", "most", "many", "some"}


def detect_scope(doc: Doc) -> dict[str, Any]:
    """Detect quantifier-quantifier, quantifier-negation, and 'not ... because' scope ambiguity."""
    quants = [t for t in doc if t.lower_ in _QUANTIFIERS and t.dep_ in {"det", "quantmod", "nummod", "nsubj", "dobj", "pobj"}]
    # Also catch compound quantifiers like "everyone", "everybody", "everything", "someone", "nobody"
    indef_quants = [t for t in doc if t.lower_ in {"everyone", "everybody", "everything", "someone", "somebody", "something", "anyone", "nobody", "nothing"}]
    all_quants = quants + indef_quants

    negs = [t for t in doc if t.dep_ == "neg" or t.lower_ in {"not", "n't", "never", "no"}]

    # Check "not ... because"
    for neg in negs:
        for tok in doc[neg.i + 1 :]:
            if tok.lower_ == "because":
                span = doc[neg.i : tok.i + 1].text
                return _result(0.88, span, "Negation + 'because' scope ambiguity (negated cause vs negated action)")

    # Check quantifier + negation (at least one strong quantifier)
    if negs and any(q.lower_ in _STRONG_QUANTIFIERS or q.lower_ in {"everyone", "everybody", "everything"} for q in all_quants):
        q0 = all_quants[0]
        span = f"{q0.text} ... {negs[0].text}"
        return _result(0.85, span, f"Quantifier '{q0.text}' interacting with negation '{negs[0].text}'")

    # Check two quantifiers where at least one is strong (e.g. "Every student read a book")
    if len(all_quants) >= 2 and any(q.lower_ in _STRONG_QUANTIFIERS or q.lower_ in {"everyone", "everybody"} for q in all_quants):
        # Make sure they belong to different noun phrases
        heads = {q.head.i for q in all_quants}
        if len(heads) >= 2:
            span = f"{all_quants[0].text} ... {all_quants[1].text}"
            return _result(0.84, span, f"Multiple quantifiers ('{all_quants[0].text}', '{all_quants[1].text}') allow wide vs narrow scope")

    return _result(0.0, "", "")


# ---------------------------------------------------------------------------
# 5. Semantic ambiguity (plural subject + action verb -> collective vs distributive)
# ---------------------------------------------------------------------------

_STATIVE_VERBS = {"be", "seem", "appear", "exist", "belong", "have", "know", "boil", "shine", "sit", "sleep"}


def detect_semantic(doc: Doc) -> dict[str, Any]:
    """Detect plural subject + action verb (collective vs distributive ambiguity)."""
    for tok in doc:
        if tok.dep_ in {"nsubj", "nsubjpass"}:
            subj_is_plural = (
                tok.tag_ in {"NNS", "NNPS"}
                or tok.lower_ in {"they", "we", "both"}
                or any(c.dep_ == "conj" for c in tok.children)
                or any(c.dep_ == "nummod" and c.lower_ not in {"1", "one"} for c in tok.children)
            )
            head_verb = tok.head
            if subj_is_plural and head_verb.pos_ == "VERB" and head_verb.lemma_.lower() not in _STATIVE_VERBS:
                # Must have an object or complement to yield collective vs distributive reading
                has_obj = any(c.dep_ in {"dobj", "obj", "prep", "attr"} for c in head_verb.children)
                if has_obj:
                    span = f"{tok.text} ... {head_verb.text}"
                    return _result(
                        0.76,
                        span,
                        f"Plural subject '{tok.text}' + action verb '{head_verb.text}' (collective vs distributive reading)",
                    )
    return _result(0.0, "", "")


# ---------------------------------------------------------------------------
# 6. Pragmatic ambiguity (indirect requests, condition statements, sarcasm)
# ---------------------------------------------------------------------------

_INDIRECT_REQ_RE = re.compile(r"\b(can|could|would|will)\s+you\b", re.IGNORECASE)
_CONDITION_STMT_RE = re.compile(
    r"\b(it'?s|it\s+is)\s+(cold|hot|freezing|warm|dark|loud|noisy|stuffy|late)\b",
    re.IGNORECASE,
)
_SARCASM_RE = re.compile(
    r"\b(yeah\s+right|oh\s+great|just\s+great|thanks\s+a\s+lot|sure\s+thing|big\s+help|nice\s+job)\b",
    re.IGNORECASE,
)


def detect_pragmatic(doc: Doc) -> dict[str, Any]:
    """Detect indirect requests, condition statements, and possible sarcasm."""
    text = doc.text

    m_req = _INDIRECT_REQ_RE.search(text)
    if m_req:
        return _result(0.84, m_req.group(0), "Indirect speech act: literal question vs polite request")

    m_cond = _CONDITION_STMT_RE.search(text)
    if m_cond:
        return _result(0.75, m_cond.group(0), "Condition statement may imply an indirect request (e.g. close window / turn on AC)")

    m_sarc = _SARCASM_RE.search(text)
    if m_sarc:
        return _result(0.72, m_sarc.group(0), "Expression commonly used sarcastically vs literally")

    return _result(0.0, "", "")


# ---------------------------------------------------------------------------
# 7. Underspecified (imperatives missing object/time/target, vague words)
# ---------------------------------------------------------------------------

_VAGUE_WORDS = {"recent", "recently", "soon", "stuff", "things", "something", "someone", "somehow"}
_VAGUE_PHRASES = re.compile(r"\b(the\s+team|the\s+report|the\s+item|the\s+issue|the\s+file|the\s+data)\b", re.IGNORECASE)


def detect_underspecified(doc: Doc) -> dict[str, Any]:
    """Detect imperative verbs missing object/time/target and vague words."""
    # Check imperative verb missing direct object
    if len(doc) > 0:
        first = doc[0]
        if first.pos_ == "VERB" and first.tag_ in {"VB", "VBP"}:
            has_subj = any(t.dep_ in {"nsubj", "nsubjpass"} for t in doc)
            has_obj = any(t.dep_ in {"dobj", "obj", "pobj", "ccomp", "xcomp"} for t in doc)
            if not has_subj and not has_obj:
                return _result(0.86, doc.text.strip(), f"Imperative '{first.text}' is missing target object/time")

    # Check vague phrases
    m_vp = _VAGUE_PHRASES.search(doc.text)
    if m_vp:
        return _result(0.70, m_vp.group(0), f"Underspecified definite reference '{m_vp.group(0)}'")

    # Check vague words
    for tok in doc:
        if tok.lower_ in _VAGUE_WORDS:
            return _result(0.68, tok.text, f"Vague term '{tok.text}' lacks concrete specification")

    return _result(0.0, "", "")


# ---------------------------------------------------------------------------
# 8. Ellipsis ("than + NP", "so can/does", dangling auxiliaries)
# ---------------------------------------------------------------------------

_SO_AUX_RE = re.compile(r"\bso\s+(can|could|does|do|did|will|would|is|are|was|were|has|have)\b", re.IGNORECASE)
_DANGLING_AUX_RE = re.compile(r"\b(can|could|does|do|did|will|would|should|might)\s*[.!?]*$", re.IGNORECASE)


def detect_ellipsis(doc: Doc) -> dict[str, Any]:
    """Detect comparative 'than + NP' ellipsis, 'so can/does', and dangling auxiliaries."""
    # A. "than + NP" without a following verb
    for tok in doc:
        if tok.lower_ == "than" and tok.i + 1 < len(doc):
            tail = doc[tok.i + 1 :]
            has_np = any(t.pos_ in {"NOUN", "PROPN", "PRON"} for t in tail)
            has_verb = any(t.pos_ == "VERB" for t in tail)
            if has_np and not has_verb:
                span = doc[tok.i :].text.strip()
                return _result(0.88, span, f"Comparative ellipsis '{span}' (subject vs object reading)")

    # B. "so can/does/did ..."
    m_so = _SO_AUX_RE.search(doc.text)
    if m_so:
        return _result(0.82, m_so.group(0), f"VP ellipsis '{m_so.group(0)}' depends on antecedent clause")

    # C. Dangling auxiliary at end of clause
    m_aux = _DANGLING_AUX_RE.search(doc.text.strip())
    if m_aux and len(doc) >= 2:
        return _result(0.75, m_aux.group(0), f"Dangling auxiliary '{m_aux.group(0)}' with elided verb phrase")

    return _result(0.0, "", "")


# ---------------------------------------------------------------------------
# 9. Temporal & numeric ambiguity (NER + regex)
# ---------------------------------------------------------------------------

_AMBIG_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|1[0-2])[/\-](\d{2,4})\b")
_REL_DATE_RE = re.compile(r"\b(next|this|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)\b", re.IGNORECASE)
_NO_AMPM_TIME_RE = re.compile(r"\b(?:at|by|around|before|after)\s+([1-9]|1[0-2])(?::([0-5]\d))?\b(?!\s*(?:am|pm|a\.m\.|p\.m\.))", re.IGNORECASE)
_BARE_AMOUNT_RE = re.compile(r"\b(send|pay|transfer|charge|cost|costs|owe|worth|about)\s+(\d+(?:\.\d+)?)\b(?!\s*(?:dollars|usd|inr|rs|rupees|eur|gbp|kg|km|hours|mins|minutes|%))", re.IGNORECASE)


def detect_temporal_numeric(doc: Doc) -> dict[str, Any]:
    """Detect dd/mm vs mm/dd, relative dates, times without AM/PM, and bare numeric amounts."""
    text = doc.text

    m_date = _AMBIG_DATE_RE.search(text)
    if m_date:
        return _result(0.92, m_date.group(0), f"Date '{m_date.group(0)}' is ambiguous between DD/MM and MM/DD")

    m_rel = _REL_DATE_RE.search(text)
    if m_rel:
        return _result(0.84, m_rel.group(0), f"Relative date '{m_rel.group(0)}' can mean upcoming vs following week")

    m_time = _NO_AMPM_TIME_RE.search(text)
    if m_time:
        return _result(0.78, m_time.group(0), f"Time '{m_time.group(0)}' lacks AM/PM")

    m_amt = _BARE_AMOUNT_RE.search(text)
    if m_amt:
        return _result(0.76, m_amt.group(0), f"Numeric amount '{m_amt.group(2)}' lacks currency or unit")

    # Check spaCy NER DATE/TIME/MONEY/CARDINAL only if it looks underspecified (not clear ISO/full month)
    for ent in doc.ents:
        if ent.label_ == "TIME" and not re.search(r"(am|pm|a\.m\.|p\.m\.|:\d{2})", ent.text, re.I):
            return _result(0.70, ent.text, f"TIME entity '{ent.text}' lacks AM/PM")

    return _result(0.0, "", "")


# ---------------------------------------------------------------------------
# 10. Cultural / domain ambiguity
# ---------------------------------------------------------------------------

DOMAIN_LEXICON: dict[str, str] = {
    "table": "parliamentary (postpone/introduce) vs database/furniture",
    "model": "ML/statistical model vs fashion/conceptual model",
    "kernel": "OS/Linux kernel vs ML kernel function vs corn/seed kernel",
    "ms": "Microsoft vs millisecond vs manuscript vs Master of Science",
    "python": "programming language vs snake",
    "java": "programming language vs Indonesian island vs coffee",
    "spring": "Java framework vs season vs mechanical coil vs water source",
    "port": "network port vs harbor vs fortified wine",
    "cell": "biology cell vs spreadsheet cell vs prison cell vs mobile phone",
    "branch": "git/code branch vs bank branch vs tree branch",
    "thread": "CPU/execution thread vs forum thread vs sewing thread",
    "pitch": "audio frequency vs sales pitch vs sports field vs tar",
    "charge": "financial fee vs electrical charge vs criminal charge vs attack",
}


def detect_cultural_domain(doc: Doc) -> dict[str, Any]:
    """Detect terms from the domain lexicon with conflicting cross-domain senses."""
    for tok in doc:
        key = tok.lemma_.lower()
        raw_key = tok.text.lower()
        matched = key if key in DOMAIN_LEXICON else (raw_key if raw_key in DOMAIN_LEXICON else None)
        if matched:
            return _result(
                0.80,
                tok.text,
                f"Domain-sensitive term '{tok.text}': {DOMAIN_LEXICON[matched]}",
            )
    return _result(0.0, "", "")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def detect_all(text: str, context: str = "") -> dict[str, dict[str, Any]]:
    """
    Parse `text` (and optional `context`) with spaCy and run all 10 detectors
    plus `"none"`.
    """
    nlp = get_nlp()
    doc = nlp(text.strip())
    context_doc = nlp(context.strip()) if context and context.strip() else None

    results: dict[str, dict[str, Any]] = {
        "lexical": detect_lexical(doc, context_doc),
        "syntactic": detect_syntactic(doc),
        "referential": detect_referential(doc, context_doc),
        "scope": detect_scope(doc),
        "semantic": detect_semantic(doc),
        "pragmatic": detect_pragmatic(doc),
        "underspecified": detect_underspecified(doc),
        "ellipsis": detect_ellipsis(doc),
        "temporal_numeric": detect_temporal_numeric(doc),
        "cultural_domain": detect_cultural_domain(doc),
    }

    max_score = max((v["score"] for v in results.values()), default=0.0)
    none_score = round(max(0.0, 1.0 - max_score), 4)
    results["none"] = _result(
        none_score,
        "",
        "No significant ambiguity detected" if none_score >= 0.65 else "Ambiguity signals present",
    )
    return results
