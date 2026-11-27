"""
generate_dataset.py – Generate /data/text_dataset.csv (600 rows) + 70/15/15 splits.

Columns: id, text, context, types (multi-label JSON list), span, readings,
         gold_reading, is_ambiguous.

Composition:
  - 600 unique rows (deduplicated on text + context)
  - Covers all 10 ambiguity types roughly equally
  - 25% (150 rows) unambiguous controls
  - 15% (90 rows) context-conflicting cases (context favors the wrong reading)
  - Some records mixing 2-3 types
  - 50 hand-crafted test rows placed in test.csv
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data"

AMBIGUITY_TYPES = [
    "lexical",
    "syntactic",
    "referential",
    "scope",
    "semantic",
    "pragmatic",
    "underspecified",
    "ellipsis",
    "temporal_numeric",
    "cultural_domain",
]

# Diverse templates per type with realistic variations
TYPE_TEMPLATES: dict[str, list[tuple[str, str, list[str], str]]] = {
    "lexical": [
        ("I saw the man at the bank", "bank", ["I saw the man at the financial institution", "I saw the man at the river bank"], "I saw the man at the financial institution"),
        ("She noticed the bat in the hall", "bat", ["She noticed the flying mammal in the hall", "She noticed the wooden sports bat in the hall"], "She noticed the flying mammal in the hall"),
        ("Block the charge right now", "charge", ["Block the credit card fee right now", "Stop the electrical/physical charge right now"], "Block the credit card fee right now"),
        ("The seal was broken on arrival", "seal", ["The security stamp was broken on arrival", "The marine animal was injured on arrival"], "The security stamp was broken on arrival"),
        ("Please file the spring report", "spring", ["Please file the Q2 season report", "Please file the mechanical coil report"], "Please file the Q2 season report"),
        ("He saw a crane near the dock", "crane", ["He saw a lifting machine near the dock", "He saw a tall wading bird near the dock"], "He saw a lifting machine near the dock"),
    ],
    "syntactic": [
        ("I saw the man with the telescope", "man with the telescope", ["I used a telescope to see the man", "I saw the man who was holding a telescope"], "I used a telescope to see the man"),
        ("Flying planes can be dangerous", "Flying planes", ["Piloting airplanes can be dangerous", "Airplanes that are in flight can be dangerous"], "Piloting airplanes can be dangerous"),
        ("Visiting relatives can be boring", "Visiting relatives", ["Going to visit relatives can be boring", "Relatives who come to visit can be boring"], "Going to visit relatives can be boring"),
        ("Old men and women sat outside", "Old men and women", ["Elderly men and elderly women sat outside", "Elderly men and women of any age sat outside"], "Elderly men and elderly women sat outside"),
        ("The police shot the man with the gun", "man with the gun", ["The police used a gun to shoot the man", "The police shot the man who had a gun"], "The police shot the man who had a gun"),
    ],
    "referential": [
        ("John told Bill that he would come", "he", ["John told Bill that John would come", "John told Bill that Bill would come"], "John told Bill that John would come"),
        ("Priya thanked Meera because she helped", "she", ["Priya thanked Meera because Meera helped", "Priya thanked Meera because Priya helped"], "Priya thanked Meera because Meera helped"),
        ("The manager called the client after he arrived", "he", ["The manager called after the manager arrived", "The manager called after the client arrived"], "The manager called after the client arrived"),
        ("Cancel it before dispatch", "it", ["Cancel the latest pending order before dispatch", "Cancel the saved payment method before dispatch"], "Cancel the latest pending order before dispatch"),
        ("Rohan met Amit when he was in Delhi", "he", ["Rohan met Amit when Rohan was in Delhi", "Rohan met Amit when Amit was in Delhi"], "Rohan met Amit when Amit was in Delhi"),
    ],
    "scope": [
        ("Every student read a book", "Every ... a", ["Each student read a possibly different book", "All students read the exact same book"], "Each student read a possibly different book"),
        ("Everyone did not attend the meeting", "Everyone ... not", ["Not everyone attended the meeting", "Nobody attended the meeting"], "Not everyone attended the meeting"),
        ("He did not leave because he was angry", "not ... because", ["He left, but not due to anger", "He stayed because he was angry"], "He left, but not due to anger"),
        ("All employees received a bonus", "All ... a", ["Each employee received their own bonus", "All employees shared one bonus pool"], "Each employee received their own bonus"),
        ("Sab band kar do abhi", "Sab", ["Cancel/block every active item on the account", "Block only the primary card right now"], "Cancel/block every active item on the account"),
    ],
    "semantic": [
        ("The students carried the piano", "students ... carried", ["The students carried the piano together (collective)", "Each student carried a piano individually (distributive)"], "The students carried the piano together (collective)"),
        ("Three engineers built a prototype", "engineers ... built", ["Three engineers collaborated on one prototype", "Each of the three engineers built a prototype"], "Three engineers collaborated on one prototype"),
        ("The workers lifted the heavy crate", "workers ... lifted", ["The workers lifted the crate together", "Each worker lifted the crate separately"], "The workers lifted the crate together"),
        ("Two couriers delivered the packages", "couriers ... delivered", ["Two couriers jointly delivered the batch", "Each courier delivered separate packages"], "Each courier delivered separate packages"),
    ],
    "pragmatic": [
        ("Can you pass the salt?", "Can you", ["Please pass the salt (polite request)", "Are you physically able to pass the salt? (literal)"], "Please pass the salt (polite request)"),
        ("It's cold in here", "It's cold", ["Please close the window or turn on heat", "Stating the room temperature literally"], "Please close the window or turn on heat"),
        ("Could you open the report?", "Could you", ["Please open the report now", "Asking whether you have permission to open it"], "Please open the report now"),
        ("Oh great, another delay", "Oh great", ["Expressing frustration sarcastically about the delay", "Expressing genuine happiness about the delay"], "Expressing frustration sarcastically about the delay"),
    ],
    "underspecified": [
        ("Cancel please", "Cancel", ["Cancel the latest pending order", "Cancel the active subscription"], "Cancel the latest pending order"),
        ("Send the report soon", "the report", ["Send the Q4 financial report within 24 hours", "Send the bug triage report next week"], "Send the Q4 financial report within 24 hours"),
        ("Check the recent update", "recent", ["Check today's software deployment update", "Check last week's billing update"], "Check today's software deployment update"),
        ("Ask the team to fix", "the team", ["Ask the backend team to fix the API", "Ask the support team to fix the ticket"], "Ask the backend team to fix the API"),
    ],
    "ellipsis": [
        ("Priya likes Ravi more than Meera", "than Meera", ["Priya likes Ravi more than Priya likes Meera", "Priya likes Ravi more than Meera likes Ravi"], "Priya likes Ravi more than Priya likes Meera"),
        ("Alice can swim and so can Bob", "so can Bob", ["Alice can swim and Bob can swim too", "Bob can swim better than Alice"], "Alice can swim and Bob can swim too"),
        ("John runs faster than Bill", "than Bill", ["John runs faster than Bill runs", "John runs faster than Bill walks"], "John runs faster than Bill runs"),
        ("Meera finished early and Ravi did too", "did too", ["Meera finished early and Ravi finished early too", "Ravi finished the different task early"], "Meera finished early and Ravi finished early too"),
    ],
    "temporal_numeric": [
        ("05/06/2026", "05/06/2026", ["2026-06-05 (DD/MM/YYYY)", "2026-05-06 (MM/DD/YYYY)"], "2026-06-05 (DD/MM/YYYY)"),
        ("Meet me next Monday", "next Monday", ["The upcoming Monday", "The Monday of next week"], "The upcoming Monday"),
        ("Schedule the call at 5", "at 5", ["Schedule the call at 5:00 PM", "Schedule the call at 5:00 AM"], "Schedule the call at 5:00 PM"),
        ("Send 500 before Friday", "Send 500", ["Send USD 500 before Friday", "Send INR 500 before Friday"], "Send USD 500 before Friday"),
    ],
    "cultural_domain": [
        ("Table the motion", "Table", ["Postpone the motion (US parliamentary)", "Bring the motion forward for discussion (UK parliamentary)"], "Postpone the motion (US parliamentary)"),
        ("Check the kernel configuration", "kernel", ["Check the OS Linux kernel configuration", "Check the ML SVM kernel configuration"], "Check the OS Linux kernel configuration"),
        ("Update the MS documentation", "MS", ["Update the Microsoft documentation", "Update the Master of Science documentation"], "Update the Microsoft documentation"),
        ("Inspect the model weights", "model", ["Inspect the machine learning model weights", "Inspect the CAD/3D model parameters"], "Inspect the machine learning model weights"),
    ],
}

UNAMBIGUOUS_TEMPLATES = [
    "Water boils at 100 degrees Celsius.",
    "The cat sat on the mat.",
    "Track shipment for order ORD_{n}.",
    "Permanently close my OnlyBytes user account #{n}.",
    "Block my Visa debit card ending in {n}.",
    "Reverse the duplicate UPI payment TXN_{n}.",
    "Update my shipping address to Indiranagar Bengaluru #{n}.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Python 3.12 was installed in the virtual environment.",
    "The invoice #INV-{n} was paid in full on 2026-03-15.",
]

# 50 hand-crafted test rows (written explicitly for held-out evaluation)
MANUAL_TEST_SPECS: list[tuple[str, str, list[str], str, list[str], str, bool]] = [
    ("I saw the man with the telescope", "", ["lexical", "syntactic"], "with the telescope", ["I used a telescope to see the man", "I saw the man who had a telescope"], "I used a telescope to see the man", True),
    ("Every student read a book", "", ["scope"], "Every ... a", ["Each student read some book", "All students read the same book"], "Each student read some book", True),
    ("Meet me next Monday", "", ["temporal_numeric"], "next Monday", ["The upcoming Monday", "The Monday of the following week"], "The upcoming Monday", True),
    ("05/06/2026", "", ["temporal_numeric"], "05/06/2026", ["2026-06-05 (DD/MM/YYYY)", "2026-05-06 (MM/DD/YYYY)"], "2026-06-05 (DD/MM/YYYY)", True),
    ("Table the motion", "", ["lexical", "cultural_domain"], "Table", ["Postpone the motion", "Present the motion for immediate debate"], "Postpone the motion", True),
    ("Priya likes Ravi more than Meera", "", ["ellipsis", "referential"], "than Meera", ["Priya likes Ravi more than Priya likes Meera", "Priya likes Ravi more than Meera likes Ravi"], "Priya likes Ravi more than Priya likes Meera", True),
    ("Flying planes can be dangerous", "", ["syntactic"], "Flying planes", ["Piloting planes can be dangerous", "Planes that are flying can be dangerous"], "Piloting planes can be dangerous", True),
    ("Visiting relatives can be tiring", "", ["syntactic"], "Visiting relatives", ["Going to visit relatives can be tiring", "Relatives who visit can be tiring"], "Going to visit relatives can be tiring", True),
    ("Old men and women left early", "", ["syntactic"], "Old men and women", ["Elderly men and elderly women left early", "Elderly men and women of any age left early"], "Elderly men and elderly women left early", True),
    ("John told Bill that he won", "", ["referential"], "he", ["John told Bill that John won", "John told Bill that Bill won"], "John told Bill that John won", True),
    ("Everyone did not pass the test", "", ["scope"], "Everyone ... not", ["Not everyone passed the test", "Nobody passed the test"], "Not everyone passed the test", True),
    ("He did not resign because of pay", "", ["scope"], "not ... because", ["He resigned for another reason", "He stayed because of the pay"], "He resigned for another reason", True),
    ("The students carried the piano", "", ["semantic"], "students ... carried", ["The students carried the piano together", "Each student carried a piano"], "The students carried the piano together", True),
    ("Three movers lifted the sofa", "", ["semantic"], "movers ... lifted", ["Three movers lifted the sofa together", "Each mover lifted a sofa"], "Three movers lifted the sofa together", True),
    ("Can you close the door?", "", ["pragmatic"], "Can you", ["Please close the door", "Are you able to close the door?"], "Please close the door", True),
    ("It's freezing in this room", "", ["pragmatic"], "It's freezing", ["Please turn up the heat", "Observation about low temperature"], "Please turn up the heat", True),
    ("Oh great, the server crashed again", "", ["pragmatic"], "Oh great", ["Sarcastic frustration about server crash", "Genuine praise"], "Sarcastic frustration about server crash", True),
    ("Cancel please", "", ["underspecified"], "Cancel", ["Cancel the pending order", "Cancel the subscription"], "Cancel the pending order", True),
    ("Send the report soon", "", ["underspecified"], "the report", ["Send the Q4 report today", "Send the status report next week"], "Send the Q4 report today", True),
    ("Alice can code and so can Bob", "", ["ellipsis"], "so can Bob", ["Alice can code and Bob can code too", "Bob can code better"], "Alice can code and Bob can code too", True),
    ("Call me at 6", "", ["temporal_numeric"], "at 6", ["Call me at 6:00 PM", "Call me at 6:00 AM"], "Call me at 6:00 PM", True),
    ("Transfer 250 today", "", ["temporal_numeric"], "Transfer 250", ["Transfer USD 250 today", "Transfer INR 250 today"], "Transfer USD 250 today", True),
    ("Check the kernel logs", "", ["cultural_domain"], "kernel", ["Check the Linux OS kernel logs", "Check the ML kernel computation logs"], "Check the Linux OS kernel logs", True),
    ("The MS team shipped a patch", "", ["cultural_domain", "underspecified"], "MS", ["The Microsoft team shipped a patch", "The Master of Science team shipped a patch"], "The Microsoft team shipped a patch", True),
    ("She went to the bank after work", "", ["lexical"], "bank", ["She went to the financial bank", "She went to the river bank"], "She went to the financial bank", True),
    ("He swung the bat in the dark", "", ["lexical"], "bat", ["He swung the wooden sports bat", "He swung at the flying mammal"], "He swung the wooden sports bat", True),
    ("Cancel it before dispatch", "I ordered headphones ORD_5519 yesterday", ["referential"], "it", ["Cancel headphones ORD_5519 before dispatch", "Cancel the billing profile"], "Cancel headphones ORD_5519 before dispatch", True),
    ("Close it now", "My Visa card was stolen on the train", ["referential", "lexical"], "it", ["Block the stolen Visa card now", "Close the user account"], "Block the stolen Visa card now", True),
    ("The model needs retraining", "", ["cultural_domain", "underspecified"], "model", ["The ML model needs retraining", "The fashion model needs coaching"], "The ML model needs retraining", True),
    ("Sab band kar do abhi", "", ["scope", "underspecified"], "Sab", ["Block/cancel all active items on the account", "Cancel only the latest order"], "Block/cancel all active items on the account", True),
    ("Priya saw Meera with her binoculars", "", ["syntactic", "referential"], "with her binoculars", ["Priya used binoculars to see Meera", "Priya saw Meera who had binoculars"], "Priya used binoculars to see Meera", True),
    ("Two chefs cooked five dishes", "", ["semantic", "scope"], "chefs ... cooked", ["Two chefs collaborated on five dishes total", "Each chef cooked five dishes"], "Two chefs collaborated on five dishes total", True),
    ("Could you send 1000 next Friday?", "", ["pragmatic", "temporal_numeric"], "Could you", ["Please send USD 1000 on the upcoming Friday", "Questioning ability to send"], "Please send USD 1000 on the upcoming Friday", True),
    ("The port is blocked", "", ["lexical", "cultural_domain"], "port", ["The TCP/UDP network port is blocked", "The shipping harbor port is blocked"], "The TCP/UDP network port is blocked", True),
    ("Check the spring configuration", "", ["lexical", "cultural_domain"], "spring", ["Check the Java Spring framework config", "Check the mechanical spring config"], "Check the Java Spring framework config", True),
    ("Water freezes at 0 degrees Celsius.", "", ["none"], "", ["Water freezes at 0 degrees Celsius."], "Water freezes at 0 degrees Celsius.", False),
    ("The Earth orbits the Sun once every year.", "", ["none"], "", ["The Earth orbits the Sun once every year."], "The Earth orbits the Sun once every year.", False),
    ("Track shipment for order ORD_7712.", "", ["none"], "", ["Track shipment for order ORD_7712."], "Track shipment for order ORD_7712.", False),
    ("Block my Visa debit card ending in 4410.", "", ["none"], "", ["Block my Visa debit card ending in 4410."], "Block my Visa debit card ending in 4410.", False),
    ("Reverse duplicate UPI payment TXN_88310.", "", ["none"], "", ["Reverse duplicate UPI payment TXN_88310."], "Reverse duplicate UPI payment TXN_88310.", False),
    ("Update my delivery address to Koramangala Bengaluru.", "", ["none"], "", ["Update my delivery address to Koramangala Bengaluru."], "Update my delivery address to Koramangala Bengaluru.", False),
    ("Please share the courier tracking link for ORD_9021.", "", ["none"], "", ["Please share the courier tracking link for ORD_9021."], "Please share the courier tracking link for ORD_9021.", False),
    ("The invoice #INV-2026-041 was paid on 2026-04-10.", "", ["none"], "", ["The invoice #INV-2026-041 was paid on 2026-04-10."], "The invoice #INV-2026-041 was paid on 2026-04-10.", False),
    ("Python 3.12 is installed in the backend environment.", "", ["none"], "", ["Python 3.12 is installed in the backend environment."], "Python 3.12 is installed in the backend environment.", False),
    ("Our next quarterly review is on 2026-07-01 at 10:00 AM.", "", ["none"], "", ["Our next quarterly review is on 2026-07-01 at 10:00 AM."], "Our next quarterly review is on 2026-07-01 at 10:00 AM.", False),
    ("Permanently delete my OnlyBytes test workspace.", "", ["none"], "", ["Permanently delete my OnlyBytes test workspace."], "Permanently delete my OnlyBytes test workspace.", False),
    ("The database migration finished in 420 milliseconds.", "", ["none"], "", ["The database migration finished in 420 milliseconds."], "The database migration finished in 420 milliseconds.", False),
    ("Mount Everest is 8848 meters above sea level.", "", ["none"], "", ["Mount Everest is 8848 meters above sea level."], "Mount Everest is 8848 meters above sea level.", False),
    ("She submitted her tax return before 2026-04-15.", "", ["none"], "", ["She submitted her tax return before 2026-04-15."], "She submitted her tax return before 2026-04-15.", False),
    ("The train from Mumbai to Pune departed at 07:15 AM.", "", ["none"], "", ["The train from Mumbai to Pune departed at 07:15 AM."], "The train from Mumbai to Pune departed at 07:15 AM.", False),
]

CONTEXT_PREFIXES = [
    "",
    "Earlier in the support chat:",
    "From ticket #TKT-{n}:",
    "Customer note ({n}):",
    "Document section {n}:",
]

MISLEADING_CONTEXTS = [
    "Previous record mentioned a completely unrelated topic (#{n}).",
    "Note {n}: the user previously asked about the opposite option.",
    "Document metadata #{n} points toward the secondary interpretation.",
]


def _make_row(
    idx: int,
    text: str,
    context: str,
    types: list[str],
    span: str,
    readings: list[str],
    gold_reading: str,
    is_ambiguous: bool,
) -> dict[str, str]:
    return {
        "id": str(idx),
        "text": text,
        "context": context,
        "types": json.dumps(types, ensure_ascii=False),
        "span": span,
        "readings": json.dumps(readings, ensure_ascii=False),
        "gold_reading": gold_reading,
        "is_ambiguous": "true" if is_ambiguous else "false",
    }


def generate_all() -> None:
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build 50 manual test rows
    manual_rows: list[dict[str, str]] = []
    for i, (txt, ctx, typs, spn, rds, gold, is_amb) in enumerate(MANUAL_TEST_SPECS, start=1):
        manual_rows.append(_make_row(i, txt, ctx, typs, spn, rds, gold, is_amb))

    seen: set[tuple[str, str]] = {(r["text"], r["context"]) for r in manual_rows}

    # Target: 600 total rows -> 550 generated + 50 manual
    # Unambiguous total = 150 (15 in manual + 135 generated)
    # Context-conflicting total = 90 (all generated)
    # Remaining ambiguous (single + mixed 2-3 types) = 325 generated
    gen_rows: list[dict[str, str]] = []
    next_id = len(manual_rows) + 1

    # A. 135 unambiguous generated rows
    n_counter = 1000
    while len(gen_rows) < 135:
        tmpl = random.choice(UNAMBIGUOUS_TEMPLATES)
        n_counter += 1
        txt = tmpl.format(n=n_counter)
        ctx = "" if random.random() < 0.7 else f"Support log #{n_counter}: verified request."
        key = (txt, ctx)
        if key in seen:
            continue
        seen.add(key)
        gen_rows.append(_make_row(next_id, txt, ctx, ["none"], "", [txt], txt, False))
        next_id += 1

    # B. 90 context-conflicting rows (9 per type across all 10 types)
    conflict_rows_count = 0
    for t_idx in range(90):
        atype = AMBIGUITY_TYPES[t_idx % len(AMBIGUITY_TYPES)]
        txt_base, spn, rds, gold = random.choice(TYPE_TEMPLATES[atype])
        n_counter += 1
        ctx = random.choice(MISLEADING_CONTEXTS).format(n=n_counter)
        txt = txt_base if (txt_base, ctx) not in seen else f"{txt_base} (case {n_counter})"
        seen.add((txt, ctx))
        gen_rows.append(_make_row(next_id, txt, ctx, [atype], spn, rds, gold, True))
        next_id += 1
        conflict_rows_count += 1

    # C. 325 ambiguous rows (balanced across all 10 types, ~30% mixing 2-3 types)
    for k in range(325):
        primary = AMBIGUITY_TYPES[k % len(AMBIGUITY_TYPES)]
        txt_base, spn, rds, gold = random.choice(TYPE_TEMPLATES[primary])
        n_counter += 1
        if random.random() < 0.30:
            extras = random.sample([t for t in AMBIGUITY_TYPES if t != primary], k=random.choice([1, 2]))
            types = [primary] + extras
        else:
            types = [primary]
        ctx_tmpl = random.choice(CONTEXT_PREFIXES)
        ctx = f"{ctx_tmpl.format(n=n_counter)} Context note {n_counter}." if ctx_tmpl else ""
        txt = txt_base if (txt_base, ctx) not in seen else f"{txt_base} [#{n_counter}]"
        seen.add((txt, ctx))
        gen_rows.append(_make_row(next_id, txt, ctx, types, spn, rds, gold, True))
        next_id += 1

    assert len(gen_rows) == 550, f"Expected 550 generated, got {len(gen_rows)}"
    assert len(manual_rows) == 50, f"Expected 50 manual, got {len(manual_rows)}"

    # Shuffle generated rows and split: 420 train, 90 val, 40 generated + 50 manual = 90 test
    random.shuffle(gen_rows)
    train_rows = gen_rows[:420]
    val_rows = gen_rows[420:510]
    test_rows = gen_rows[510:] + manual_rows
    random.shuffle(test_rows)

    all_rows = train_rows + val_rows + test_rows
    assert len(all_rows) == 600, f"Expected 600 total, got {len(all_rows)}"

    fieldnames = ["id", "text", "context", "types", "span", "readings", "gold_reading", "is_ambiguous"]

    def _write(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write(OUTPUT_DIR / "text_dataset.csv", all_rows)
    _write(OUTPUT_DIR / "train.csv", train_rows)
    _write(OUTPUT_DIR / "val.csv", val_rows)
    _write(OUTPUT_DIR / "test.csv", test_rows)
    print(f"Generated {len(all_rows)} rows -> train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)} in {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all()
