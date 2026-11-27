"""
pdf_export.py – Generate downloadable PDF reports for:
  1. Evaluation results (`/export/evaluation.pdf`)
  2. Resolved text records (`/export/resolved.pdf`)
"""

from __future__ import annotations

from typing import Any

from fpdf import FPDF


def _safe(text: Any) -> str:
    """Convert arbitrary text to latin-1 safe string for core PDF fonts."""
    s = str(text if text is not None else "")
    replacements = {
        "→": "->",
        "↓": "(lower is better)",
        "↑": "(higher is better)",
        "—": "-",
        "–": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "·": "|",
        "⚡": "[CONFLICT]",
        "°": " deg",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _new_pdf(title: str) -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _safe(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    return pdf


def build_evaluation_pdf(eval_data: dict[str, Any]) -> bytes:
    """Build a PDF report for the final held-out evaluation results."""
    ds = eval_data.get("dataset", {})
    cats = ds.get("categories", {})
    m = eval_data.get("systems", {}).get("uncertainty_resolver", {})

    pdf = _new_pdf("Ambiguity Resolver - Final Evaluation Report")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        6,
        _safe(
            f"Dataset: {ds.get('source', 'test.csv')} | Total: {ds.get('total', 0)} "
            f"(Ambiguous: {cats.get('ambiguous', 0)}, "
            f"Context-conflicting: {cats.get('context_conflicting', 0)}, "
            f"Unambiguous: {cats.get('unambiguous', 0)})"
        ),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)

    # Summary Metrics
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Final System Metrics (spaCy + LLM Uncertainty Resolver)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    metrics_rows = [
        ("Detection Precision", m.get("detection_precision", 0)),
        ("Detection Recall", m.get("detection_recall", 0)),
        ("Detection F1", m.get("detection_f1", 0)),
        ("False-Positive Rate (Unambiguous)", m.get("false_positive_rate", 0)),
        ("Resolution Accuracy (Overall)", m.get("resolution_accuracy", 0)),
        ("Resolution Accuracy (Normal)", m.get("resolution_accuracy_normal", 0)),
        ("Resolution Accuracy (Context-Conflicting)", m.get("resolution_accuracy_conflict", 0)),
        ("Expected Calibration Error (ECE)", m.get("ece", 0)),
        ("Failure Count", m.get("failure_count", len(m.get("failure_cases", [])))),
    ]
    for label, val in metrics_rows:
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        pdf.cell(95, 6, _safe(label), border=1)
        pdf.cell(45, 6, _safe(val_str), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Per-Type P/R/F1
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Per-Type Precision / Recall / F1", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 9)
    for hdr, w in [("Type", 55), ("Precision", 30), ("Recall", 30), ("F1", 30)]:
        pdf.cell(w, 6, hdr, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for t_name, pt in (m.get("per_type") or {}).items():
        pdf.cell(55, 6, _safe(t_name), border=1)
        pdf.cell(30, 6, f"{pt.get('precision', 0.0):.3f}", border=1)
        pdf.cell(30, 6, f"{pt.get('recall', 0.0):.3f}", border=1)
        pdf.cell(30, 6, f"{pt.get('f1', 0.0):.3f}", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Accuracy vs Coverage
    curve = m.get("accuracy_vs_coverage") or []
    if curve:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Accuracy vs. Coverage Curve", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 9)
        for hdr, w in [("Threshold", 35), ("Coverage", 35), ("Accuracy", 35)]:
            pdf.cell(w, 6, hdr, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for pt in curve:
            pdf.cell(35, 6, f"{pt['threshold']:.2f}", border=1)
            pdf.cell(35, 6, f"{pt['coverage']:.3f}", border=1)
            pdf.cell(35, 6, f"{pt['accuracy']:.3f}", border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # Failure Cases
    failures = m.get("failure_cases") or []
    if failures:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"Failure Cases ({len(failures)} shown)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        for f in failures:
            line = (
                f"#{f.get('id')} [{f.get('category')}] {f.get('text')} | "
                f"gold={f.get('gold_type')} pred={f.get('pred_type')} | "
                f"reason: {f.get('reason')}"
            )
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 5, _safe(line))
            pdf.ln(1)

    return bytes(pdf.output())


def build_resolved_pdf(results: list[dict[str, Any]]) -> bytes:
    """Build a PDF report for resolved text records."""
    pdf = _new_pdf("Ambiguity Resolver - Resolved Text Report")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _safe(f"Total Records: {len(results)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    for idx, r in enumerate(results, start=1):
        pdf.set_font("Helvetica", "B", 11)
        conf_pct = round(float(r.get("confidence", 0.0)) * 100)
        header = (
            f"Record #{idx} | Type: {r.get('ambiguity_type', 'none')} | "
            f"Status: {r.get('status', '')} | Confidence: {conf_pct}%"
        )
        if r.get("conflict"):
            header += " | [CONFLICT]"
        if r.get("assumed"):
            header += " | [ASSUMED]"
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 6, _safe(header))

        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 5, _safe(f"Original Text: {r.get('text', '')}"))
        resolved = r.get("resolved_text") or r.get("top_reading") or r.get("text", "")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 5, _safe(f"Resolved Text: {resolved}"))
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 5, _safe(f"Top Reading:   {r.get('top_reading', '')}"))

        alts = r.get("alternatives") or []
        if alts:
            alt_str = "; ".join(
                f"{a.get('reading', '')} ({round(float(a.get('probability', 0.0)) * 100)}%)"
                for a in alts
            )
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 5, _safe(f"Alternatives:  {alt_str}"))

        changes = r.get("changes") or []
        if changes:
            ch_str = "; ".join(
                f"[{c.get('span', '')}] '{c.get('before', '')}' -> '{c.get('after', '')}' ({c.get('reason', '')})"
                for c in changes
            )
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 5, _safe(f"Changes:       {ch_str}"))

        if r.get("context_used"):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.epw, 5, _safe(f"Evidence:      {r.get('context_used')}"))
        pdf.ln(3)

    return bytes(pdf.output())
