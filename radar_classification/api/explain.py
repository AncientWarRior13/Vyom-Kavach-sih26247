"""Live explainability API: real SHAP + LIME for one detection, rendered to
JSON (for the frontend panel) or PDF (for an exportable, shareable report).

This is the deployment counterpart to radarsense/explain.py, which only ever
produces static offline artifacts (a fixed test-set instance, written once to
artifacts/plots/). Here every call explains whatever the caller sends --
matching FEAT31/FEAT42's "live is different from offline" pattern already
established by masking_rescue.py.

Both explainers wrap the SAME artifacts/models/final_model.joblib CatBoost
model already validated by explain.py's physics-sanity check -- not a
separate, unvalidated model.
"""

import io
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, create_model
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether,
)

from radarsense.explain_live import explain, FEATURES, CLASSES

router = APIRouter(prefix="/explain", tags=["explain"])

ExplainSample: type[BaseModel] = create_model(
    "ExplainSample", **{f: (float, ...) for f in FEATURES}
)


@router.post("")
def explain_json(sample: ExplainSample):
    """JSON bundle for an inline frontend explanation panel."""
    try:
        return explain(sample.model_dump())
    except Exception as e:
        raise HTTPException(500, f"explain error: {str(e)}")


@router.post("/pdf")
def explain_pdf(sample: ExplainSample):
    """The same bundle, rendered to a downloadable PDF report."""
    try:
        bundle = explain(sample.model_dump())
        pdf_bytes = render_pdf(bundle)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="radarsense_explanation.pdf"'},
        )
    except Exception as e:
        raise HTTPException(500, f"pdf export error: {str(e)}")



def _bar_chart_png(contributions, value_key, title, xlabel):
    top = contributions[:8]
    labels = [c["feature"] for c in top][::-1]
    values = [c[value_key] for c in top][::-1]
    colors_ = ["#dc2626" if v < 0 else "#f59e0b" for v in values]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(labels, values, color=colors_)
    ax.axvline(0, color="#71717a", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _section(elements, styles, title, contributions, value_key, xlabel):
    section = [Paragraph(title, styles["Heading2"])]
    img_buf = _bar_chart_png(contributions, value_key, title, xlabel)
    section.append(Image(img_buf, width=6.2 * inch, height=3.1 * inch))

    is_shap = value_key == "shap_value"
    header = ["Feature", "Value", "SHAP value"] if is_shap else ["Rule", "LIME weight"]
    table_data = [header]
    for c in contributions[:8]:
        if is_shap:
            table_data.append([c["feature"], f"{c['value']:.3f}", f"{c[value_key]:+.4f}"])
        else:
            table_data.append([c["feature"], f"{c[value_key]:+.4f}"])
    col_widths = [2.6 * inch, 1.8 * inch, 1.8 * inch] if is_shap else [4.4 * inch, 1.8 * inch]
    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18181b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d4d4d8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f5")]),
    ]))
    section.append(Spacer(1, 6))
    section.append(t)
    section.append(Spacer(1, 14))
    elements.append(KeepTogether(section))


def render_pdf(bundle: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.6 * inch, rightMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=8,
                               textColor=colors.HexColor("#52525b")))
    elements = []

    elements.append(Paragraph("RadarSense — Detection Explanation Report", styles["Title"]))
    elements.append(Paragraph(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"real SHAP (exact TreeExplainer) + LIME (local surrogate, 500 samples) on "
        f"artifacts/models/final_model.joblib — not a client-side approximation.",
        styles["Mono"]))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#d4d4d8")))
    elements.append(Spacer(1, 10))

    pred = bundle["prediction"]
    elements.append(Paragraph("Prediction", styles["Heading2"]))
    pred_rows = [["Predicted class", pred["label"].upper()]]
    for cls in CLASSES:
        pred_rows.append([f"p({cls})", f"{pred['probabilities'][cls]:.4f}"])
    pred_rows.append(["Drone gate threshold", f"{pred['drone_threshold']:.2f}"])
    pred_rows.append(["Drone gate fired", str(pred["drone_gate_fired"])])
    pt = Table(pred_rows, colWidths=[2.6 * inch, 3.6 * inch])
    pt.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d4d4d8")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fef3c7")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(pt)
    elements.append(Spacer(1, 8))

    feat_rows = [["Feature", "Value"]] + [[f, f"{bundle['sample'][f]:.3f}"] for f in FEATURES]
    ft = Table(feat_rows, colWidths=[3.0 * inch, 3.2 * inch])
    ft.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e4e4e7")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#18181b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ]))
    elements.append(Paragraph("Input feature vector", styles["Heading3"]))
    elements.append(ft)
    elements.append(Spacer(1, 14))

    for cls, shap_res in bundle["shap"].items():
        _section(elements, styles, f"SHAP contributions — {cls} class",
                  shap_res["contributions"], "shap_value",
                  f"SHAP value (base value {shap_res['base_value']:+.3f})")

    for cls, lime_res in bundle["lime"].items():
        _section(elements, styles, f"LIME local explanation — {cls} class",
                  lime_res["weights"], "weight", "LIME weight")

    elements.append(HRFlowable(width="100%", color=colors.HexColor("#d4d4d8")))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "SHAP and LIME explain the MODEL's local reasoning, not ground physical truth — "
        "both are reported here exactly as computed, with no post-hoc filtering.",
        styles["Mono"]))

    doc.build(elements)
    return buf.getvalue()
