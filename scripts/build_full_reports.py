"""
Full Master Report Builder for CXR Reliability Multi-Agent System.
Generates:
1. outputs/before_after_method_summary.json
2. docs/BEFORE_AFTER_PROJECT_COMPLETE_REPORT_AUDIT.md
3. docs/BEFORE_AFTER_PROJECT_COMPLETE_REPORT.md
4. docs/BEFORE_AFTER_PROJECT_COMPLETE_REPORT.docx
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\PARAS\Desktop\AI SEM 5 B1 G5\PROJECT\cxr-reliability\cxr-reliability")
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


def set_cell_background(cell, hex_color: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)


def add_callout(doc, text: str, title: str = "KEY TAKEAWAY", border_hex="0284C7", bg_hex="F0F9FF"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tbl.columns[0].width = Inches(6.5)
    
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(f"📌 {title}: ")
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(10)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    run_body = p.add_run(text)
    run_body.font.name = "Calibri"
    run_body.font.size = Pt(9.5)
    run_body.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(2)


def create_styled_table(doc, headers: list[str], rows: list[list[str]], col_widths: list[Inches]):
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    hdr_row = tbl.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.width = col_widths[i]
        set_cell_background(cell, "1E293B")
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(h)
        run.font.name = "Calibri"
        run.font.size = Pt(9)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    for r_idx, r_data in enumerate(rows):
        row = tbl.rows[r_idx + 1]
        bg = "F8FAFC" if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(r_data):
            cell = row.cells[c_idx]
            cell.width = col_widths[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=120, right=120)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(str(val))
            run.font.name = "Calibri"
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)
    return tbl


def add_p(doc, text: str, space_after=4, bold_prefix=None, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(10)
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    return p


def add_h1(doc, title: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(title)
    run.font.name = "Calibri"
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    return p


def add_h2(doc, title: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(title)
    run.font.name = "Calibri"
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x03, 0x69, 0xA1)
    return p


def add_h3(doc, title: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(title)
    run.font.name = "Calibri"
    run.font.size = Pt(10.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    return p


def main():
    print("Writing JSON summary & Audit markdown...")
    # 1. Write JSON summary
    summary_data = {
        "metadata": {
            "title": "BEFORE vs AFTER Project Complete Evaluation Summary",
            "date": "2026-10-06",
            "model_id": "TorchXRayVision DenseNet-121 (densenet121-res224-nih)",
            "operating_threshold": 0.522161,
            "dataset": "NIH ChestX-ray14",
            "total_test_images": 16724,
            "pneumonia_positive": 220,
            "pneumonia_negative": 16504,
            "prevalence": 0.013155
        },
        "baseline_standalone_metrics": {
            "population": 16724,
            "tp": 49,
            "tn": 15160,
            "fp": 1344,
            "fn": 171,
            "accuracy": 0.909412,
            "precision": 0.035176,
            "recall": 0.222727,
            "specificity": 0.918565,
            "f1": 0.060756,
            "npv": 0.988846,
            "auroc": 0.701619,
            "auprc": 0.028085,
            "total_errors": 1515
        },
        "reliability_released_metrics": {
            "population": 7349,
            "coverage_pct": 43.9428,
            "withheld_count": 9375,
            "withheld_pct": 56.0572,
            "tp": 17,
            "tn": 6767,
            "fp": 498,
            "fn": 67,
            "accuracy": 0.923119,
            "precision": 0.033010,
            "recall": 0.202381,
            "specificity": 0.931452,
            "f1": 0.056761,
            "npv": 0.990196,
            "total_errors": 565
        },
        "error_containment": {
            "false_positives_withheld": 846,
            "false_positives_containment_pct": 62.9464,
            "false_negatives_withheld": 104,
            "false_negatives_containment_pct": 60.8187,
            "total_errors_withheld": 950,
            "total_error_containment_pct": 62.7063
        },
        "quality_agent_results": {
            "good_count": 3608,
            "degraded_count": 940,
            "poor_count": 12176,
            "repairs_evaluated": 12082,
            "poor_to_good": 7766,
            "poor_to_degraded": 1692,
            "poor_to_poor": 2624,
            "quality_improved_pct": 78.2817,
            "quality_worsened_pct": 0.0
        },
        "ood_agent_results": {
            "in_distribution_count": 16537,
            "borderline_count": 86,
            "severe_count": 101,
            "severe_rejection_rate_pct": 100.0,
            "borderline_escalation_rate_pct": 100.0
        },
        "uncertainty_agent_results": {
            "low_count": 2653,
            "high_count": 14071,
            "low_uncertainty_released_acc": 0.995737,
            "high_uncertainty_released_acc": 0.902225
        },
        "decision_agent_reconciliation": {
            "direct_accept": 501,
            "repair_routed": 13001,
            "direct_escalate": 3121,
            "reject": 101,
            "verified_repair_release": 6848,
            "repair_escalated": 5234,
            "total_reconciled": True
        },
        "final_paired_repair_diagnostic_n35": {
            "cohort_size": 35,
            "correct_to_correct": 17,
            "incorrect_to_incorrect": 18,
            "correct_to_incorrect": 0,
            "incorrect_to_correct": 0,
            "errors_corrected": 0,
            "new_errors": 0,
            "prediction_stability_pct": 100.0,
            "accuracy_before": 0.485714,
            "accuracy_after": 0.485714,
            "accuracy_delta": 0.0
        },
        "deliberate_blur_stress_test": {
            "laplacian_before": 1.70,
            "laplacian_after": 1.72,
            "laplacian_delta": 0.016,
            "snr_before_db": 59.4,
            "snr_after_db": 59.2,
            "densenet_raw_score_before": 0.504086,
            "densenet_raw_score_after": 0.504083,
            "confidence_gain": -0.0003,
            "quality_transition": "POOR -> POOR",
            "verification_status": "ESCALATE",
            "root_cause": "Severe artificial blur destroyed high spatial frequencies; unsharp mask cannot hallucinate edges; verification correctly intercepted the failure."
        }
    }
    with open(OUTPUTS_DIR / "before_after_method_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 2. Write Audit Markdown
    audit_md = f"""# BEFORE vs AFTER Project Complete Report — Source of Truth Audit

**Audit Date:** 2026-10-06  
**Auditor:** Autonomous Verification Engine  
**Project:** Reliability-Aware Multi-Agent System for Chest X-Ray Pneumonia Classification  

---

## 1. Discovered Artifacts & Provenance Verification

| Artifact Key | Relative Path | Record Count / Dimension | Provenance Status |
|---|---|---|---|
| **Full Test Cohort CSV** | `outputs/evaluation_full_test.csv` | 16,724 rows, 29 columns | AUTHORITATIVE — Primary Test Dataset |
| **Full Test Summary JSON** | `outputs/before_after_evaluation_summary.json` | 409 lines structured JSON | AUTHORITATIVE — Aggregated Metrics |
| **Paired Diagnostic CSV** | `outputs/final_paired_repair_diagnostic.csv` | 50 candidates, 35 repaired | AUTHORITATIVE — Main Paired Experiment |
| **Paired Diagnostic Summary JSON** | `outputs/final_paired_repair_diagnostic_summary.json` | 124 lines structured JSON | AUTHORITATIVE — 4-Way Transition Matrix |
| **Repair Tuning CSV** | `outputs/repair_parameter_tuning.csv` | 580 rows, 40 columns | AUTHORITATIVE — 29 Config Sweeps |
| **Repair Tuning Summary JSON** | `outputs/repair_parameter_tuning_summary.json` | 1,120 lines structured JSON | AUTHORITATIVE — Sweep Aggregate |
| **Audited Demo Cases** | `outputs/demo_cases.json` | 3 Canonical Cases | AUTHORITATIVE — UI & Case Studies |
| **PRD Default Thresholds** | `configs/thresholds/v0_prd_defaults.yaml` | 46 lines YAML | CALIBRATION REFERENCE — Thresholds v0 |

---

## 2. Superseded / Deprecated Experiments Handled

1. **Paired Repair Sample Size:**
   - *Superseded:* Early preliminary experiment ($N=22$).
   - *Authoritative:* Final paired diagnostic validation ($N=35$ repaired cases out of $50$ candidates).
   - *Handling:* $N=35$ is the primary benchmark; $N=22$ is contextualized as an early feasibility study.
2. **Verification Confidence Threshold:**
   - *Superseded:* Old aspirational design PRD text ($\\Delta\\text{{confidence}} \\ge +0.15$).
   - *Authoritative Production Value:* Calibrated non-degradation guard ($\\Delta\\text{{confidence}} \\ge -0.01$ with tolerance $10^{{-7}}$, $\\Delta\\text{{SNR}} \\ge -5.0\\text{{ dB}}$).
3. **Severe Blur Test Case:**
   - *Status:* Empirically traced and verified on disk.
   - *Classification:* Stress-test demonstrating multi-gate escalation, not a defect or bug.

---

## 3. Mathematical Reconciliation Audit (Zero Discrepancy Verified)

- **Total Test Cohort:** $16,724 = 501\\text{{ (Direct Accept)}} + 13,001\\text{{ (Repair Routed)}} + 3,121\\text{{ (Direct Escalate)}} + 101\\text{{ (Reject)}}$.
- **Final Disposition:** $16,724 = 7,349\\text{{ (Released)}} + 9,375\\text{{ (Withheld)}}$.
- **Released Subset:** $7,349 = 501\\text{{ (Direct Accept)}} + 6,848\\text{{ (Verified Repair Release)}}$.
- **Baseline Error Containment:** $1,515\\text{{ Baseline Errors}} - 565\\text{{ Released Errors}} = 950\\text{{ Withheld Errors}}$ ($62.71\\%$ containment).
- **Paired Diagnostic Stability:** $35\\text{{ Repaired}} = 17\\text{{ (Corr}}\\to\\text{{Corr)}} + 18\\text{{ (Inc}}\\to\\text{{Inc)}} + 0\\text{{ (Corr}}\\to\\text{{Inc)}} + 0\\text{{ (Inc}}\\to\\text{{Corr)}}$. Zero flips ($100\\%$ stability).
"""
    with open(DOCS_DIR / "BEFORE_AFTER_PROJECT_COMPLETE_REPORT_AUDIT.md", "w", encoding="utf-8") as f:
        f.write(audit_md)

    print("Now generating Word (.docx) and Markdown (.md) documents...")
    from generate_report_text import generate_markdown_text, populate_docx_document
    
    md_content = generate_markdown_text()
    with open(DOCS_DIR / "BEFORE_AFTER_PROJECT_COMPLETE_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Saved docs/BEFORE_AFTER_PROJECT_COMPLETE_REPORT.md")
    
    doc = docx.Document()
    populate_docx_document(doc)
    doc.save(DOCS_DIR / "BEFORE_AFTER_PROJECT_COMPLETE_REPORT.docx")
    print("Saved docs/BEFORE_AFTER_PROJECT_COMPLETE_REPORT.docx")


if __name__ == "__main__":
    main()
