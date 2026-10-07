"""
Assembler and builder for the complete technical project report in Markdown and DOCX formats.
Imports modular section text from report_sections_part1, report_sections_part2,
report_sections_part3, and report_sections_part4.
"""

from __future__ import annotations

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

from report_sections_part1 import (
    SECTION_1_TITLE, SECTION_1_BODY,
    SECTION_2_TITLE, SECTION_2_BODY,
    SECTION_3_TITLE, SECTION_3_BODY
)
from report_sections_part2 import (
    SECTION_4_TITLE, SECTION_4_BODY,
    SECTION_5_TITLE, SECTION_5_BODY,
    SECTION_6_TITLE, SECTION_6_BODY,
    SECTION_7_TITLE, SECTION_7_BODY,
    SECTION_8_TITLE, SECTION_8_BODY
)
from report_sections_part3 import (
    SECTION_9_TITLE, SECTION_9_BODY,
    SECTION_10_TITLE, SECTION_10_BODY,
    SECTION_11_TITLE, SECTION_11_BODY,
    SECTION_12_TITLE, SECTION_12_BODY,
    SECTION_13_TITLE, SECTION_13_BODY,
    SECTION_14_TITLE, SECTION_14_BODY,
    SECTION_15_TITLE, SECTION_15_BODY,
    SECTION_16_TITLE, SECTION_16_BODY
)
from report_sections_part4 import (
    SECTION_17_TITLE, SECTION_17_BODY,
    SECTION_18_TITLE, SECTION_18_BODY,
    SECTION_19_TITLE, SECTION_19_BODY,
    SECTION_20_TITLE, SECTION_20_BODY,
    SECTION_21_TITLE, SECTION_21_BODY,
    SECTION_22_TITLE, SECTION_22_BODY,
    SECTION_23_TITLE, SECTION_23_BODY,
    SECTION_24_TITLE, SECTION_24_BODY,
    SECTION_25_TITLE, SECTION_25_BODY,
    SECTION_26_TITLE, SECTION_26_BODY,
    SECTION_QA_TITLE, SECTION_QA_BODY
)

ALL_SECTIONS = [
    (SECTION_1_TITLE, SECTION_1_BODY),
    (SECTION_2_TITLE, SECTION_2_BODY),
    (SECTION_3_TITLE, SECTION_3_BODY),
    (SECTION_4_TITLE, SECTION_4_BODY),
    (SECTION_5_TITLE, SECTION_5_BODY),
    (SECTION_6_TITLE, SECTION_6_BODY),
    (SECTION_7_TITLE, SECTION_7_BODY),
    (SECTION_8_TITLE, SECTION_8_BODY),
    (SECTION_9_TITLE, SECTION_9_BODY),
    (SECTION_10_TITLE, SECTION_10_BODY),
    (SECTION_11_TITLE, SECTION_11_BODY),
    (SECTION_12_TITLE, SECTION_12_BODY),
    (SECTION_13_TITLE, SECTION_13_BODY),
    (SECTION_14_TITLE, SECTION_14_BODY),
    (SECTION_15_TITLE, SECTION_15_BODY),
    (SECTION_16_TITLE, SECTION_16_BODY),
    (SECTION_17_TITLE, SECTION_17_BODY),
    (SECTION_18_TITLE, SECTION_18_BODY),
    (SECTION_19_TITLE, SECTION_19_BODY),
    (SECTION_20_TITLE, SECTION_20_BODY),
    (SECTION_21_TITLE, SECTION_21_BODY),
    (SECTION_22_TITLE, SECTION_22_BODY),
    (SECTION_23_TITLE, SECTION_23_BODY),
    (SECTION_24_TITLE, SECTION_24_BODY),
    (SECTION_25_TITLE, SECTION_25_BODY),
    (SECTION_26_TITLE, SECTION_26_BODY),
    (SECTION_QA_TITLE, SECTION_QA_BODY)
]


def generate_markdown_text() -> str:
    """Combines all modular sections into a single authoritative markdown report."""
    md_lines = [
        "# BEFORE vs AFTER: Project Improvement, Method Justification and Experimental Evaluation",
        "## Reliability-Aware Multi-Agent System for Chest X-Ray Pneumonia Classification",
        "",
        "**Document Type:** Comprehensive Academic & Technical Project Report  ",
        "**Evaluation Date:** October 6, 2026  ",
        "**Target Benchmark:** NIH ChestX-ray14 (Full Test Cohort, N=16,724)  ",
        "**Diagnostic Engine:** TorchXRayVision DenseNet-121 (`densenet121-res224-nih`)  ",
        "**Operating Threshold:** tau = 0.522161 (Calibrated F1-Optimal Operating Point)  ",
        "**Project Scope:** Medical Artificial Intelligence Supervisory Governance & Active Reliability Architecture  ",
        "",
        "---",
        "",
        "## Table of Contents",
        ""
    ]
    
    for idx, (title, _) in enumerate(ALL_SECTIONS, 1):
        md_lines.append(f"{idx}. [{title}](#{title.lower().replace(' ', '-').replace('—', '').replace(':', '').replace('(', '').replace(')', '').replace('/', '').replace('&', '').replace('?', '')})")
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    for title, body in ALL_SECTIONS:
        md_lines.append(f"## {title}")
        md_lines.append("")
        # Clean leading/trailing whitespaces in body lines
        body_cleaned = "\n".join(line.rstrip() for line in body.strip().split("\n"))
        md_lines.append(body_cleaned)
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
    return "\n".join(md_lines)


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
    p.paragraph_format.line_spacing = 1.15
    run_t = p.add_run(f"📌 {title}: ")
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(9.5)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    run_body = p.add_run(text)
    run_body.font.name = "Calibri"
    run_body.font.size = Pt(9.5)
    run_body.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(3)


def create_styled_table(doc, headers: list[str], rows: list[list[str]], col_widths: list[Inches]):
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    hdr_row = tbl.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.width = col_widths[i]
        set_cell_background(cell, "0F172A")
        set_cell_margins(cell, top=90, bottom=90, left=110, right=110)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(h)
        run.font.name = "Calibri"
        run.font.size = Pt(8.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
    for r_idx, r_data in enumerate(rows):
        row = tbl.rows[r_idx + 1]
        bg = "F8FAFC" if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(r_data):
            cell = row.cells[c_idx]
            cell.width = col_widths[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=65, bottom=65, left=110, right=110)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(str(val))
            run.font.name = "Calibri"
            run.font.size = Pt(8.0)
            run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
            
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(3)
    return tbl


def populate_docx_document(doc):
    """Generates the full-featured, professional academic Word report."""
    # Set standard 1-inch margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    # --- COVER PAGE ---
    cover_title_p = doc.add_paragraph()
    cover_title_p.paragraph_format.space_before = Pt(72)
    cover_title_p.paragraph_format.space_after = Pt(6)
    r_main = cover_title_p.add_run("BEFORE vs AFTER")
    r_main.font.name = "Calibri"
    r_main.font.size = Pt(28)
    r_main.font.bold = True
    r_main.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    cover_sub_p = doc.add_paragraph()
    cover_sub_p.paragraph_format.space_before = Pt(0)
    cover_sub_p.paragraph_format.space_after = Pt(14)
    r_sub = cover_sub_p.add_run("Project Improvement, Method Justification and Experimental Evaluation")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(16)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)
    
    cover_tag_p = doc.add_paragraph()
    cover_tag_p.paragraph_format.space_before = Pt(0)
    cover_tag_p.paragraph_format.space_after = Pt(36)
    r_tag = cover_tag_p.add_run("Reliability-Aware Multi-Agent System for Chest X-Ray Pneumonia Classification")
    r_tag.font.name = "Calibri"
    r_tag.font.size = Pt(13)
    r_tag.font.italic = True
    r_tag.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    
    # Meta Box Table on Cover Page
    meta_headers = ["Project Metadata Field", "Authoritative Specification"]
    meta_rows = [
        ["Document Classification", "Comprehensive Academic & Faculty-Facing Technical Report"],
        ["Target Benchmark Dataset", "NIH ChestX-ray14 (Full Frozen Test Partition, N=16,724 images)"],
        ["Disease Diagnostic Backbone", "TorchXRayVision DenseNet-121 (densenet121-res224-nih, Frozen Weights)"],
        ["Clinical Operating Point", "tau = 0.522161 (Calibrated F1-Optimal Operating Threshold)"],
        ["Supervisory Architecture", "7-Agent Multi-Agent Active Reliability Pipeline"],
        ["Evaluation Date", "October 6, 2026"],
        ["Project Classification", "Medical AI Safety & Reliability Research Prototype"]
    ]
    create_styled_table(doc, meta_headers, meta_rows, [Inches(2.5), Inches(4.0)])
    
    # Core Paradigm Shift Callout on Cover Page
    add_callout(
        doc,
        "BEFORE: 'What is the model prediction?' → AFTER: 'Does this radiograph meet the requisite quality, distribution, and confidence conditions for its prediction to be safely released?' The system transitions medical AI from prediction-only black-box inference to an evidence-governed supervisory framework.",
        title="CORE RESEARCH PARADIGM SHIFT",
        border_hex="0284C7",
        bg_hex="F0F9FF"
    )
    
    doc.add_page_break()
    
    # --- TABLE OF CONTENTS SECTION ---
    toc_p = doc.add_paragraph()
    toc_p.paragraph_format.space_before = Pt(12)
    toc_p.paragraph_format.space_after = Pt(12)
    r_toc = toc_p.add_run("Table of Contents")
    r_toc.font.name = "Calibri"
    r_toc.font.size = Pt(16)
    r_toc.font.bold = True
    r_toc.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    toc_headers = ["No.", "Section Title", "Core Focus & Evaluative Scope"]
    toc_rows = [
        ["1", "Executive Summary", "The Core Paradigm Shift: From Prediction-Only to Reliability-Aware AI"],
        ["2", "Complete Architecture", "Seven-Agent Functional Topology and End-to-End Pipeline Dataflow"],
        ["3", "Dataset & Baseline", "NIH ChestX-ray14 Split Protocol and Frozen DenseNet-121 Benchmark"],
        ["4", "Agent 1: Quality Agent", "Signal Processing Rationale (Laplacian, SNR, Histogram) and Full-Cohort Results"],
        ["5", "Agent 2: DenseNet-121", "Diagnostic Backbone and 1024-D Feature Vector Extraction"],
        ["6", "Agent 3: Calibration & Uncertainty", "Platt Scaling Calibration, Confidence Metrics, and Normalized Shannon Entropy"],
        ["7", "Agent 4: OOD Detection", "Mahalanobis Distance via Cholesky Decomposition and Distribution Screening"],
        ["8", "Agent 5: Decision Agent", "Rule Precedence Hierarchy (R1–R7) and Routing vs Disposition Reconciliation"],
        ["9", "Agent 6: Image Repair", "Restorative Algorithms (CLAHE, NLMeans, Unsharp Masking) and Design Rationales"],
        ["10", "Repair Parameter Tuning", "Rigorous Evaluation Across 29 Algorithmic Configurations and 580 Inference Runs"],
        ["11", "Final Paired Diagnostic Validation", "Authoritative N=35 Paired Benchmark and Diagnostic Correctness Transition Matrix"],
        ["12", "Repair by Defect Cohort", "Stratified Empirical Findings across Blur, Noise, and Exposure Modes"],
        ["13", "Deliberately Blurred Stress Test", "Empirical Proof of Non-Forced Acceptance and Verification Escalation"],
        ["14", "Agent 7: Verification Agent", "Five Mandatory Post-Repair Safety Gates and Escalation Rationale"],
        ["15", "Selective Release Architecture", "Safe Automated Coverage (43.94%) vs Human Review Triage (56.06%)"],
        ["16", "Error Containment Analysis", "Containment of 950 Baseline Errors (62.71% Containment Rate)"],
        ["17", "Overall Before vs After", "Comprehensive 17-Row Comparison Matrix and Population Disambiguation"],
        ["18", "What Each Agent Improved", "Detailed Analytical Narrative of Individual Agent Contributions"],
        ["19", "Evaluation Methodology", "Task-Specific Objective Evaluation vs Generic Accuracy"],
        ["20", "Demonstration Case Studies", "Four Canonical Case Studies Tracing Realistic Clinical Trajectories"],
        ["21", "What the Project Proves", "Thirteen Rigorously Supported Empirical Claims"],
        ["22", "What the Project Does NOT Prove", "Eight Explicit Negative Boundaries and Scientific Limitations"],
        ["23", "Limitations & Future Work", "Class Imbalance, Label Noise, and Clinical Translation Frontiers"],
        ["24", "Method Selection Rationale", "Consolidated Matrix of Algorithmic Trade-offs and Justifications"],
        ["25", "Faculty Presentation Pitch", "Concise Three-Minute Oral Defense Summary"],
        ["26", "Final Conclusion", "Architectural Synthesis: Prediction-Only AI to Reliability-Aware AI"],
        ["Q&A", "Faculty Defense Q&A", "Definitive Technical Answers to 30 Rigorous Faculty Questions"]
    ]
    create_styled_table(doc, toc_headers, toc_rows, [Inches(0.5), Inches(2.7), Inches(3.3)])
    
    doc.add_page_break()
    
    # --- BODY SECTIONS (1 TO 26 + Q&A) ---
    for title, body in ALL_SECTIONS:
        # Heading 1
        h1 = doc.add_paragraph()
        h1.paragraph_format.space_before = Pt(16)
        h1.paragraph_format.space_after = Pt(4)
        h1.paragraph_format.keep_with_next = True
        r_h1 = h1.add_run(title)
        r_h1.font.name = "Calibri"
        r_h1.font.size = Pt(13)
        r_h1.font.bold = True
        r_h1.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        
        # Paragraphs
        paragraphs = body.strip().split("\n\n")
        for p_text in paragraphs:
            p_text = p_text.strip()
            if not p_text:
                continue
                
            # Check if this paragraph is a callout / takeaway candidate
            if p_text.startswith("CRITICAL") or p_text.startswith("KEY TAKEAWAY") or p_text.startswith("IMPORTANT"):
                colon_idx = p_text.find(":")
                callout_title = p_text[:colon_idx].strip() if colon_idx != -1 else "CRITICAL FINDING"
                callout_body = p_text[colon_idx+1:].strip() if colon_idx != -1 else p_text
                add_callout(doc, callout_body, title=callout_title, border_hex="DC2626", bg_hex="FEF2F2")
                continue
                
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            
            # Sub-headers detection inside body
            if p_text.startswith("STAGE ") or p_text.startswith("QUESTION ") or p_text.startswith("CASE ") or (len(p_text) < 60 and p_text.isupper()):
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.keep_with_next = True
                run = p.add_run(p_text)
                run.font.name = "Calibri"
                run.font.size = Pt(10.5)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0x02, 0x84, 0xC7)
            else:
                # Regular paragraph text
                run = p.add_run(p_text)
                run.font.name = "Calibri"
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
                
    # Footer with document metadata
    for s in doc.sections:
        footer = s.footer
        p_ft = footer.paragraphs[0]
        p_ft.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_ft = p_ft.add_run("Reliability-Aware CXR Multi-Agent System — Academic Evaluation Report")
        r_ft.font.name = "Calibri"
        r_ft.font.size = Pt(8.5)
        r_ft.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        
    print("Populated Word document content with cover, TOC, 27 sections, tables, and callouts.")
