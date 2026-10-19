"""Render the plain-language report to a Word .docx.

Handles the subset of Markdown the report actually uses: ATX headings, pipe tables,
fenced code blocks, bullet lists, images, horizontal rules, and inline **bold** /
*italic* / `code`.
"""
import io
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

SRC = Path(__file__).resolve().parent.parent / "v7_1_reject_stages_report.md"
DST = SRC.with_suffix(".docx")
ROOT = SRC.parent

ACCENT = RGBColor(0x1F, 0x4E, 0x9C)
MUTED = RGBColor(0x55, 0x55, 0x55)
CODE_BG = "F2F4F7"


# ---------------------------------------------------------------- helpers
def shade(el, fill):
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), fill)
    el.get_or_add_pPr().append(sh)


def cell_shade(cell, fill):
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(sh)


def hrule(par):
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "CCCCCC")
    pbdr.append(bottom)
    par._p.get_or_add_pPr().append(pbdr)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\n]+?\*)", re.S)


def add_inline(par, text, base_size=None, base_color=None,
               bold=False, italic=False):
    """Write text into a paragraph, honouring nested **bold**, *italic* and `code`."""
    text = text.replace(" ", " ")
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            add_inline(par, piece[2:-2], base_size, base_color, True, italic)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            r = par.add_run(piece[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt((base_size or 10.5) - 1)
            r.bold, r.italic = bold, italic
            if base_color is not None:
                r.font.color.rgb = base_color
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            add_inline(par, piece[1:-1], base_size, base_color, bold, True)
        else:
            r = par.add_run(piece)
            r.bold, r.italic = bold, italic
            if base_size:
                r.font.size = Pt(base_size)
            if base_color is not None:
                r.font.color.rgb = base_color
    return par


def plain(text):
    """Heading text with inline markers removed (headings carry their own weight)."""
    return re.sub(r"\*\*|`|(?<![A-Za-z0-9])\*|\*(?![A-Za-z0-9])", "", text)


def strip_md(text):
    """Plain text with the inline markers removed (for table cells that need it raw)."""
    return text


# ---------------------------------------------------------------- document
doc = Document()

st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(10.5)
st.paragraph_format.space_after = Pt(7)
st.paragraph_format.line_spacing = 1.12

for name, size, color, before, after in [
    ("Heading 1", 19, ACCENT, 20, 8),
    ("Heading 2", 14.5, ACCENT, 16, 6),
    ("Heading 3", 12, RGBColor(0x2F, 0x4F, 0x7A), 12, 4),
]:
    s_ = doc.styles[name]
    s_.font.name = "Calibri"
    s_.font.size = Pt(size)
    s_.font.color.rgb = color
    s_.font.bold = True
    s_.paragraph_format.space_before = Pt(before)
    s_.paragraph_format.space_after = Pt(after)
    s_.paragraph_format.keep_with_next = True

for sec in doc.sections:
    sec.top_margin = Inches(0.85)
    sec.bottom_margin = Inches(0.85)
    sec.left_margin = Inches(0.95)
    sec.right_margin = Inches(0.95)

USABLE = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin

lines = io.open(SRC, encoding="utf-8").read().split("\n")
i = 0
n_tables = n_figs = 0

while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    # ---- fenced code -------------------------------------------------
    if stripped.startswith("```"):
        i += 1
        buf = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            buf.append(lines[i])
            i += 1
        i += 1
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.left_indent = Inches(0.2)
        shade(p._p, CODE_BG)
        r = p.add_run("\n".join(buf))
        r.font.name = "Consolas"
        r.font.size = Pt(9)
        continue

    # ---- image -------------------------------------------------------
    m = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", stripped)
    if m:
        path = (ROOT / m.group(2)).resolve()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        p.add_run().add_picture(str(path), width=USABLE)
        n_figs += 1
        i += 1
        continue

    # ---- table -------------------------------------------------------
    if stripped.startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        header, body = rows[0], rows[2:]
        ncol = len(header)
        t = doc.add_table(rows=1 + len(body), cols=ncol)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = True
        for c, h in enumerate(header):
            cell = t.cell(0, c)
            cell.text = ""
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            par.paragraph_format.space_before = Pt(2)
            add_inline(par, h, base_size=9.5)
            for run in par.runs:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            cell_shade(cell, "1F4E9C")
        for r_, row in enumerate(body, start=1):
            for c in range(ncol):
                cell = t.cell(r_, c)
                cell.text = ""
                par = cell.paragraphs[0]
                par.paragraph_format.space_after = Pt(2)
                par.paragraph_format.space_before = Pt(2)
                add_inline(par, row[c] if c < len(row) else "", base_size=9.5)
            if r_ % 2 == 0:
                for c in range(ncol):
                    cell_shade(t.cell(r_, c), "F4F7FB")
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
        n_tables += 1
        continue

    # ---- headings ----------------------------------------------------
    if stripped.startswith("#"):
        lvl = len(stripped) - len(stripped.lstrip("#"))
        text = plain(stripped[lvl:].strip())
        if lvl == 1:
            p = doc.add_paragraph(style="Title" if "Title" in doc.styles else None)
            r = p.add_run(text)
            r.font.size = Pt(24)
            r.font.bold = True
            r.font.color.rgb = ACCENT
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(4)
        else:
            doc.add_heading(text, level=min(lvl - 1, 3) if lvl > 1 else 1)
        i += 1
        continue

    # ---- horizontal rule ---------------------------------------------
    if stripped == "---":
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(8)
        hrule(p)
        i += 1
        continue

    # ---- blank -------------------------------------------------------
    if not stripped:
        i += 1
        continue

    # ---- bullet ------------------------------------------------------
    if re.match(r"^[-*]\s+", stripped):
        buf = re.sub(r"^[-*]\s+", "", stripped)
        i += 1
        while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                and not re.match(r"^\s*[-*]\s+", lines[i]) and not lines[i].strip().startswith("|"):
            buf += " " + lines[i].strip()
            i += 1
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        add_inline(p, buf)
        continue

    # ---- numbered ----------------------------------------------------
    if re.match(r"^\d+\.\s+", stripped):
        buf = re.sub(r"^\d+\.\s+", "", stripped)
        i += 1
        while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
            buf += " " + lines[i].strip()
            i += 1
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(4)
        add_inline(p, buf)
        continue

    # ---- paragraph (join soft-wrapped lines) -------------------------
    buf = [stripped]
    i += 1
    while i < len(lines):
        nxt = lines[i].strip()
        if (not nxt or nxt.startswith("#") or nxt.startswith("|") or nxt == "---"
                or nxt.startswith("```") or re.match(r"^[-*]\s+", nxt)
                or re.match(r"^\d+\.\s+", nxt) or nxt.startswith("![")):
            break
        buf.append(nxt)
        i += 1
    text = " ".join(buf)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    # figure captions and the "source notebook" footer get a muted look
    if text.startswith("**Figure ") or text.startswith("*Source notebook"):
        add_inline(p, text, base_size=9.5, base_color=MUTED)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if text.startswith("**Figure ") \
            else WD_ALIGN_PARAGRAPH.LEFT
    else:
        add_inline(p, text)

doc.save(DST)
print(f"wrote {DST}  ({n_tables} tables, {n_figs} figures)")
