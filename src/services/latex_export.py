import re

from src.models.schemas import LiteratureReview


LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{setspace}
\usepackage{parskip}
\onehalfspacing
\usepackage{ctex}

\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\title{%s}
\author{Automated Literature Review Platform}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

"""

LATEX_POSTAMBLE = r"""
\end{document}
"""


def escape_latex(text: str) -> str:
    """转义 LaTeX 特殊字符，保留中文和常见数学符号。"""
    # 必须先转义 \，避免后续替换被影响
    text = text.replace("\\", "\\textbackslash{}")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("$", "\\$")
    text = text.replace("&", "\\&")
    text = text.replace("#", "\\#")
    text = text.replace("%", "\\%")
    text = text.replace("~", "\\textasciitilde{}")
    text = text.replace("^", "\\textasciicircum{}")
    # 下划线在文本中常见（如 model_name），保留\_转义
    text = text.replace("_", "\\_")
    return text


def content_to_latex(content: str) -> str:
    """将段落文本转为 LaTeX：分段 + 简单 markdown 转换。"""
    escaped = escape_latex(content)

    # 加粗 **text** → \textbf{text}
    escaped = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", escaped)
    # 斜体 *text* → \textit{text}
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", escaped)

    paragraphs = [p.strip() for p in escaped.split("\n\n") if p.strip()]
    return "\n\n".join(paragraphs)


def section_to_latex(section, level: int = 1) -> str:
    """将 ReviewSection 递归转为 LaTeX"""
    lines = []
    cmd = "section" if level == 1 else "subsection" if level == 2 else "subsubsection"

    title = escape_latex(section.title)

    lines.append(f"\\{cmd}{{{title}}}")
    if section.content:
        lines.append(content_to_latex(section.content))
    for sub in section.subsections:
        lines.append(section_to_latex(sub, level + 1))

    return "\n\n".join(lines)


def section_to_markdown(section, level: int = 2) -> str:
    """将 ReviewSection 递归转为 Markdown"""
    lines = []
    prefix = "#" * level
    lines.append(f"{prefix} {section.title}")
    if section.content:
        lines.append(section.content)
    for sub in section.subsections:
        lines.append(section_to_markdown(sub, level + 1))
    return "\n\n".join(lines)


def review_to_markdown(review: LiteratureReview) -> str:
    """将 LiteratureReview 导出为 Markdown"""
    parts = [f"# {review.title}\n"]
    for section in review.sections:
        parts.append(section_to_markdown(section, 2))
    if review.references:
        refs = "\n".join(f"{i+1}. {r.replace(chr(10), ' ').replace(chr(13), ' ')}" for i, r in enumerate(review.references))
        parts.append(f"## 参考文献\n\n{refs}")
    return "\n\n".join(parts)


def review_to_latex(review: LiteratureReview) -> str:
    """将完整的 LiteratureReview 导出为 LaTeX 源码"""
    title = escape_latex(review.title)
    parts = [LATEX_PREAMBLE % title]

    for section in review.sections:
        parts.append(section_to_latex(section))

    if review.references:
        ref_items = "\n".join(
            f"\\bibitem{{{i + 1}}} {escape_latex(r.replace(chr(10), ' ').replace(chr(13), ' '))}"
            for i, r in enumerate(review.references)
        )
        parts.append(
            f"\\begin{{thebibliography}}{{{len(review.references)}}}\n\n"
            f"{ref_items}\n\n"
            f"\\end{{thebibliography}}"
        )

    parts.append(LATEX_POSTAMBLE)
    return "\n\n".join(parts)
