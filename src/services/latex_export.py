from src.models.schemas import LiteratureReview


LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{setspace}
\onehalfspacing

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
    """转义 LaTeX 特殊字符"""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("$", "\\$"),
        ("&", "\\&"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("%", "\\%"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def section_to_latex(section, level: int = 1) -> str:
    """将 ReviewSection 递归转为 LaTeX"""
    lines = []
    cmd = "section" if level == 1 else "subsection" if level == 2 else "subsubsection"

    title = escape_latex(section.title)
    content = escape_latex(section.content)

    lines.append(f"\\{cmd}{{{title}}}")
    if content:
        lines.append(content)
    for sub in section.subsections:
        lines.append(section_to_latex(sub, level + 1))

    return "\n\n".join(lines)


def review_to_latex(review: LiteratureReview) -> str:
    """将完整的 LiteratureReview 导出为 LaTeX 源码"""
    title = escape_latex(review.title)
    parts = [LATEX_PREAMBLE % title]

    for section in review.sections:
        parts.append(section_to_latex(section))

    parts.append(LATEX_POSTAMBLE)
    return "\n\n".join(parts)
