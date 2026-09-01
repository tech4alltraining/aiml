"""Write a session notebook from a list of (kind, source) cells.

kind is "md" or "code". `execute` is accepted for backward compatibility and
is always False in this course: the guides carry the outputs, the notebooks
carry the snippets.
"""
import json, pathlib, uuid

OUT = pathlib.Path("/Users/ahk/Workspace/aiml/mlai-genai-internship/student/notebooks")
REPO = ("https://colab.research.google.com/github/tech4alltraining/aiml/blob/main/"
        "mlai-genai-internship/student/notebooks/")

META = {
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"display_name": "Python (genai)", "language": "python", "name": "genai"},
    "language_info": {"name": "python", "version": "3.12"},
}

def _cell(kind, src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    c = {"cell_type": "markdown" if kind == "md" else "code",
         "id": uuid.uuid4().hex[:12], "metadata": {}, "source": source}
    if kind != "md":
        c["execution_count"] = None
        c["outputs"] = []
    return c

def build(filename, title, subtitle, cells, execute=False):
    header = (f"# {title}\n\n"
              f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
              f"({REPO}{filename})\n\n{subtitle}")
    all_cells = [_cell("md", header)] + [_cell(k, s) for k, s in cells]
    nb = {"cells": all_cells, "metadata": META, "nbformat": 4, "nbformat_minor": 5}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / filename).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    n_code = sum(1 for c in all_cells if c["cell_type"] == "code")
    print(f"  {filename}: {len(all_cells)} cells ({n_code} code)")
