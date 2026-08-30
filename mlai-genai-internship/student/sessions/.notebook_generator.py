"""Build a snippet notebook that mirrors a session guide section for section.

Rules:
  - every heading starts a new markdown cell, with the prose that follows it
  - every python block becomes its own code cell
  - a ```text block straight after a python block is that code's expected
    output, and is attached as a small "Output" markdown cell
  - MCQ / task / checklist sections are replaced by a pointer to the guide
"""
import re, sys, pathlib
sys.path.insert(0, "/private/tmp/claude-501/-Users-ahk-Workspace-aiml/86131020-2bfd-490b-b9c3-e301ecd62195/scratchpad")
from mknb2 import build

S = pathlib.Path("/Users/ahk/Workspace/aiml/mlai-genai-internship/student")
slug, title, subtitle, intro = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
text = (S / "sessions" / f"{slug}.md").read_text()

def fix_links(md):
    md = re.sub(r'\]\((session-\d[^)]*\.md)', r'](../sessions/\1', md)
    md = md.replace("](images/", "](../sessions/images/")
    md = md.replace("](README.md)", "](../sessions/README.md)")
    return md

# Remove the exercise sections - they live in the guide, not the notebook.
# Handles both layouts: one MCQ block at the end, or one per topic.
def strip_sections(md):
    """Drop every Tasks / Practice / MCQ section up to the next heading.

    A '# ...' line inside a fenced code block is a comment, not a heading,
    so fences must be tracked or the skip stops in the middle of a solution.
    """
    out, i, fence = [], 0, False
    lines = md.split("\n")
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            fence = not fence
        m = None if fence else re.match(r'^(#{1,3}) (?:✏️|❓|🎯)', line)
        if m:
            level = len(m.group(1))
            i += 1
            inner = False
            while i < len(lines):
                if lines[i].startswith("```"):
                    inner = not inner
                if not inner:
                    h = re.match(r'^(#{1,3}) ', lines[i])
                    if h and len(h.group(1)) <= level:
                        break
                i += 1
            fence = False          # the skipped block closed its own fences
            continue
        out.append(line)
        i += 1
    return "\n".join(out)

text = strip_sections(text)

# The guide's H1 and frontmatter table duplicate the notebook header
first_section = min((i for i in (text.find("\n## 🎯"), text.find("\n## How this"))
                     if i > 0), default=0)
if first_section:
    text = text[first_section:]

cells = [("md", intro)]
pos = 0
buf = []          # markdown accumulating for the current cell

def flush():
    global buf
    md = "\n\n".join(x.strip() for x in buf if x.strip())
    md = md.strip().strip("-").strip()          # a lone --- carries nothing
    if md:
        cells.append(("md", fix_links(md)))
    buf = []

# Walk the document: headings, python blocks and text blocks, in order
TOKEN = re.compile(
    r'(?P<head>^\#{1,3} [^\n]+$)'
    r'|(?P<py>^```python\n.*?^```$)'
    r'|(?P<out>^```text\n.*?^```$)',
    re.M | re.S)

last_was_code = False
for m in TOKEN.finditer(text):
    between = text[pos:m.start()]
    pos = m.end()

    if between.strip():
        buf.append(between)
        last_was_code = False

    if m.group("head"):
        level = len(m.group("head")) - len(m.group("head").lstrip("#"))
        # A new section heading closes the previous cell
        if level <= 2:
            flush()
        buf.append(m.group("head"))
        last_was_code = False

    elif m.group("py"):
        flush()
        code = m.group("py")[len("```python\n"):-len("```")].rstrip()
        if code.lstrip().startswith("# illustrative:"):
            cells.append(("md", "```python\n" + code + "\n```"))
            last_was_code = False
        else:
            cells.append(("code", code))
            last_was_code = True

    elif m.group("out"):
        block = m.group("out")
        if last_was_code and not buf:
            cells.append(("md", "**Output:**\n\n" + block))
        else:
            buf.append(block)
        last_was_code = False

tail = text[pos:]
if tail.strip():
    buf.append(tail)
flush()

cells.append(("md", f"""---

# Next steps

**The 20 MCQs and 20 preprocessing tasks are in the guide:**
[Session 3 guide]({f"../sessions/{slug}.md"})

| | |
|---|---|
| **Previous** | [Session 2 — NumPy, Pandas & Visualisation](../sessions/session-02-numpy-pandas.md) |
| **Next** | [Session 4 — Introduction to AI & ML](../sessions/session-04-intro-ml-ai.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |"""))

build(f"{slug}.ipynb", title, subtitle, cells, execute=False)
