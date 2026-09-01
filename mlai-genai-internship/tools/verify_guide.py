"""Verify every python block in a session guide.

Narrative blocks run CUMULATIVELY, top to bottom, exactly as a notebook does.
Solution blocks must be SELF-CONTAINED and are run alone.
"""
import re, ast, pathlib, subprocess, sys, tempfile, os

SP = "/private/tmp/claude-501/-Users-ahk-Workspace-aiml/86131020-2bfd-490b-b9c3-e301ecd62195/scratchpad"
PRE = (pathlib.Path(f"{SP}/stdin_shim.py").read_text()
       + '\nimport matplotlib; matplotlib.use("Agg")\n'
       + pathlib.Path(f"{SP}/cache_shim.py").read_text() + "\n")
S = pathlib.Path("/Users/ahk/Workspace/aiml/mlai-genai-internship/student")

slug = sys.argv[1]
text = (S / "sessions" / f"{slug}.md").read_text()

# Markers for blocks that legitimately cannot run standalone here
SKIP = ("This is Session 4 code", "Not run here", "# illustrative:",
        "# needs-download:", "# api-only:", "# streamlit-only:", "# needs-install:")

# Which blocks sit inside a <details> Solutions section?
sol_spans = [(m.start(), m.end()) for m in
             re.finditer(r'<details><summary>Solutions</summary>.*?</details>', text, re.S)]
def in_solution(pos):
    return any(a <= pos <= b for a, b in sol_spans)

blocks = [(m.start(), m.group(1)) for m in re.finditer(r'```python\n(.*?)\n```', text, re.S)]
print(f"{len(blocks)} python blocks\n")

accumulated = PRE
syntax_bad, run_bad, ran, skipped = [], [], 0, 0

def run(src, timeout=90):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, cwd=tempfile.gettempdir())
        return r.returncode, (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    finally:
        os.unlink(path)

for pos, body in blocks:
    line = text[:pos].count("\n") + 1
    # A marked block is skipped before the syntax check: notebook cells may
    # legitimately contain `!pip ...`, which is not valid Python.
    if any(k in body for k in SKIP):
        skipped += 1; continue
    try:
        ast.parse(body)
    except SyntaxError as e:
        syntax_bad.append((line, str(e))); continue

    if in_solution(pos):
        code, err = run(PRE + "\n" + body)          # must stand alone
        if code: run_bad.append((line, "SOLUTION", err))
        else: ran += 1
    else:
        code, err = run(accumulated + "\n" + body)  # cumulative, like a notebook
        if code:
            run_bad.append((line, "narrative", err))
        else:
            ran += 1
            accumulated += "\n" + body

print(f"ran clean : {ran}")
print(f"skipped   : {skipped}")
print(f"syntax    : {len(syntax_bad)}")
for l, e in syntax_bad: print(f"   line {l}: {e}")
print(f"runtime   : {len(run_bad)}")
for l, kind, e in run_bad: print(f"   line {l:>5} [{kind}]: {e}")
sys.exit(1 if (syntax_bad or run_bad) else 0)
