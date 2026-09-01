"""Consistency check for one session: notebook, solutions, MCQs, links."""
import sys, re, pathlib, ast, subprocess, tempfile, os
import nbformat

SHIM_PATH = ("/private/tmp/claude-501/-Users-ahk-Workspace-aiml/86131020-2bfd-490b-b9c3-e301ecd62195/scratchpad/stdin_shim.py")

S = pathlib.Path("/Users/ahk/Workspace/aiml/mlai-genai-internship/student")
slug = sys.argv[1]
guide = S / "sessions" / f"{slug}.md"
nbp   = S / "notebooks" / f"{slug}.ipynb"
fails = 0

# --- notebook ---
if nbp.exists():
    nb = nbformat.read(nbp, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    errs = [o for c in code for o in c.outputs if o.output_type == "error"]
    imgs = sum(1 for c in code for o in c.outputs if "image/png" in o.get("data", {}))
    print(f"notebook      : {len(nb.cells)} cells, {len(code)} code, "
          f"{sum(1 for c in code if c.outputs)} with output, {imgs} charts, {len(errs)} errors")
    for o in errs:
        print("   ", o.get("ename"), o.get("evalue")); fails += 1
else:
    print("notebook      : MISSING"); fails += 1

text = guide.read_text()

# --- solution blocks actually run ---
sols = re.findall(r'<details><summary>Solutions</summary>\n\n```python\n(.*?)```', text, re.S)
ok = bad = skipped = 0
for i, s in enumerate(sols, 1):
    try:
        ast.parse(s)
    except SyntaxError as e:
        print(f"   solution {i} SYNTAX: {e}"); bad += 1; continue
    head = s.splitlines()[0]
    if any(k in head for k in ("# streamlit-only:", "# api-only:", "# needs-download:", "# needs-install:")):
        skipped += 1; continue          # needs a runtime/API key; syntax-checked above
    # Blocks that prompt a user get a canned input() so they can still run.
    shim = pathlib.Path(SHIM_PATH).read_text() if "input(" in s else ""
    # plt.show() blocks on a GUI window; force a headless backend so
    # plotting code is actually verified rather than hanging the check.
    if "matplotlib" in s or "seaborn" in s or "plt." in s:
        shim = 'import matplotlib; matplotlib.use("Agg")\n' + shim
    # Serve course dataset URLs from the local repo so checks stay fast.
    if "raw.githubusercontent" in s:
        shim = pathlib.Path(SHIM_PATH).parent.joinpath("cache_shim.py").read_text() + "\n" + shim
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(shim + "\n" + s); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, cwd=str(S), timeout=60)
        if r.returncode:
            print(f"   solution {i} FAILED: {r.stderr.strip().splitlines()[-1]}"); bad += 1
        else:
            ok += 1
    except subprocess.TimeoutExpired:
        print(f"   solution {i} TIMEOUT (possible infinite loop)"); bad += 1
    finally:
        os.unlink(path)
print(f"solutions     : {ok} run, {bad} failed" + (f", {skipped} syntax-only (needs runtime/API key)" if skipped else ""))
fails += bad

# --- MCQs: every question answered ---
blocks = re.split(r'\n#{1,2} ❓ [^\n]*', text)[1:]
qs = [len(re.findall(r'\*\*Q\d+\.\*\*', b)) for b in blocks]
as_ = [len(re.findall(r'\*\*A\d+ —', b)) for b in blocks]
mism = [i for i, (q, a) in enumerate(zip(qs, as_), 1) if q != a]
print(f"MCQs          : {len(blocks)} blocks, {sum(qs)} questions, {sum(as_)} answers"
      + (f"  MISMATCH in block(s) {mism}" if mism else ""))
fails += len(mism)

# --- structure counts ---
practice = text.count('## ✏️ Practice') + text.count('## ✏️ Tasks')
tasks = text.count('## 🎯 Tasks') + text.count('## 🌍 Scenarios')
print(f"structure     : {text.count('## 📘 Examples')} examples, "
      f"{practice} practice/task blocks, {tasks} task/scenario blocks")

# --- links ---
bad_links = []
def scan(src_dir, s, label):
    for m in re.finditer(r'\[([^\]]*)\]\(([^)\s]+)\)', s):
        t = m.group(2)
        if t.startswith(("http", "#", "mailto:")): continue
        p = t.partition("#")[0]
        if p and not (src_dir / p).resolve().exists():
            bad_links.append(f"{label} -> {t}")
scan(guide.parent, text, slug + ".md")
if nbp.exists():
    for c in nbformat.read(nbp, as_version=4).cells:
        if c.cell_type == "markdown": scan(nbp.parent, c.source, slug + ".ipynb")
print(f"links         : {len(bad_links)} broken")
for b in bad_links: print("   ", b)
fails += len(bad_links)

# --- nested fences: a literal ``` inside a python block breaks the fence ---
nested = [i for i, b in enumerate(
    re.findall(r'```python\n(.*?)\n```', text, re.S), 1) if "`" * 3 in b]
print(f"nested fences : {'none' if not nested else f'IN BLOCK(S) {nested}'}")
fails += len(nested)

# --- fences ---
f = sum(1 for l in text.split("\n") if l.startswith("```"))
print(f"fences        : {'balanced' if f % 2 == 0 else 'UNBALANCED'}")
if f % 2: fails += 1

print("\n" + ("ALL CHECKS PASSED" if fails == 0 else f"{fails} PROBLEM(S)"))
sys.exit(1 if fails else 0)
