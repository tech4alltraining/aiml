"""Repo-wide consistency check for the internship course."""
import json, pathlib, re, sys
S = pathlib.Path("/Users/ahk/Workspace/aiml/mlai-genai-internship/student")
ROOT = pathlib.Path("/Users/ahk/Workspace/aiml")
problems, notes = [], []

guides = sorted((S / "sessions").glob("session-*.md"))
print(f"{len(guides)} session guides\n")

def anchor(h):
    a = re.sub(r"[^\w\s-]", "", h.lower()).strip().replace(" ", "-")
    return a

# ---------- 1. guide <-> notebook pairing ----------
print("--- 1. guide / notebook pairing ---")
for g in guides:
    nb = S / "notebooks" / (g.stem + ".ipynb")
    if not nb.exists():
        problems.append(f"missing notebook for {g.name}")
    else:
        print(f"  ok  {g.stem}")
orphans = [n for n in (S / "notebooks").glob("session-*.ipynb")
           if not (S / "sessions" / (n.stem + ".md")).exists()]
for o in orphans:
    problems.append(f"orphan notebook with no guide: {o.name}")

# ---------- 2. prev / next chain ----------
print("\n--- 2. previous / next chain ---")
order = [g.stem for g in guides]
for i, g in enumerate(guides):
    t = g.read_text()
    prev = re.search(r"\*\*Previous\*\* \| \[[^\]]+\]\(([^)]+)\)", t)
    nxt = re.search(r"\*\*Next\*\* \| \[[^\]]+\]\(([^)]+)\)", t)
    exp_prev = order[i-1] + ".md" if i > 0 else None
    exp_next = order[i+1] + ".md" if i < len(order)-1 else None
    if exp_prev and (not prev or prev.group(1) != exp_prev):
        problems.append(f"{g.name}: Previous is {prev.group(1) if prev else 'MISSING'}, expected {exp_prev}")
    if exp_next and (not nxt or nxt.group(1) != exp_next):
        problems.append(f"{g.name}: Next is {nxt.group(1) if nxt else 'MISSING'}, expected {exp_next}")
print(f"  checked {len(guides)} guides")

# ---------- 3. internal anchors ----------
print("\n--- 3. internal anchors ---")
for g in guides:
    t = g.read_text()
    heads = {anchor(h) for h in re.findall(r"^#{1,4} (.+)$", t, re.M)}
    for link in re.findall(r"\]\(#([^)]+)\)", t):
        if link.lstrip("-") not in {h.lstrip("-") for h in heads}:
            problems.append(f"{g.name}: broken anchor #{link}")
print(f"  checked {len(guides)} guides")

# ---------- 4. cross-session anchors ----------
print("\n--- 4. cross-session anchors ---")
heads_by_file = {}
for g in guides:
    heads_by_file[g.name] = {anchor(h).lstrip("-") for h in re.findall(r"^#{1,4} (.+)$", g.read_text(), re.M)}
for g in guides:
    for target, frag in re.findall(r"\]\((session-[\w.-]+\.md)#([^)]+)\)", g.read_text()):
        if target not in heads_by_file:
            problems.append(f"{g.name}: link to unknown file {target}")
        elif frag.lstrip("-") not in heads_by_file[target]:
            problems.append(f"{g.name}: broken cross-link {target}#{frag}")
print("  done")

# ---------- 5. MCQ parity ----------
print("\n--- 5. MCQ question/answer parity ---")
for g in guides:
    t = g.read_text()
    qs = re.findall(r"^\*\*Q(\d+)\.\*\*", t, re.M)
    ans = re.findall(r"^\*\*A(\d+) — ", t, re.M)
    if not qs:
        notes.append(f"{g.name}: no MCQs")
        continue
    if len(qs) != len(ans):
        problems.append(f"{g.name}: {len(qs)} questions but {len(ans)} answers")
    blocks = t.count("## ❓") + t.count("# ❓")
    if blocks == 1:
        # one MCQ section -> numbering must be 1..n
        if sorted(map(int, qs)) != list(range(1, len(qs)+1)):
            problems.append(f"{g.name}: question numbering has gaps or duplicates")
    else:
        # per-topic MCQ blocks restart at Q1 by design
        notes.append(f"{g.name}: {blocks} per-topic MCQ blocks (numbering restarts - by design)")
    m = re.search(r"# ❓[^\n]*?(\d+) MCQs", t)
    if m and int(m.group(1)) != len(qs):
        problems.append(f"{g.name}: heading says {m.group(1)} MCQs, found {len(qs)}")
    print(f"  {g.stem:<44} {len(qs)} Q / {len(ans)} A")

# ---------- 6. images ----------
print("\n--- 6. referenced images ---")
missing_img = 0
for g in guides:
    for img in re.findall(r"!\[[^\]]*\]\((images/[^)]+)\)", g.read_text()):
        if not (S / "sessions" / img).exists():
            problems.append(f"{g.name}: missing image {img}"); missing_img += 1
print(f"  {missing_img} missing")

# ---------- 7. dataset URLs ----------
print("\n--- 7. dataset URLs resolve to files in the repo ---")
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/"
seen = set()
for f in list(guides) + list((S/"notebooks").glob("*.ipynb")):
    for url in re.findall(r"https://raw\.githubusercontent\.com/tech4alltraining/aiml/refs/heads/main/([^\"'\s)]+)", f.read_text()):
        seen.add(url.rstrip('\\n"'))
for rel in sorted(seen):
    if not (ROOT / rel).exists():
        problems.append(f"dataset URL has no local file: {rel}")
print(f"  {len(seen)} distinct dataset URLs, all checked")

# ---------- 8. index ----------
print("\n--- 8. sessions/README.md index ---")
idx = (S / "sessions" / "README.md").read_text()
for g in guides:
    if g.name not in idx:
        problems.append(f"sessions/README.md does not link {g.name}")
print("  done")

# ---------- 9. requirements ----------
print("\n--- 9. third-party imports vs requirements.txt ---")
req = (S / "requirements.txt").read_text().lower()
STD = {"os","sys","json","time","re","math","random","pathlib","itertools","collections",
       "warnings","datetime","typing","io","contextlib","string","functools","textwrap",
       "multiprocessing","subprocess","tempfile","shutil","csv","glob","copy","hashlib",
       "importlib","abc","dataclasses","enum","operator","statistics","zipfile","urllib"}
ALIAS = {"sklearn":"scikit-learn","google":"google-genai","dotenv":"python-dotenv",
         "PIL":"pillow","imblearn":"imbalanced-learn","cv2":"opencv-python",
         "yaml":"pyyaml","bs4":"beautifulsoup4"}
found = set()
for g in guides:
    for block in re.findall(r"```python\n(.*?)```", g.read_text(), re.S):
        for m in re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", block, re.M):
            if m not in STD and not m.startswith("_"):
                found.add(m)
LOCAL = {"genai_helper", "train", "app", "stdin_shim", "cache_shim"}
# deliberately not course dependencies - every use is marked `# needs-install:`
OPTIONAL = {"optuna", "hyperopt", "torch", "transformers", "datasets", "gradio", "mlxtend", "umap"}
for mod in sorted(found - LOCAL - OPTIONAL):
    pkg = ALIAS.get(mod, mod)
    if pkg.lower() not in req:
        problems.append(f"import '{mod}' -> package '{pkg}' not in requirements.txt")
print(f"  {len(found - LOCAL - OPTIONAL)} required + {len(found & OPTIONAL)} optional (marked needs-install)")

# ---------- report ----------
print("\n" + "=" * 62)
if notes:
    print("NOTES")
    for n in notes: print("  -", n)
if problems:
    print(f"\n{len(problems)} PROBLEM(S)")
    for p in problems: print("  ✗", p)
    sys.exit(1)
print("\nALL CONSISTENCY CHECKS PASSED")
