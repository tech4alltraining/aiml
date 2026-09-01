import re, pathlib, sys, json
ROOT = pathlib.Path("/Users/ahk/Workspace/aiml/mlai-genai-internship")
bad = []
def scan(src, text, label):
    for m in re.finditer(r'\[([^\]]*)\]\(([^)\s]+)\)', text):
        t = m.group(2)
        if t.startswith(("http", "#", "mailto:")): continue
        p = t.partition("#")[0]
        if p and not (src.parent / p).resolve().exists():
            bad.append(f"{label} -> {t}")

n_md = n_nb = 0
for f in ROOT.rglob("*.md"):
    if "/latex/" in str(f) or "/archive/" in str(f): continue
    n_md += 1; scan(f, f.read_text(), str(f.relative_to(ROOT)))
for f in ROOT.rglob("*.ipynb"):
    if "/.ipynb_checkpoints/" in str(f): continue
    n_nb += 1
    nb = json.loads(f.read_text())
    for c in nb.get("cells", []):
        if c.get("cell_type") == "markdown":
            scan(f, "".join(c.get("source", [])), str(f.relative_to(ROOT)))

print(f"scanned {n_md} markdown files, {n_nb} notebooks")
print(f"broken links: {len(bad)}")
for b in sorted(set(bad)): print("  ", b)
