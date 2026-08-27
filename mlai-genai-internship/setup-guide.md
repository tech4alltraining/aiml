# Environment Setup Guide

**ML/AI & GenAI Internship Program**

Complete installation instructions for **Windows, Ubuntu/Linux and macOS**, using either **`venv`** or **`conda`**.

> **Do this before Module 1.** Nothing in the course works until it is done, and a room of twenty students installing PyTorch simultaneously does not work.

| Related | For |
|---|---|
| [Troubleshooting Guide](troubleshooting.md) | When something goes wrong |
| [Student Handbook](student-handbook.md) | The course itself |
| [Notebooks](notebooks) | Ready-to-run Colab/Jupyter notebooks |
| [requirements.txt](requirements.txt) | The package list |

---

## Contents

1. [Which route should you take?](#which-route-should-you-take)
2. [Step 0: Check what you already have](#step-0-check-what-you-already-have)
3. [Route A: `venv`](#route-a-venv-built-into-python) — [Windows](#windows) · [Ubuntu](#ubuntu--linux) · [macOS](#macos)
4. [Route B: `conda`](#route-b-conda-miniconda) — [Windows](#windows-1) · [Ubuntu](#ubuntu--linux-1) · [macOS](#macos-1)
5. [Step 3: Install the packages](#step-3-install-the-packages)
6. [Step 4: Verify the installation](#step-4-verify-the-installation)
7. [Step 5: Register the environment with Jupyter](#step-5-register-the-environment-with-jupyter)
8. [Step 6: Set up VS Code](#step-6-set-up-vs-code)
9. [Google Colab](#google-colab-no-installation-at-all)
10. [Everyday commands](#everyday-commands)
11. [Storing your Gemini API key safely](#storing-your-gemini-api-key-safely)
12. [Setup checklist](#setup-checklist)

---

# What an environment is, and why you need one

**Every command in this course runs inside a Python environment named `genai`.** An environment is a private, isolated copy of Python and its packages, kept in its own folder.

> 🧠 **Analogy: separate toolboxes.** Imagine one shared toolbox for every job in the workshop. Someone swaps the 10mm spanner for a slightly different one, and now three other people's jobs break. A Python environment is your own toolbox for this course. Install what you like in it; nothing you do can break the system Python your operating system depends on, and nothing anyone else does can break yours.


**Every command in this handbook runs inside a Python environment named `genai`.** An environment is a private, isolated copy of Python and its packages, kept in its own folder.

> 🧠 **Analogy: separate toolboxes.** Imagine one shared toolbox for every job in the workshop. Someone swaps the 10mm spanner for a slightly different one, and now three other people's jobs break. A Python environment is your own toolbox for this course. Install what you like in it; nothing you do can break the system Python your operating system depends on, and nothing anyone else does can break yours.

## Which route should you take?

There are two standard ways to make an environment. **Both work for the entire course.** Pick one and stay on it.

| | **Route A — `venv`** | **Route B — `conda`** |
|---|---|---|
| Comes with | Python itself — nothing extra to install | Miniconda (a ~90 MB download) |
| Download size | Small | Larger |
| Best if | You already have Python 3.10+ installed | You are on Windows, or want a data-science setup that "just works" |
| Manages | Python packages only | Python packages **and** the Python version itself |
| Command style | `python -m venv genai` | `conda create -n genai` |
| Used by | Most Python developers | Most data scientists |

**Not sure? Use Route B (conda).** It handles the numerical libraries more reliably, especially on Windows.

Whichever route you choose, jump to your operating system below, then continue to [Step 3](#step-3-install-the-packages) — that step onwards is identical for everybody.

---

## Step 0: Check what you already have

Open a terminal and run **both** of these. Nothing is broken if one fails — you are just finding out where you stand.

**Windows** — open **PowerShell** (press `Win`, type `powershell`, press Enter):

```powershell
python --version
```

```powershell
conda --version
```

**Ubuntu / Linux** — open a terminal (`Ctrl+Alt+T`):

```bash
python3 --version
```

```bash
conda --version
```

**macOS** — open **Terminal** (press `Cmd+Space`, type `terminal`, press Enter):

```bash
python3 --version
```

```bash
conda --version
```

| What you saw | What it means |
|---|---|
| `Python 3.10` or higher | ✅ Route A (venv) will work |
| `Python 3.9` or lower | Install a newer Python, or use Route B |
| `conda 24.x` or similar | ✅ Route B will work |
| `command not found` / `not recognized` | That tool is not installed yet — see below |

> ⚠️ **Windows users:** if `python --version` opens the Microsoft Store, Python is not properly installed. Follow the Windows instructions below and **tick "Add python.exe to PATH"** during installation.

---

# Route A: `venv` (built into Python)

## Windows

### A1. Install Python

1. Go to [python.org/downloads](https://www.python.org/downloads/) and download Python 3.12.
2. Run the installer.
3. **On the first screen, tick "Add python.exe to PATH".** This is the single most important click. If you miss it, every command below will fail with `'python' is not recognized`.
4. Click **Install Now**.
5. **Close PowerShell and open a new one** — PATH changes only apply to new terminals.

Verify:

```powershell
python --version
```

### A2. Create a project folder and the environment

```powershell
mkdir C:\Users\%USERNAME%\mlai-genai
```

```powershell
cd C:\Users\%USERNAME%\mlai-genai
```

```powershell
python -m venv genai
```

This creates a `genai` folder containing your private Python.

### A3. Activate it

In **PowerShell**:

```powershell
.\genai\Scripts\Activate.ps1
```

In **Command Prompt (cmd.exe)**:

```bat
genai\Scripts\activate.bat
```

Your prompt should now start with `(genai)`.

> ⚠️ **PowerShell error: "running scripts is disabled on this system"?**
> Windows blocks scripts by default. Run this once, answer `Y`, then activate again:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
>
> This only affects your own user account and is the standard fix.

Now go to [Step 3](#step-3-install-the-packages).

---

## Ubuntu / Linux

### A1. Install Python and the venv module

Ubuntu ships Python but **not** the `venv` module — you must install it separately, which catches almost everyone out.

```bash
sudo apt update
```

```bash
sudo apt install -y python3 python3-pip python3-venv build-essential
```

Verify:

```bash
python3 --version
```

### A2. Create a project folder and the environment

```bash
mkdir -p ~/mlai-genai && cd ~/mlai-genai
```

```bash
python3 -m venv genai
```

### A3. Activate it

```bash
source genai/bin/activate
```

Your prompt should now start with `(genai)`.

> ⚠️ **`Error: Command '...' returned non-zero exit status 1`** when creating the venv usually means `python3-venv` is missing. Run the `apt install` line above.

> ⚠️ **`error: externally-managed-environment`** when you `pip install` means you are **outside** the environment and pip is protecting your system Python. Run `source genai/bin/activate` first. Never use `--break-system-packages` to get around this.

Now go to [Step 3](#step-3-install-the-packages).

---

## macOS

### A1. Install Python

macOS includes an old Python. Install a current one:

```bash
brew install python@3.12
```

No Homebrew? Install it first:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Or download the installer from [python.org/downloads](https://www.python.org/downloads/).

### A2. Create a project folder and the environment

```bash
mkdir -p ~/mlai-genai && cd ~/mlai-genai
```

```bash
python3 -m venv genai
```

### A3. Activate it

```bash
source genai/bin/activate
```

Now go to [Step 3](#step-3-install-the-packages).

---

# Route B: `conda` (Miniconda)

## Windows

### B1. Install Miniconda

1. Download the **Windows 64-bit** installer from [docs.conda.io/projects/miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
2. Run it and accept the defaults.
3. When finished, open **Anaconda Prompt** from the Start menu — **not** PowerShell.

> **Use Anaconda Prompt for everything on Windows.** `conda` is not on PowerShell's PATH by default, and the number one Windows problem in this course is students typing conda commands into the wrong terminal.

Verify:

```bat
conda --version
```

### B2. Create the environment

```bat
conda create -n genai python=3.12 -y
```

### B3. Activate it

```bat
conda activate genai
```

Your prompt should now start with `(genai)`.

Now go to [Step 3](#step-3-install-the-packages).

---

## Ubuntu / Linux

### B1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
```

```bash
bash ~/miniconda.sh -b -p $HOME/miniconda3
```

```bash
$HOME/miniconda3/bin/conda init bash
```

**Close the terminal and open a new one**, then verify:

```bash
conda --version
```

> Using zsh instead of bash? Run `$HOME/miniconda3/bin/conda init zsh`.

### B2. Create the environment

```bash
conda create -n genai python=3.12 -y
```

### B3. Activate it

```bash
conda activate genai
```

Now go to [Step 3](#step-3-install-the-packages).

---

## macOS

### B1. Install Miniconda

Download the installer from [docs.conda.io/projects/miniconda](https://docs.conda.io/projects/miniconda/en/latest/).

### Mac users on Apple Silicon: check this before you go further

> ⚠️ **Choose the right build.** On an M1/M2/M3/M4 Mac take the **Apple Silicon (arm64)** installer, not the Intel one. Check your hardware with `uname -m` — `arm64` means Apple Silicon. A Rosetta (x86_64) conda installs fine and runs almost everything, but caps PyTorch at 2.2.2, which breaks the Module 5 Hugging Face examples. Details in the [Troubleshooting Guide](troubleshooting.md#environment-problems).

Or via Homebrew:

```bash
brew install --cask miniconda
```

```bash
conda init zsh
```

Close the terminal and open a new one.

### B2. Create the environment

```bash
conda create -n genai python=3.12 -y
```

### B3. Activate it

```bash
conda activate genai
```

---

# Step 3: Install the packages

**From here on, every instruction is identical on Windows, Ubuntu and macOS, and identical for venv and conda.**

> ⚠️ **Before you type anything: does your prompt show `(genai)`?**
>
> ```text
> (genai) C:\Users\you\mlai-genai>     ← Windows, correct
> (genai) you@ubuntu:~/mlai-genai$     ← Ubuntu, correct
> you@ubuntu:~/mlai-genai$             ← WRONG - activate first
> ```
>
> If `(genai)` is missing, the packages install somewhere else and nothing will work. This is the most common problem in the whole course.

## 3.1 Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 3.2 Install the Module 1–3 packages (data and machine learning)

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

| Package | What it does | First used |
|---|---|---|
| `numpy` | Fast numerical arrays | Module 1 |
| `pandas` | Tables of data (DataFrames) | Module 1 |
| `matplotlib` | Plotting | Module 2 |
| `seaborn` | Prettier statistical plots | Module 2 |
| `scikit-learn` | Machine learning models and metrics | Module 2 |
| `joblib` | Saving and loading trained models | Module 5 |

## 3.3 Install the notebook tools

```bash
pip install jupyter ipykernel notebook
```

## 3.4 Install the Module 4–5 packages (GenAI and web apps)

```bash
pip install google-genai streamlit pillow python-dotenv
```

| Package | What it does | First used |
|---|---|---|
| `google-genai` | Calls the Gemini API | Module 4 |
| `streamlit` | Turns a Python script into a web app | Module 4 |
| `pillow` | Image handling, for image-input apps | Module 4 |
| `python-dotenv` | Reads your API key from a `.env` file | Module 4 |

## 3.5 Install the Module 5 Hugging Face packages

> ⚠️ **This is a 2–3 GB download. Do it the evening before Module 5, not during the session.** Twenty students downloading PyTorch at once on classroom wifi does not work.

```bash
pip install transformers torch datasets evaluate gradio
```

**Ubuntu/Windows with no GPU?** Install the smaller CPU-only build instead — about 800 MB rather than 2.5 GB:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install transformers datasets evaluate gradio
```

## 3.6 Or install everything at once

Create a file named **`requirements.txt`** in your project folder:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
jupyter
ipykernel
notebook
google-genai
streamlit
pillow
python-dotenv
```

Then:

```bash
pip install -r requirements.txt
```

A ready-made copy is in this repository: [`requirements.txt`](requirements.txt).

## 3.7 See what you installed

```bash
pip list
```

Save your exact versions, so a classmate or a marker can reproduce your results:

```bash
pip freeze > requirements-lock.txt
```

---

# Step 4: Verify the installation

Do not skip this. Two minutes here saves an hour on Module 1.

Create **`check_setup.py`** in your project folder:

```python
"""Run this once after setup. Every line should print OK."""

import sys
import platform

print("Python  :", sys.version.split()[0])
print("Platform:", platform.system(), platform.machine())
print("Running :", sys.executable)
print("-" * 55)

packages = [
    ("numpy", "Module 1 - arrays"),
    ("pandas", "Module 1 - dataframes"),
    ("matplotlib", "Module 2 - plotting"),
    ("seaborn", "Module 2 - statistical plots"),
    ("sklearn", "Module 2 - machine learning"),
    ("joblib", "Module 5 - saving models"),
    ("streamlit", "Module 4 - web apps"),
    ("PIL", "Module 4 - images"),
    ("dotenv", "Module 4 - API keys"),
]

missing = []
for name, purpose in packages:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "installed")
        print(f"OK    {name:<12} {version:<10} {purpose}")
    except ImportError:
        print(f"FAIL  {name:<12} {'-':<10} {purpose}")
        missing.append(name)

try:
    from google import genai
    print(f"OK    {'google-genai':<12} {'installed':<10} Module 4 - Gemini API")
except ImportError:
    print(f"FAIL  {'google-genai':<12} {'-':<10} Module 4 - Gemini API")
    missing.append("google-genai")

print("-" * 55)
if missing:
    print("MISSING:", ", ".join(missing))
    print("Fix: activate the environment, then re-run the pip install commands.")
else:
    print("All set. You are ready for Module 1.")
```

Run it:

```bash
python check_setup.py
```

Expected output (version numbers will differ):

```text
Python  : 3.12.11
Platform: Linux x86_64
Running : /home/you/mlai-genai/genai/bin/python
-------------------------------------------------------
OK    numpy        2.4.0      Module 1 - arrays
OK    pandas       2.3.3      Module 1 - dataframes
OK    matplotlib   3.10.8     Module 2 - plotting
OK    seaborn      0.13.2     Module 2 - statistical plots
OK    sklearn      1.9.0      Module 2 - machine learning
OK    joblib       1.5.3      Module 5 - saving models
OK    streamlit    1.59.2     Module 4 - web apps
OK    PIL          12.0.0     Module 4 - images
OK    dotenv       installed  Module 4 - API keys
OK    google-genai installed  Module 4 - Gemini API
-------------------------------------------------------
All set. You are ready for Module 1.
```

**Look at the `Running :` line.** The path must contain `genai`. If it does not, you are running the wrong Python — your environment is not active.

---

# Step 5: Register the environment with Jupyter

So `genai` appears as a kernel in Jupyter and in VS Code notebooks:

```bash
python -m ipykernel install --user --name genai --display-name "Python (genai)"
```

Start Jupyter:

```bash
jupyter notebook
```

Inside a notebook choose **Kernel → Change kernel → Python (genai)**.

> **A notebook that cannot find pandas** is nearly always a notebook running on the wrong kernel. Check the kernel name in the top-right corner before you debug anything else.

---

# Step 6: Set up VS Code

### 6.1 Install VS Code

| OS | How |
|---|---|
| **Windows** | Download from [code.visualstudio.com](https://code.visualstudio.com) |
| **Ubuntu** | `sudo snap install --classic code` — or download the `.deb` |
| **macOS** | Download from [code.visualstudio.com](https://code.visualstudio.com), or `brew install --cask visual-studio-code` |

### 6.2 Install the extensions

Open the Extensions panel (`Ctrl+Shift+X`, or `Cmd+Shift+X` on macOS) and install:

- **Python** (Microsoft) — required
- **Jupyter** (Microsoft) — for notebooks inside VS Code
- **Pylance** (Microsoft) — autocomplete and error highlighting

### 6.3 Open your project folder

```bash
code .
```

If `code` is not recognised, open VS Code manually and use **File → Open Folder**.

### 6.4 Select the interpreter — the step everyone forgets

1. Press `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS).
2. Type **Python: Select Interpreter**.
3. Choose the one whose path contains **`genai`**.

The bottom-right of the VS Code window should now show your environment name. Every terminal you open inside VS Code will activate it automatically.

> **"But it works in my terminal and not in VS Code!"** — you have selected a different interpreter. Repeat 6.4.

---

# Google Colab: no installation at all

Colab runs Python in your browser on Google's machines. Nothing to install, works on any laptop, and it is free.

**Use Colab when:** your laptop is slow, the install fails, or you just want to experiment quickly.
**Use a local environment when:** you are building Streamlit apps, or working on your capstone.

## Opening a course notebook

Every demo in this course is a ready-to-run notebook in [`notebooks/`](notebooks). Click the **Open in Colab** badge at the top of any of them.

## Installing packages in Colab

```python
# The ! runs a shell command from inside a notebook cell.
# -q keeps the output quiet.
!pip install -q -U google-genai
```

Colab already includes: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`.
You must install: `google-genai`, and `streamlit` if you want to inspect app code.

> ⚠️ **Installs do not persist.** When the runtime disconnects, Colab resets. Put every `!pip install` in the **first cell** so re-running is one click.

## Your API key in Colab — two safe ways

**Method 1: `getpass`** (works everywhere, ask each session)

```python
import os
from getpass import getpass

# getpass hides your typing, so the key is never saved in the notebook.
os.environ["GEMINI_API_KEY"] = getpass("Enter your Gemini API key: ")
```

**Method 2: Colab Secrets** (better — store it once)

1. Click the 🔑 key icon in the left sidebar.
2. **Add new secret**, name it `GEMINI_API_KEY`, paste the value.
3. Toggle **Notebook access** on.

```python
from google.colab import userdata

api_key = userdata.get("GEMINI_API_KEY")
```

> ⚠️ **Never type your key directly into a cell.** It is saved inside the `.ipynb` file, and it travels with the notebook when you share it or commit it to GitHub.

## Getting data into Colab

**Easiest — read straight from a URL:**

```python
import pandas as pd

BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "classification/iris.csv")
df.head()
```

**Upload from your computer:**

```python
from google.colab import files

uploaded = files.upload()   # opens a file picker
```

**Mount Google Drive** (the only way to keep files between sessions):

```python
from google.colab import drive

drive.mount('/content/drive')
# Your files are then under /content/drive/MyDrive/
```

## Colab limits worth knowing

| Limit | What it means |
|---|---|
| Idle disconnect | Leave it alone too long and the runtime stops. Variables are lost. |
| Session length | Free sessions are capped at a few hours. |
| Storage is temporary | Files vanish when the runtime resets. Save to Drive. |
| GPU is rationed | Free GPU access is limited and not guaranteed. |
| No Streamlit | Apps need a local server. Run those on your own machine. |

## Saving your work

- **File → Save a copy in Drive** keeps your own version.
- **File → Download → .ipynb** gives you the file to commit to GitHub.
- **Runtime → Restart and run all** before you submit anything — it proves the notebook works top to bottom.

---

# Everyday commands

You will type these many times. Learn them now.

| Task | venv | conda |
|---|---|---|
| **Activate** (Windows) | `.\genai\Scripts\Activate.ps1` | `conda activate genai` |
| **Activate** (Ubuntu/macOS) | `source genai/bin/activate` | `conda activate genai` |
| **Deactivate** | `deactivate` | `conda deactivate` |
| **List environments** | *(look for the folders)* | `conda env list` |
| **What is installed** | `pip list` | `pip list` |
| **Add a package** | `pip install <name>` | `pip install <name>` |
| **Remove a package** | `pip uninstall <name>` | `pip uninstall <name>` |
| **Delete and start over** | delete the `genai` folder | `conda env remove -n genai` |

And these, regardless of route:

| Task | Command |
|---|---|
| Run a Python file | `python myfile.py` |
| Run a Streamlit app | `streamlit run myapp.py` |
| Start Jupyter | `jupyter notebook` |
| Stop a running server | `Ctrl+C` in the terminal |
| Which Python am I using? | `python -c "import sys; print(sys.executable)"` |

> **You must activate the environment in every new terminal.** Activation is not permanent. Closing the terminal ends it.

---

# Storing your Gemini API key safely

**Never put your API key in a `.py` file that you commit to GitHub.** Keys pushed to public repositories are found by automated scanners and abused within minutes.

## Get a key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Sign in with your Google account.
3. Click **Create API key**.
4. Copy it. It looks like `AIzaSy...` and is about 39 characters.

## Method 1: a `.env` file (for plain Python scripts)

Create a file named exactly **`.env`** in your project folder:

```text
GEMINI_API_KEY=AIzaSy_your_actual_key_here
```

No quotes, no spaces around the `=`.

Read it in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GEMINI_API_KEY"]
```

**Creating a dot-file per OS:**

| OS | How |
|---|---|
| **Windows** | In VS Code: **File → New File**, save as `.env`. (Notepad appends `.txt` — use VS Code.) |
| **Ubuntu/macOS** | `touch .env` then `code .env`, or `nano .env` |

> Files starting with a dot are hidden. In VS Code they show normally. In a file manager press `Ctrl+H` (Ubuntu) or `Cmd+Shift+.` (macOS) to reveal them.

## Method 2: `.streamlit/secrets.toml` (for Streamlit apps)

```bash
mkdir .streamlit
```

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "AIzaSy_your_actual_key_here"
```

Note the **quotes** here — TOML requires them, `.env` does not.

Read it in Python:

```python
import streamlit as st

api_key = st.secrets["GEMINI_API_KEY"]
```

## Always create a `.gitignore` — before your first commit

Create a file named **`.gitignore`** containing:

```text
# Secrets - never commit these
.env
.streamlit/secrets.toml

# Environments
genai/
venv/
.venv/

# Python
__pycache__/
*.pyc

# Trained models (often too large for Git)
*.joblib
*.pkl

# Notebooks
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

Check it is working:

```bash
git status
```

`.env` must **not** appear in the list. If it does, `.gitignore` is in the wrong folder or misspelled.

> **If you ever push a key by accident**, revoke it immediately at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a new one. Deleting the commit is **not** enough — the key is already in the Git history and in anyone's clone.

---

# Setup checklist

Tick every box before Module 1:

- Python 3.10+ or Miniconda installed
- Project folder created
- `genai` environment created
- Environment activates, and the prompt shows `(genai)`
- Module 1–3 packages installed
- Notebook tools installed
- Module 4–5 packages installed
- `python check_setup.py` prints **All set**
- Jupyter kernel registered
- VS Code installed with the Python extension
- VS Code interpreter set to `genai`
- Gemini API key created
- `.env` file created with the key
- `.gitignore` created, and `git status` does not show `.env`
- (Before Module 5) Hugging Face packages installed

---
