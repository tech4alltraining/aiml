# Troubleshooting Guide

**ML/AI & GenAI Internship Program**

Every error you are likely to hit, and how to fix it. Find your error message below.

> **How to read an error.** Beginners see a red wall of text and panic. Experienced programmers read the **last line first** — that line names the actual problem. Everything above it is just the path the computer took to get there.

| Related | For |
|---|---|
| [Setup Guide](setup-guide.md) | Installing everything in the first place |
| [Student Handbook](student-handbook.md) | The course itself |

---

## The four questions to ask before anything else

Nine times in ten, one of these is the answer:

1. **Does my prompt show `(genai)`?** If not, activate the environment.
2. **Am I in the right folder?** Run `pwd` (Ubuntu/macOS) or `cd` (Windows) and check.
3. **Did I run the training script first?** Apps that load a `.joblib` need it to exist.
4. **Is VS Code using the right interpreter?** `Ctrl+Shift+P` → **Python: Select Interpreter**.

---

## Contents

1. [Environment problems](#environment-problems)
2. [API problems](#api-problems)
3. [Modelling problems](#modelling-problems)
4. [Streamlit problems](#streamlit-problems)
5. [Notebook problems](#notebook-problems)
6. [Still stuck?](#still-stuck)

---

## Environment problems

**`conda: command not found`**
Conda is not on your PATH. On Windows, use the **Anaconda Prompt** rather than PowerShell. On macOS/Linux, close and reopen the terminal, or run `source ~/miniconda3/etc/profile.d/conda.sh`.

**`ModuleNotFoundError: No module named 'pandas'`**
You installed into a different environment than the one you are running from. Fix:

```bash
conda activate genai
python -c "import sys; print(sys.executable)"
pip install pandas
```

The printed path must contain `envs/genai`. If it does not, your terminal is not in the environment.

**VS Code cannot find the package but the terminal can**
VS Code is using a different interpreter. `Cmd/Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one under `envs/genai`.

**`pip install` is very slow or fails on PyTorch**
PyTorch is a large download. Use a stable connection and let it finish; do not interrupt it. If it fails halfway, re-run the same command — pip resumes from its cache.

**`RuntimeError: Numpy is not available` when running a Hugging Face pipeline**
Your PyTorch was built against NumPy 1.x but your environment has NumPy 2.x. Check which PyTorch you actually got:

```bash
conda activate genai
python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"
```

If torch is **2.2.2 or older on macOS**, you are running an Intel (x86_64) environment on an Apple Silicon Mac — PyTorch stopped publishing Intel macOS wheels after 2.2.2. See [the Apple Silicon check](setup-guide.md#mac-users-on-apple-silicon-check-this-before-you-go-further) in the Setup Guide; the real fix is to reinstall the arm64 Miniconda and recreate the environment.

If you cannot reinstall right now, the temporary workaround is to pin NumPy down to match your PyTorch:

```bash
conda activate genai
pip install "numpy<2"
```

This gets Hugging Face working, but it downgrades NumPy for everything else in the environment. Prefer fixing the architecture.

**`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`**
Same cause as above. It is a warning at import time and an error the moment the model actually runs.

## API problems

**`400 API key not valid`**
The key is wrong, has extra spaces or quotes, or was revoked. Copy it fresh from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

**`KeyError: 'GEMINI_API_KEY'`**
Your `.env` file is missing, is in a different folder than the script, or you forgot `load_dotenv()`. The `.env` must be in the folder you run the command **from**.

**`429 RESOURCE_EXHAUSTED`**
You have exceeded the free-tier rate limit. Wait a minute. If you are calling the API inside a loop, add a pause between calls or reduce the number of iterations.

**`st.secrets` raises `StreamlitSecretNotFoundError`**
The file must be at `.streamlit/secrets.toml` relative to where you run `streamlit run`, and the folder name starts with a dot.

**The response is empty or cut off mid-sentence**
`max_output_tokens` is too low. Raise it. Also check `response.candidates[0].finish_reason` — if it says `SAFETY`, your prompt triggered a safety filter.

## Modelling problems

**`ValueError: could not convert string to float`**
A text column reached the model. Encode every categorical column before `fit()`.

**`ValueError: X has 12 features, but ... is expecting 13 features`**
Your prediction input has different columns, or a different **order**, than your training data. Always reindex with the saved feature list:

```python
input_df = input_df[FEATURE_COLUMNS]
```

**Accuracy is 1.0 — suspiciously perfect**
Almost always leakage. Check that the target column, or a column derived from it, is not sitting in `X`.

**Accuracy is around 0.5 on a two-class problem**
The model has learned nothing. Check that `y` is what you think it is, and that your features are not all constant.

**`ConvergenceWarning` from LogisticRegression**
Increase `max_iter` (2000 usually suffices) or scale your features first.

## Streamlit problems

**`streamlit: command not found`**
Not installed in this environment: `conda activate genai` then `pip install streamlit`.

**Port 8501 already in use**
Another Streamlit app is still running. Find and stop it, or use `streamlit run app.py --server.port 8502`.

**The app reloads and loses everything on every click**
That is Streamlit's normal re-run behaviour. Store what must survive in `st.session_state`, and wrap model loading in `@st.cache_resource`.

**`FileNotFoundError: rf_model.joblib`**
The model file is not in the folder you ran `streamlit run` from. Either move it there, or build an absolute path:

```python
from pathlib import Path
MODEL_PATH = Path(__file__).parent / "rf_model.joblib"
```

---

---

# Notebook problems

**The notebook cannot find pandas, but my terminal can**
The notebook is running on a different kernel. Check the kernel name in the top-right corner. Choose **Kernel → Change kernel → Python (genai)**. If `Python (genai)` is not listed, register it:

```bash
conda activate genai
python -m ipykernel install --user --name genai --display-name "Python (genai)"
```

**`NameError: name 'df' is not defined` — but I defined it above**
You ran the cells out of order, or restarted the kernel. Cell numbers in the margin (`[1]`, `[2]`, …) show the order they actually ran in. Use **Run → Restart Kernel and Run All Cells** to prove your notebook works top to bottom.

> **Always restart and run all before submitting a notebook.** A notebook that only works in the order *you happened* to click is not reproducible, and markers will run it top to bottom.

**Colab: `ModuleNotFoundError` for a package I installed**
Colab resets when the runtime disconnects. Re-run the `!pip install` cell. Put all installs in the first cell so this is one click.

**Colab: my file disappeared**
Colab storage is temporary. Save work to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Colab is very slow, or disconnects**
Free Colab has usage limits and disconnects idle sessions. For long training runs, use your local environment instead.

**Plots do not appear in a `.py` file**
`plt.show()` opens a window that some terminals cannot display. Save to a file instead:

```python
plt.savefig("chart.png", dpi=120)
```

In notebooks, plots appear automatically — you do not need `plt.show()`.

---

# Still stuck?

Work through this in order:

1. **Read the last line of the error.** Search that exact line — with your own file paths removed.
2. **Check the four questions** at the top of this page.
3. **Restart.** Kernel restart for notebooks, `Ctrl+C` and re-run for Streamlit, new terminal for environment weirdness.
4. **Reproduce it small.** Make the smallest file that still shows the error. Half the time you find the cause doing this.
5. **Ask, with detail.** "It doesn't work" cannot be answered. Include: what you ran, the full error, your OS, and the output of:

```bash
python -c "import sys; print(sys.executable)"
```

> Twenty minutes stuck is learning. Two hours stuck is waste. Ask.
