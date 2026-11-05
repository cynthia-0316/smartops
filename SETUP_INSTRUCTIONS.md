# SmartOps – Local Setup Instructions (for an AI coding assistant)

> **Audience:** an AI assistant (Claude Code, Cursor, Copilot Chat, ChatGPT, etc.) helping a **teammate who may be a complete beginner** set up this project on their own computer.
> **Human teammate:** you can just tell your AI: *"Follow SETUP_INSTRUCTIONS.md in this repo and set everything up for me."*

## Rules for the AI assistant (read first)

1. **Go one step at a time.** Run the step, check the result, tell the human in plain English what happened, then continue. Never skip a step, even if it looks like it is already done – run its **Check** to prove it.
2. **Assume no prior knowledge.** Explain terms briefly the first time (terminal, repo, clone, virtual environment, branch). Don't use jargon without saying what it is.
3. **Ask before installing software** and say what it is and why it is needed.
4. **Never type, store, guess or ask for the human's passwords or tokens.** When a step needs a login (GitHub), tell the human to do it themselves in the browser window that opens.
5. **Never commit or push to `main`.** Never run `git push --force`, `git reset --hard`, or delete anything you did not create in this setup.
6. If a command fails, read the error, check **Troubleshooting** at the bottom, fix it, and re-run the **Check**. If still stuck after two attempts, stop and tell the human exactly what failed and what to paste to the team chat.
7. **Do not modify project files** (notebooks, README, requirements, .gitignore) during setup.
8. At the end, run `python scripts/verify_setup.py` and report the result.

## Project facts

| Item | Value |
|---|---|
| Repo | https://github.com/cynthia-0316/smartops (**private** – needs an invitation) |
| Owner | `cynthia-0316` |
| Teammates with access | `xwang660-dev`, `Noah-DV`, `mkhan352` |
| Language | Python 3.10 or newer (3.12 or 3.13 are safest) |
| Packages | pandas, numpy, scikit-learn, matplotlib, jupyter (`requirements.txt`) |
| Dataset | https://huggingface.co/datasets/jskswamy/predictive-maintenance-data (MIT license, ~8 MB, public – no login needed) |
| Course | CIS 508 Machine Learning (ASU) |

Suggested folder for the project (any folder works, avoid folders synced by OneDrive/iCloud if possible):
- **Mac/Linux:** `~/Documents/ASU/CIS 508 Machine Learning/project`
- **Windows:** `C:\Users\<you>\Documents\ASU\CIS 508 Machine Learning\project`

---

## Step 0 – Detect the operating system

Determine whether the human is on **macOS**, **Windows**, or **Linux**, and use the matching commands below. On Windows, use **PowerShell** (not the old Command Prompt). Commands marked `(all)` are the same everywhere.

**Check:** state the OS and shell you detected.

## Step 1 – Accept the GitHub invitation (human does this)

The repo is private. The human must have accepted the invite sent by `cynthia-0316`.

1. Human opens https://github.com/cynthia-0316/smartops while logged in to GitHub.
2. If the page says "404", they check their email (and https://github.com/notifications) for an invitation from `cynthia-0316` and click **Accept invitation**.
3. If they have no GitHub account: create one at https://github.com/signup (human does this, AI must not), then message Cynthia the username so she can add them.

**Check:** the human can see the repo's file list in the browser.

## Step 2 – Install Git

**Check first:** `git --version` (all). If it prints a version, skip to Step 3.

- **macOS:** run `xcode-select --install` (a pop-up appears; click Install, wait several minutes). Or `brew install git` if Homebrew exists.
- **Windows:** `winget install --id Git.Git -e --source winget`. If `winget` is missing, human downloads from https://git-scm.com/download/win and clicks through the installer with default options.
- **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install -y git`

Then **close and reopen the terminal** so the new command is found.

**Check:** `git --version` prints e.g. `git version 2.x.x`.

## Step 3 – Tell Git who you are

Ask the human for the **name** and **email** they want on their commits (should match their GitHub account; it will be public in the commit history). Then:

```bash
git config --global user.name "THEIR NAME"
git config --global user.email "THEIR EMAIL"
```

**Check:** `git config --global user.name` and `git config --global user.email` print the values.

## Step 4 – Install Python

**Check first:** run `python3 --version` (Mac/Linux) or `python --version` (Windows). Need **3.10 or newer**. If it is older, missing, or on Windows opens the Microsoft Store, install:

- **macOS:** `brew install python@3.12` if Homebrew exists; otherwise human downloads the installer from https://www.python.org/downloads/ and runs it.
- **Windows:** `winget install --id Python.Python.3.12 -e`. Or download from https://www.python.org/downloads/ and **tick "Add python.exe to PATH"** on the first installer screen (very important).
- **Linux:** `sudo apt install -y python3 python3-venv python3-pip`

Close and reopen the terminal.

**Check:** `python3 --version` (Mac/Linux) / `python --version` (Windows) prints 3.10+.
(Below, `python` means whichever of these works. On Mac/Linux use `python3` if `python` is not found.)

## Step 5 – Install the GitHub CLI and log in

The GitHub CLI (`gh`) lets the terminal talk to the private repo using the human's GitHub login, so no password is ever typed into the terminal.

**Check first:** `gh --version`. If installed, skip the install.

- **macOS:** `brew install gh` (if no Homebrew, install it from https://brew.sh first, or use the installer at https://cli.github.com).
- **Windows:** `winget install --id GitHub.cli -e`
- **Linux:** follow https://github.com/cli/cli/blob/trunk/docs/install_linux.md

Close and reopen the terminal, then:

```bash
gh auth login
```

Tell the human to choose: **GitHub.com** → **HTTPS** → **Yes** (authenticate Git with GitHub credentials) → **Login with a web browser**. They copy the one-time code shown, press Enter, and approve in the browser. **The human does the browser part; the AI must not handle the code or token.**

**Check:** `gh auth status` says "Logged in to github.com account <their-username>". Then `gh repo view cynthia-0316/smartops --json nameWithOwner,viewerPermission` shows `WRITE`. If it errors with "not found", go back to Step 1.

## Step 6 – Download (clone) the project

```bash
mkdir -p "$HOME/Documents/ASU/CIS 508 Machine Learning/project"     # Mac/Linux
cd "$HOME/Documents/ASU/CIS 508 Machine Learning/project"
gh repo clone cynthia-0316/smartops
cd smartops
```

Windows PowerShell equivalent:

```powershell
New-Item -ItemType Directory -Force "$HOME\Documents\ASU\CIS 508 Machine Learning\project"
cd "$HOME\Documents\ASU\CIS 508 Machine Learning\project"
gh repo clone cynthia-0316/smartops
cd smartops
```

(If the human already has a different preferred folder, use that. If a `smartops` folder already exists, do **not** delete it: `cd` into it and run `git pull` instead.)

**Check:** `git status -sb` shows `## main...origin/main`, and `ls` (Mac/Linux) / `dir` (Windows) shows `README.md`, `requirements.txt`, `notebooks`, `scripts`.

> **All remaining commands run from inside the `smartops` folder.**

## Step 7 – Create the virtual environment and install packages

A *virtual environment* is a private box of Python packages for this project so nothing conflicts with other projects.

```bash
python -m venv .venv          # use python3 on Mac/Linux if needed
```

Activate it:

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
  - If PowerShell says *"running scripts is disabled"*, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and answer `Y`, then activate again.
- **Windows Command Prompt:** `.venv\Scripts\activate.bat`

The terminal prompt should now start with `(.venv)`. Then:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes. Wait for it to finish.

**Check:** `pip list` shows pandas, numpy, scikit-learn, matplotlib and jupyter.

> Every time the human opens a new terminal to work on the project, they must `cd` into `smartops` and activate the environment again (the activate command above).

## Step 8 – Download the dataset

The data files are intentionally **not** stored in git, so download them:

```bash
python scripts/download_data.py
```

This puts 4 CSV files plus a README in `data/predictive-maintenance-data/` (~8 MB). It is safe to re-run.

**Check:** the script prints `ok` or `skip` for every file and "Done".

(Separately, `data/engine/engine_data.csv` already came with the clone – that is a different copy used by the existing notebooks.)

## Step 9 – Verify everything

```bash
python scripts/verify_setup.py
```

**Check:** every line shows `[PASS]` and the last line says `ALL GOOD - setup is complete.` If any line says FAIL, follow the hint printed next to it and re-run.

## Step 10 – Open the notebooks (human-facing smoke test)

```bash
jupyter notebook
```

A browser tab opens. Have the human open `notebooks/engine_condition_modeling.ipynb` to confirm it loads. To stop Jupyter later: press `Ctrl+C` in the terminal, then `y`.

Optional quick data peek in a Python shell or a new notebook cell:

```python
import pandas as pd
df = pd.read_csv("data/predictive-maintenance-data/train.csv")
print(df.shape)          # expect (14650, 23)
print(df["Engine Condition"].value_counts())
```

*Paths in notebooks are relative to the notebook's folder; from `notebooks/` the data is at `../data/predictive-maintenance-data/train.csv`.*

## Step 11 – Teach the team workflow (read aloud / summarize to the human)

Rules for working in this shared repo:

1. **Never work directly on `main`.** Make your own branch for each task:
   ```bash
   git switch main && git pull
   git switch -c yourname/short-task-name
   ```
2. Save work in small commits:
   ```bash
   git add <the files you changed>
   git commit -m "Short description of what you did"
   git push -u origin yourname/short-task-name
   ```
3. Open a **Pull Request** on GitHub (`gh pr create` or the green button on the website) and ask a teammate to review it before merging.
4. Start each work session with `git switch main && git pull` to get teammates' latest changes.
5. **Do not commit data files** (`data/` is git-ignored on purpose) and **never commit `.venv/`** or passwords/keys (`.env` files).
6. Clear notebook outputs or keep them small before committing notebooks, to avoid huge, conflicting diffs.
7. If anything looks scary ("merge conflict", "detached HEAD"), stop and ask the team – don't force anything.

## Step 12 – Report back

Tell the human, in plain language: what was installed, where the project folder is, how to re-open it later (`cd` into the folder, activate the venv, `jupyter notebook`), and that setup is complete. Then close the GitHub issue assigned to them (comment "Setup done" with the output of `python scripts/verify_setup.py`).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: git / python / gh` right after installing | Close **all** terminal windows and open a new one. On Windows, sign out/in if still failing. |
| `python` opens the Microsoft Store (Windows) | Install Python from python.org with **Add to PATH** ticked, or `winget install Python.Python.3.12`; then use `py -3 -m venv .venv`. |
| `python` is Python 2 or too old (Mac) | Use `python3` for every command, or `brew install python@3.12`. |
| `gh repo clone` says "Could not resolve to a Repository" / 404 | Invitation not accepted or wrong GitHub account logged in: redo Step 1, then `gh auth logout` and `gh auth login` with the right account. |
| `gh auth login` browser step fails | Try the other option "Paste an authentication token" **only if the human creates the token themselves** – never paste it into chat. |
| PowerShell: "running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`. |
| `pip install` fails building a package (very new Python, e.g. 3.14) | Install Python 3.12 (Step 4), delete the `.venv` folder you created, and redo Step 7 using that Python. |
| `pip` / `python` points to the wrong Python | Make sure the prompt shows `(.venv)`; use `python -m pip ...`. |
| `download_data.py` fails with network/SSL error | Retry on a different network (not blocked campus Wi-Fi/VPN); macOS python.org installs: run "Install Certificates.command" in /Applications/Python 3.x. |
| `verify_setup.py`: "virtual environment is active" FAIL | Activate `.venv` (Step 7) and re-run. |
| `jupyter: command not found` | Venv not active, or `pip install -r requirements.txt` did not finish – redo Step 7. |
| `git pull` says "local changes would be overwritten" | `git stash`, `git pull`, `git stash pop` – or ask the team. |
| Anything else | Copy the **full** error text and post it in the team chat. Do not retry random commands. |
