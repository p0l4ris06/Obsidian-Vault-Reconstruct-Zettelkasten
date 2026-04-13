# Vault Reconstruct

This repo contains a set of scripts for:
- Converting markdown notes into an Obsidian Zettelkasten-style vault
- Adding wikilinks
- Consolidating tags
- Generating MOCs (Maps of Content)
- Producing vault health reports / fixing broken links

The scripts originally lived as separate entrypoints. Use `vault_cli.py` to run them from one place.

## Quick start

Run a full Ollama-based conversion pipeline:

```bash
python vault_cli.py convert-ollama
```

Recommended local model for your machine (per `llm-checker smart-recommend`):

```bash
ollama pull qwen2.5-coder:0.5b-base-q8_0
```

Quick integration + performance dry-run (no vault writes):

```bash
# Current provider only
python vault_cli.py doctor -- --ping --ping-repeats 3

# Test ollama + gemini + azure (providers missing keys are skipped)
python vault_cli.py doctor -- --all --ping --ping-repeats 2
```

Launch the interactive HUD (recommended on Windows):

```bash
python vault_hud.py
```

### Desktop app (.exe)

Build the app (recommended: onedir):

```powershell
.\build_exe.ps1 -Mode onedir
```

Compatibility note: `.\build_hud.ps1` is a thin wrapper around `.\build_exe.ps1` (same flags/behavior).

There are two PyInstaller modes:

- **Recommended (most reliable)**: onedir build  
  Run: `dist\VaultHUD\VaultHUD.exe`  
  This does **not** extract to a temp `_MEI...` folder, so it avoids common onefile DLL-load failures.
  - If launched from **cmd.exe**, it will **auto-relaunch into PowerShell** for better rendering.
  - To disable relaunch: run with `--no-relaunch`

- **Onefile build**: `dist\VaultHUD.exe`  
  If you hit an error like `Failed to load Python DLL ... python311.dll`, use the **onedir** build instead.

#### Create a shortcut (Windows)

After building the onedir app, you can create a Desktop shortcut:

```powershell
.\create_shortcut.ps1 -Location Desktop
```

Or create both Desktop + Start Menu shortcuts:

```powershell
.\create_shortcut.ps1 -Location Both
```

Run the Gemini-based splitter/linker:

```bash
python vault_cli.py convert-gemini
```

Run architecture passes (threaded linking + tag consolidation + MOCs):

```bash
python vault_cli.py architecture
```

Generate Anki decks from your zettels:

```bash
python vault_cli.py anki-export -- --vault "D:\Obsidian Vault" --out "D:\Anki Decks"
```

## Passing options to underlying scripts

Anything after `--` is passed through to the underlying script.

Example (vault improver dry-run):

```bash
python vault_cli.py improve -- --vault "D:\Obsidian Vault" --dry-run
```

Example (tag consolidation dry-run with custom min count):

```bash
python vault_cli.py tag-consolidate -- --dry-run --min 5
```

## Repo hygiene

Generated logs (`*.log`) and generated mappings (`tag_mapping.json`, `tag_changes.json`, etc.) are ignored via `.gitignore` to keep the repo compact.

## Model backends (Ollama / Gemini / Azure OpenAI)

The main Ollama pipeline (`Vault Reconstruct Ollama.py`) can now use different backends via environment variables:

- **Ollama (default)**:
  - Set `VAULT_LLM_PROVIDER=ollama`
  - Optional cloud-first: set `OLLAMA_API_KEY`
  - Optional local model override: set `VAULT_OLLAMA_MODEL` (single model for **all** tasks)
  - Optional **JSON phases only** (split / quarantine / AI links): set `VAULT_OLLAMA_INSTRUCT_MODEL` to an installed instruct/chat-tuned tag
  - Optional cloud model override: set `VAULT_OLLAMA_CLOUD_MODEL`
  - If `VAULT_OLLAMA_MODEL` is **not** set, the Ollama pipelines pick:
    - an **instruction-tuned** installed model for JSON-heavy prompts
    - a `llm-checker smart-recommend` pick for **MOC prose** (Map of Content body text)
- **Gemini**:
  - Set `VAULT_LLM_PROVIDER=gemini`
  - Set `GEMINI_API_KEY`
  - Optional model override: set `VAULT_GEMINI_MODEL`
- **Azure OpenAI**:
  - Set `VAULT_LLM_PROVIDER=azure`
  - Set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
  - Optional: set `AZURE_OPENAI_API_VERSION` (default: `2024-10-21`)
  - Set `VAULT_AZURE_MODEL` to your Azure deployment name (not the base model name)

### Ollama notes (Windows)

- If you have `OLLAMA_HOST=0.0.0.0:11434`, that’s a **bind address** (server-side). Clients typically need a connectable address like `127.0.0.1:11434`.
- If `llm-checker` recommends a model tag you haven’t pulled yet, `doctor` will still report it; JSON-heavy work prefers **instruct/chat** tags you already have. Pull the checker’s tag with `ollama pull <tag>` if you want that exact weight for prose routes.

## .env (recommended)

Copy `\.env.example` to `\.env` and fill in the keys you actually use.

Minimum examples:

- **Ollama local**: set `VAULT_LLM_PROVIDER=ollama` and ensure `OLLAMA_HOST` is correct.
- **Gemini**: set `VAULT_LLM_PROVIDER=gemini` + `GEMINI_API_KEY`
- **Azure OpenAI**: set `VAULT_LLM_PROVIDER=azure` + `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` + `VAULT_AZURE_MODEL`

Run a quick dry-run sanity check:

```bash
python vault_cli.py doctor -- --all --ping --ping-repeats 2
```


