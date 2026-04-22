# Vault Reconstruct: Command Reference 🛠️

This document provides a technical guide to the CLI entry points for Vault Reconstruct. While the **`vault-recon` TUI** (run via `python vault_hud.py`) is the recommended interface, these scripts can be run headless for automation.

---

## 🚄 Core Pipeline: `tools/reconstruct.py`

The main phase-aware rebuilder.

**Usage:** `python tools/reconstruct.py [options]`

| Flag | Description |
| :--- | :--- |
| `--phase N` | Run only a specific phase (0-4). |
| `--provider X` | Override LLM provider (ollama, gemini, azure, autoresearch). |
| `--vault PATH` | Path to the output vault (defaults to config). |
| `--dry-run` | Analyze files but don't write changes. |

### Phases:
- **0**: Recovery (Fixes QUARANTINE notes)
- **1**: Splitting (Atomic note creation)
- **2**: Linking (Wikilink generation)
- **3**: Frontmatter (YAML & Tagging)
- **4**: MOC (Map of Content creation)

---

## 🧬 Research & RAG: `tools/research.py`

Grounded knowledge synthesis.

**Usage:** `python tools/research.py [topic_a] [topic_b] [options]`

| Flag | Description |
| :--- | :--- |
| `--sync` | **(IMPORTANT)** Crawl and index knowledge from PubMed/arXiv/Wiki. |
| `--rag` | Enable Local RAG context for generating grounded notes. |
| `--provider X` | Override synthesis model provider (defaults to `ollama`). |

---

## 🏥 Maintenance: `tools/maintenance.py`

Vault health and repair tools.

**Usage:** `python tools/maintenance.py [options]`

| Flag | Description |
| :--- | :--- |
| `--fix-tags` | Standardize all tags to UK English. |
| `--fix-links` | Auto-repair broken wikilinks via fuzzy matching. |
| `--repair` | Attempt automated recovery of stuck/broken notes. |

---

## 🩺 Diagnostics: `tools/doctor.py`

System verification.

**Usage:** `python tools/doctor.py [options]`

| Flag | Description |
| :--- | :--- |
| `--all` | Check LLM connections, environment vars, and paths. |
| `--ping` | Test latency and connectivity to configured providers. |

---

## 📥 Exports: `tools/anki_exporter.py`

Flashcard generation.

**Usage:** `python tools/anki_exporter.py [options]`

| Flag | Description |
| :--- | :--- |
| `--vault PATH` | Path to the source vault. |
| `--out PATH` | Directory to save flashcard .apkg or .csv files. |
| `--reset` | Export all notes, ignoring the delta tracker. |

---
*Note: All core tools are now located in the `/tools` directory. Build scripts can be found in `/scripts`.*

