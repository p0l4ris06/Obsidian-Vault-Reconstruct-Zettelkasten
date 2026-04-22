# Vault Reconstruct 🧬

**Transform fragmented notes into a grounded, high-performance Obsidian Zettelkasten.**

Vault Reconstruct is a unified, phase-aware pipeline designed to rebuild and optimize Obsidian vaults. It moves beyond simple "AI writing" by implementing a **Local RAG (Retrieval-Augmented Generation)** pipeline that grounds knowledge synthesis in real academic data (arXiv, PubMed, Wikipedia).

## ✨ Key Features

- **Unified Dashboard**: Manage your entire vault from the `vault-recon` TUI.
- **5-Phase Pipeline**:
    - **Phase 0**: Rescue and repair quarantined or stuck notes.
    - **Phase 1**: Atomic splitting of large journals into atomic Zettels.
    - **Phase 2**: High-speed linking and cross-referencing (Rust-powered).
    - **Phase 3**: Automated YAML frontmatter and tag standardization.
    - **Phase 4**: Map of Content (MOC) generation for high-level navigation.
- **Local RAG Research**: Synthesize new research notes grounded in facts. Crawls external academic APIs and indexes them into a local vector store for offline use.
- **Vault Maintenance**: Auto-fix broken links, standardize tags (UK English), and prune health reports.
- **Anki Integration**: Export your Zettels directly to Anki flashcards with one command.

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-repo/vault-reconstruct.git
cd vault-reconstruct
uv pip install -e .
```

### 2. Configuration

Copy and configure your environment variables:
```bash
cp .env.example .env
```

### 3. Launch the Dashboard

The recommended way to use Vault Reconstruct is via the interactive Text User Interface (TUI):

```bash
vault-recon
```

## 🧠 Local RAG Pipeline

To generate fact-grounded research notes:
1. **Sync**: Run `Sync Knowledge (RAG)` in the dashboard. This crawls arXiv, PubMed, and Wiki for your vault tags.
2. **Generate**: Use `Grounded Research Note` to write a literature note connecting two topics using your local index.

## 🚄 Performance

Link reconstruction (Phase 2) is powered by a custom **Rust module** for maximum speed on large vaults (>10k notes).

To build the Rust module locally:
```bash
cd reconstruct_rust
maturin develop --release
```

## 🛠️ Technical Details

All CLI tools are located in the `tools/` directory, and utility/build scripts are in `scripts/`.

- **Tools**: `tools/vault_reconstruct.py`, `tools/vault_maintenance.py`, `tools/vault_researcher.py`.
- **Scripts**: `scripts/build_exe.ps1`, `scripts/create_shortcut.ps1`.

- **Python 3.10+**
- **Ollama** (Local LLM backend)
- **Rust/Cargo** (Only for building the performance module)
- **SQLite** (For local indexing)

## 🩺 Diagnostics

Run the "Doctor" to verify your AI providers and environment setup:
```bash
python vault_doctor.py --all
```

---
*Maintained by the Advanced Agentic Coding team.*
