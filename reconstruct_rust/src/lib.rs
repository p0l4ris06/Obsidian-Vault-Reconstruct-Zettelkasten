use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use std::collections::{HashSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use rayon::prelude::*;
use regex::{Regex, Captures};
use std::fs;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

// --- Constants / Regexes ---
lazy_static::lazy_static! {
    static ref UNSAFE_CHARS: Regex = Regex::new(r#"[\\/:*?"<>|]"#).unwrap();
    static ref FRONTMATTER_RE: Regex = Regex::new(r"(?s)^---\s*\n.*?\n---\s*\n").unwrap();
    static ref CODE_FENCE_RE: Regex = Regex::new(r"(?s)```.*?```").unwrap();
    static ref INLINE_CODE_RE: Regex = Regex::new(r"`[^`]+`").unwrap();
    static ref WIKILINK_RE: Regex = Regex::new(r"\[\[.*?\]\]").unwrap();
}

// --- Helper Functions (ported from Python) ---
fn safe_filename(title: &str) -> String {
    let cleaned = UNSAFE_CHARS.replace_all(title, "-");
    let trimmed = cleaned.trim_end_matches(". ");
    let sliced = if trimmed.len() > 200 { &trimmed[..200] } else { trimmed };
    if sliced.is_empty() {
        "Untitled".to_string()
    } else {
        sliced.to_string()
    }
}

fn mask_protected(text: &str) -> (String, Vec<(String, String)>) {
    let mut placeholders: Vec<(String, String)> = Vec::new();
    let mut masked_text = text.to_string();

    let mut replace_one = |m: Captures| {
        let token = format!("\x00PH{}\x00", placeholders.len());
        placeholders.push((token.clone(), m[0].to_string()));
        token
    };

    // Frontmatter (only replace once)
    masked_text = FRONTMATTER_RE.replacen(&masked_text, 1, |m: &Captures| replace_one(m.to_owned())).to_string();
    
    // Code fences
    masked_text = CODE_FENCE_RE.replace_all(&masked_text, |m: Captures| replace_one(m.to_owned())).to_string();

    // Inline code
    masked_text = INLINE_CODE_RE.replace_all(&masked_text, |m: Captures| replace_one(m.to_owned())).to_string();

    // Wikilinks
    masked_text = WIKILINK_RE.replace_all(&masked_text, |m: Captures| replace_one(m.to_owned())).to_string();

    (masked_text, placeholders)
}

fn restore_protected(text: &str, placeholders: &[(String, String)]) -> String {
    let mut restored_text = text.to_string();
    for (token, original) in placeholders {
        restored_text = restored_text.replace(token, original);
    }
    restored_text
}


#[pyfunction]
fn run_link_phase(vault_path_str: &str) -> PyResult<usize> {
    let vault_path = PathBuf::from(vault_path_str);
    if !vault_path.is_dir() {
        return Err(PyIOError::new_err(format!("Vault path does not exist or is not a directory: {}", vault_path_str)));
    }

    println!("INFO: === PHASE 2 (Rust): Auto-generating wiki-links ===");

    // Phase 1: Collect all markdown files and their titles (stems)
    let mut file_paths_and_titles: Vec<(PathBuf, String)> = Vec::new();
    let mut unique_titles: HashSet<String> = HashSet::new();

    for entry in WalkDir::new(&vault_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && e.path().extension().map_or(false, |ext| ext == "md"))
    {
        let path = entry.path().to_path_buf();
        let stem = path.file_stem().and_then(|s| s.to_str()).map_or_else(|| "".to_string(), |s| s.to_string());

        if !stem.is_empty() {
            // Filter out internal obsidian files and tracker file
            if path.components().any(|comp| comp.as_os_str() == ".obsidian") || stem.starts_with(".processed_files") {
                continue;
            }
            file_paths_and_titles.push((path, stem.clone()));
            unique_titles.insert(stem);
        }
    }

    println!("INFO: Linking {} titles across {} files...", unique_titles.len(), file_paths_and_titles.len());

    // Sort titles by length (longest first) to avoid partial replacements
    let mut sorted_titles: Vec<String> = unique_titles.into_iter().collect();
    sorted_titles.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b))); // Secondary sort for stable order

    let files_modified_count = AtomicUsize::new(0);

    // Phase 2: Process files in parallel to add links
    file_paths_and_titles.par_iter().for_each(|(file_path, current_title)| {
        match fs::read_to_string(file_path) {
            Ok(original_content) => {
                let (masked_content, placeholders) = mask_protected(&original_content);
                let mut current_masked_content = masked_content;
                let mut links_added_to_file = 0;

                for title in &sorted_titles {
                    if title == current_title {
                        continue; // Don't link a note to itself
                    }

                    // Word-boundary match, case-insensitive, first occurrence only
                    let pattern_str = format!(r"\b({})\b", regex::escape(title));
                    let link_pattern = Regex::new(&pattern_str).unwrap();

                    // Replace only the first occurrence
                    let replaced_content_tuple = link_pattern.replacen(&current_masked_content, 1, |caps: &Captures| {
                        links_added_to_file += 1;
                        format!("[[{}]]", &caps[1])
                    });
                    current_masked_content = replaced_content_tuple.to_string();
                }

                let result_content = restore_protected(&current_masked_content, &placeholders);

                if result_content != original_content {
                    match fs::write(file_path, result_content) {
                        Ok(_) => {
                            println!("INFO: Added {} link(s) -> {}", links_added_to_file, file_path.file_stem().unwrap().to_str().unwrap_or("UNKNOWN"));
                            files_modified_count.fetch_add(1, AtomicOrdering::SeqCst);
                        }
                        Err(e) => {
                            eprintln!("ERROR: Could not write to file {}: {}", file_path.display(), e);
                        }
                    }
                }
            },
            Err(e) => {
                eprintln!("ERROR: Could not read file {}: {}", file_path.display(), e);
            }
        }
    });

    println!("INFO: Phase 2 (Rust) complete.");
    Ok(files_modified_count.load(AtomicOrdering::SeqCst))
}

#[pymodule]
fn reconstruct_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_link_phase, m)?)?;
    Ok(())
}
