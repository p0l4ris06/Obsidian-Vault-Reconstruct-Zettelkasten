mod dataloader;
use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use std::collections::HashSet;
use std::path::PathBuf;
use walkdir::WalkDir;
use rayon::prelude::*;
use regex::{Regex, Captures};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

lazy_static::lazy_static! {
    static ref UNSAFE_CHARS: Regex = Regex::new("[\\\\/:*?\\\"<>|]").unwrap();
    static ref FRONTMATTER_RE: Regex = Regex::new("(?s)^---\\s*\\n.*?\\n---\\s*\\n").unwrap();
    static ref CODE_FENCE_RE: Regex = Regex::new("(?s)```.*?```").unwrap();
    static ref INLINE_CODE_RE: Regex = Regex::new("`[^`]+`").unwrap();
    static ref WIKILINK_RE: Regex = Regex::new("\\[\\[.*?\\]\\]").unwrap();
}

fn mask_protected(text: &str) -> (String, Vec<(String, String)>) {
    let mut placeholders: Vec<(String, String)> = Vec::new();
    let mut masked_text = text.to_string();

    let mut replace_one = |m: &Captures| {
        let token = format!("\x00PH{}\x00", placeholders.len());
        placeholders.push((token.clone(), m[0].to_string()));
        token
    };

    masked_text = FRONTMATTER_RE.replacen(&masked_text, 1, &mut replace_one).to_string();
    masked_text = CODE_FENCE_RE.replace_all(&masked_text, &mut replace_one).to_string();
    masked_text = INLINE_CODE_RE.replace_all(&masked_text, &mut replace_one).to_string();
    masked_text = WIKILINK_RE.replace_all(&masked_text, &mut replace_one).to_string();

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
        return Err(PyIOError::new_err(format!("Vault path does not exist: {}", vault_path_str)));
    }

    println!("INFO: === PHASE 2 (Rust): Auto-generating wiki-links ===");

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
            if path.components().any(|comp| comp.as_os_str() == ".obsidian") || stem.starts_with(".processed_files") {
                continue;
            }
            file_paths_and_titles.push((path, stem.clone()));
            unique_titles.insert(stem);
        }
    }

    let mut sorted_titles: Vec<String> = unique_titles.into_iter().collect();
    sorted_titles.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

    let files_modified_count = AtomicUsize::new(0);

    file_paths_and_titles.par_iter().for_each(|(file_path, current_title)| {
        if let Ok(original_content) = fs::read_to_string(file_path) {
            let (masked_content, placeholders) = mask_protected(&original_content);
            let mut current_masked_content = masked_content;

            for title in &sorted_titles {
                if title == current_title { continue; }

                let pattern_str = format!("\\b({})\\b", regex::escape(title));
                if let Ok(link_pattern) = Regex::new(&pattern_str) {
                    current_masked_content = link_pattern.replacen(&current_masked_content, 1, |caps: &Captures| {
                        format!("[[{}]]", &caps[1])
                    }).to_string();
                }
            }

            let result_content = restore_protected(&current_masked_content, &placeholders);
            if result_content != original_content {
                if fs::write(file_path, result_content).is_ok() {
                    files_modified_count.fetch_add(1, AtomicOrdering::SeqCst);
                }
            }
        }
    });

    Ok(files_modified_count.load(AtomicOrdering::SeqCst))
}

#[pymodule]
fn reconstruct_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_link_phase, m)?)?;
    m.add_class::<dataloader::FastDataLoader>()?;
    Ok(())
}