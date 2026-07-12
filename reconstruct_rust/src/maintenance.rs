use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use walkdir::WalkDir;
use rayon::prelude::*;
use regex::Regex;
use fancy_regex::Regex as FancyRegex;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use strsim::levenshtein;

lazy_static::lazy_static! {
    static ref TAG_PATTERN: FancyRegex = FancyRegex::new(r"(?<!\w)#([\w/\-]+)").unwrap();
    static ref FRONTMATTER_TAGS_PATTERN: Regex = Regex::new(r"(?m)^---\s*\ntags:\s*\n((?:  - .+\n?)+)").unwrap();
    static ref YAML_TAG_VALUE_PATTERN: Regex = Regex::new(r"  - (.+)").unwrap();
    static ref FRONTMATTER_STRIP_PATTERN: Regex = Regex::new(r"(?s)^---\s*\n.*?\n---\s*\n").unwrap();
    static ref WIKILINK_PATTERN: Regex = Regex::new(r"\[\[([^\]|#]+)").unwrap();
}

#[derive(Clone)]
struct ScannerConfig {
    vault_path: PathBuf,
    fix_tags: bool,
    fix_links: bool,
    dry_run: bool,
    short_threshold: usize,
    tag_mappings: HashMap<String, String>,
}

fn find_fuzzy_match<'a>(broken: &str, titles: &'a [String], title_index: &HashMap<String, String>) -> Option<&'a String> {
    let norm = broken.to_lowercase().trim().to_string();
    if let Some(exact) = title_index.get(&norm) {
        return Some(titles.iter().find(|t| *t == exact).unwrap());
    }
    
    // Fuzzy match (similar to difflib.get_close_matches with cutoff 0.8)
    let mut best_match: Option<&String> = None;
    let mut best_score = 0.0;
    
    for title in titles {
        let title_norm = title.to_lowercase();
        let max_len = std::cmp::max(norm.chars().count(), title_norm.chars().count()) as f64;
        if max_len == 0.0 { continue; }
        
        let dist = levenshtein(&norm, &title_norm) as f64;
        let score = 1.0 - (dist / max_len);
        
        if score >= 0.8 && score > best_score {
            best_score = score;
            best_match = Some(title);
        }
    }
    
    best_match
}

#[pyfunction]
#[pyo3(signature = (vault_path_str, fix_tags, fix_links, dry_run, short_threshold, tag_mappings))]
pub fn run_maintenance(
    vault_path_str: &str,
    fix_tags: bool,
    fix_links: bool,
    dry_run: bool,
    short_threshold: usize,
    tag_mappings: HashMap<String, String>
) -> PyResult<String> {
    let vault_path = PathBuf::from(vault_path_str);
    if !vault_path.is_dir() {
        return Err(PyIOError::new_err(format!("Vault path does not exist: {}", vault_path_str)));
    }
    
    let config = ScannerConfig {
        vault_path,
        fix_tags,
        fix_links,
        dry_run,
        short_threshold,
        tag_mappings,
    };
    
    let mut file_paths_and_titles: Vec<(PathBuf, String)> = Vec::new();
    
    for entry in WalkDir::new(&config.vault_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && e.path().extension().map_or(false, |ext| ext == "md"))
    {
        let path = entry.path().to_path_buf();
        let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        
        if !stem.is_empty() && !stem.starts_with('.') {
            file_paths_and_titles.push((path, stem));
        }
    }
    
    let total_notes = file_paths_and_titles.len();
    
    let tags: Arc<Mutex<HashMap<String, HashSet<String>>>> = Arc::new(Mutex::new(HashMap::new()));
    let broken: Arc<Mutex<HashMap<String, Vec<String>>>> = Arc::new(Mutex::new(HashMap::new()));
    let short: Arc<Mutex<Vec<(String, String, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let quarantine: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    
    // Build title index for fast lookup
    let titles: Vec<String> = file_paths_and_titles.iter().map(|(_, t)| t.clone()).collect();
    let mut title_index: HashMap<String, String> = HashMap::new();
    for t in &titles {
        title_index.insert(t.to_lowercase(), t.clone());
    }

    let modified_files = AtomicUsize::new(0);

    file_paths_and_titles.par_iter().for_each(|(f, title)| {
        if title.contains("QUARANTINE_") {
            quarantine.lock().unwrap().push(f.to_string_lossy().to_string());
            return;
        }
        
        if let Ok(mut content) = fs::read_to_string(f) {
            let original_content = content.clone();
            
            // Parse Tags
            let mut local_tags = Vec::new();
            for cap in TAG_PATTERN.captures_iter(&content) {
                if let Ok(c) = cap {
                    local_tags.push(c[1].to_lowercase());
                }
            }
            
            if let Some(cap) = FRONTMATTER_TAGS_PATTERN.captures(&content) {
                for tag_cap in YAML_TAG_VALUE_PATTERN.captures_iter(&cap[1]) {
                    local_tags.push(tag_cap[1].trim().to_lowercase());
                }
            }
            
            if !local_tags.is_empty() {
                let mut tags_lock = tags.lock().unwrap();
                for t in &local_tags {
                    tags_lock.entry(t.clone()).or_insert_with(HashSet::new).insert(title.clone());
                }
            }
            
            // Parse Short Notes
            let body = FRONTMATTER_STRIP_PATTERN.replace(&content, "");
            let body_trim = body.trim();
            if body_trim.len() < config.short_threshold {
                short.lock().unwrap().push((title.clone(), f.to_string_lossy().to_string(), body_trim.len()));
            }
            
            // Parse Links
            let mut local_broken = Vec::new();
            for cap in WIKILINK_PATTERN.captures_iter(&content) {
                let target = cap[1].trim().to_string();
                if !title_index.values().any(|t| t == &target) {
                    local_broken.push(target);
                }
            }
            
            if !local_broken.is_empty() {
                broken.lock().unwrap().insert(title.clone(), local_broken.clone());
            }
            
            // FIX LOGIC
            let mut is_modified = false;
            
            if config.fix_tags {
                for tag in &local_tags {
                    if let Some(new_tag) = config.tag_mappings.get(tag) {
                        // Case-insensitive replacement for #tag
                        let tag_re = Regex::new(&format!(r"(?i)#{}", regex::escape(tag))).unwrap();
                        content = tag_re.replace_all(&content, &format!("#{}", new_tag)).to_string();
                        
                        // Case-insensitive replacement for - tag (YAML list)
                        let yaml_tag_re = Regex::new(&format!(r"(?i)- {}", regex::escape(tag))).unwrap();
                        content = yaml_tag_re.replace_all(&content, &format!("- {}", new_tag)).to_string();
                        
                        is_modified = true;
                    }
                }
            }
            
            if config.fix_links {
                for b in &local_broken {
                    if let Some(match_title) = find_fuzzy_match(b, &titles, &title_index) {
                        // Case-insensitive replacement for [[b]]
                        let link_re = Regex::new(&format!(r"(?i)\[\[{}\]\]", regex::escape(b))).unwrap();
                        content = link_re.replace_all(&content, &format!("[[{}]]", match_title)).to_string();
                        
                        // Case-insensitive replacement for [[b|
                        let pipe_link_re = Regex::new(&format!(r"(?i)\[\[{}\|", regex::escape(b))).unwrap();
                        content = pipe_link_re.replace_all(&content, &format!("[[{}|", match_title)).to_string();
                        
                        is_modified = true;
                    }
                }
            }
            
            if is_modified && content != original_content && !config.dry_run {
                if fs::write(f, content).is_ok() {
                    modified_files.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
    });
    
    let tags = Arc::try_unwrap(tags).unwrap().into_inner().unwrap();
    let broken = Arc::try_unwrap(broken).unwrap().into_inner().unwrap();
    let short = Arc::try_unwrap(short).unwrap().into_inner().unwrap();
    let quarantine = Arc::try_unwrap(quarantine).unwrap().into_inner().unwrap();
    
    let broken_count: usize = broken.values().map(|v| v.len()).sum();
    
    // Construct JSON result
    let result = serde_json::json!({
        "total_notes": total_notes,
        "broken_links": broken_count,
        "short_notes": short.len(),
        "quarantined": quarantine.len(),
        "modified_files": modified_files.load(Ordering::SeqCst),
        "tags_count": tags.len(),
    });
    
    Ok(result.to_string())
}
