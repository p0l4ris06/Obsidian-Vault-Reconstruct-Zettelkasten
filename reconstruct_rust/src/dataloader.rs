use pyo3::prelude::*;
use polars::prelude::*;
use std::path::PathBuf;
use std::fs::File;
use std::sync::Arc;

#[pyclass]
pub struct FastDataLoader {
    filenames: Vec<PathBuf>,
    batch_size: usize,
    seq_len: usize,
    tokenizer_bos_id: u32,
    current_shard_idx: usize,
    current_docs: Vec<String>,
    current_doc_idx: usize,
}

#[pymethods]
impl FastDataLoader {
    #[new]
    pub fn new(filenames: Vec<String>, batch_size: usize, seq_len: usize, bos_id: u32) -> Self {
        FastDataLoader {
            filenames: filenames.into_iter().map(PathBuf::from).collect(),
            batch_size,
            seq_len,
            tokenizer_bos_id: bos_id,
            current_shard_idx: 0,
            current_docs: Vec::new(),
            current_doc_idx: 0,
        }
    }

    pub fn next_batch_strings(&mut self) -> PyResult<Option<Vec<String>>> {
        if self.current_docs.is_empty() || self.current_doc_idx + self.batch_size > self.current_docs.len() {
            if !self.load_next_shard()? {
                return Ok(None);
            }
        }

        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if self.current_doc_idx < self.current_docs.len() {
                batch.push(self.current_docs[self.current_doc_idx].clone());
                self.current_doc_idx += 1;
            }
        }

        Ok(Some(batch))
    }
}

impl FastDataLoader {
    fn load_next_shard(&mut self) -> PyResult<bool> {
        if self.current_shard_idx >= self.filenames.len() {
            return Ok(false); 
        }

        let path = &self.filenames[self.current_shard_idx];
        self.current_shard_idx += 1;

        let file = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let df = ParquetReader::new(file)
            .finish()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let text_col = df.column("text")
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .str()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        self.current_docs = text_col.into_iter()
            .filter_map(|opt_s| opt_s.map(|s| s.to_string()))
            .collect();
        self.current_doc_idx = 0;
        
        Ok(true)
    }
}
