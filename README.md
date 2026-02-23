# FAIR-Checker API Integration Pipeline

[![DOI](https://zenodo.org/badge/1162863255.svg)](https://doi.org/10.5281/zenodo.18734665)

A reproducible pipeline for programmatic FAIRness evaluation using the FAIR-Checker API. This project supports publication-grade metadata quality assessment by constructing a citation-stratified sample, submitting DOI/URL targets to FAIR-Checker with retry logic and rate-limit jitter, handling all-zero metric payloads via within-bucket replacement, and exporting structured FAIR metric outputs for downstream analysis.

---

## Overview

This pipeline operationalizes FAIRness assessment across scholarly records or datasets through the following stages:

1. Load an Excel-based universe dataset.
2. Clean and normalize citation count values (Unicode normalization and hidden character removal).
3. Filter records by document type, publication year, and source exclusions.
4. Compute a citation-stratified sampling strategy using inverse-proportional selection and elbow-point minimum enforcement (via `KneeLocator`).
5. Sample records within each citation bucket.
6. Submit sampled records to the FAIR-Checker API (DOI preferred, URL fallback).
7. Detect and handle all-zero FAIR metric responses with replacement logic.
8. Aggregate API responses and compute FAIR component scores.
9. Export reproducible, publication-ready outputs.

---

## Repository Structure

This repository centers three primary files:

- `data_collection.py` — Main FAIRness pipeline
- `setup.sh` — Environment setup script
- `run.sh` — Pipeline execution script

Additional files may reflect exploratory or developmental work and are not required for the primary workflow.

---

## Quick Start

### 1. Environment Setup

From the root of the repository:

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:

- Verify Python 3.10 or higher
- Create a virtual environment (`.venv`)
- Install dependencies from `requirements.txt`
- Create the expected directory structure
- Make execution scripts executable

---

### 2. Add Input Data

Place your Excel input file inside:

```
Inbound/
```

---

### 3. Run the Pipeline

```bash
chmod +x run.sh
./run.sh
```

The execution script activates the virtual environment and runs `data_collection.py`.

---

## Expected Directory Structure

```
Inbound/
  <Excel input file>
Outbound/
  Logging/
  Sampling/
  Raw Data/
  Results/
  Error Record Reports/
```

All generated outputs are written beneath `Outbound/`.

---

## Output Artifacts

The pipeline generates:

- Structured logging files
- Sampling strategy spreadsheets
- Sampled datasets (original and updated if replacements occur)
- Replacement and removal logs
- Raw API JSON responses
- FAIR metric tables
- Final merged analysis dataset

All output files are timestamped for reproducibility.

---

## FAIR Metric Calculations

The pipeline computes:

- Findable (F) sum and percentage  
- Accessible (A) sum and percentage  
- Interoperable (I) sum and percentage  
- Reusable (R) sum and percentage  
- Weighted FAIR Score (0–100)

Final merged outputs are written to:

```
Outbound/Results/<sheet>/
```

---

## Configuration

Pipeline behavior can be configured directly within `data_collection.py`, including:

- `TARGET` — Name of Excel file located in `Inbound/`
- `SHEET_NAME` — Sheet to process
- `MODE` — `test`, `full`, or `rerun`
- `ROOT` — Base working directory

---

## Requirements

- Python 3.10 or higher
- Internet access (for FAIR-Checker API calls)

Dependencies are installed automatically via `setup.sh` using the pinned versions in `requirements.txt`.

---

## Citation

If you use this software in academic work, please cite:

Hirt, J. (2026). FAIR-Checker API Integration Pipeline [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.18734665

### BibTeX

```bibtex
@software{hirt_fairness_pipeline_2026,
  author    = {Hirt, Juliana},
  title     = {FAIR-Checker API Integration Pipeline},
  year      = {2026},
  doi       = {10.5281/zenodo.18734665},
  url       = {https://doi.org/10.5281/zenodo.18734665},
  publisher = {Zenodo}
}
```

---

## License

MIT License

Copyright (c) 2026 Juliana Hirt

See the `LICENSE` file for full license text.

---

## AI Disclosure

No AI tools were used for the creation of any code or documentation in this repository.
