# FAIR-Checker API Integration Pipeline

[![DOI](https://zenodo.org/badge/1162863255.svg)](https://doi.org/10.5281/zenodo.18732906)

A reproducible pipeline for programmatic FAIRness evaluation using the FAIR-Checker API. This project supports publication-grade metadata quality assessment by constructing a citation-stratified sample, submitting DOI/URL targets to FAIR-Checker with retry logic and rate-limit jitter, handling all-zero metric payloads via within-bucket replacement, and exporting structured FAIR metric outputs for downstream analysis.

---

## Overview

This pipeline operationalizes FAIRness assessment across scholarly records or datasets through the following stages:

1. Load an Excel-based universe dataset.
2. Merge optional zero-citation support files.
3. Clean and normalize citation count values (Unicode normalization and hidden character removal).
4. Filter records by document type, publication year, and source exclusions.
5. Compute a citation-stratified sampling strategy using inverse-proportional selection and elbow-point minimum enforcement (via `KneeLocator`).
6. Sample records within each citation bucket.
7. Submit sampled records to the FAIR-Checker API (DOI preferred, URL fallback).
8. Detect and handle all-zero FAIR metric responses with replacement logic.
9. Aggregate API responses and compute FAIR component scores.
10. Export reproducible, publication-ready outputs.

---

## Repository Focus

This repository centers one production script:

`data_collection.py`

Other files in the repository may reflect exploratory or developmental work and are not required for the primary FAIRness pipeline workflow.

---

## Expected Directory Structure

The script assumes a working directory structured as:

```
Sample Collection/
  Inbound/
    <Excel input file>
    0 citation support files/
      <discipline>/
        *.txt
  Outbound/
    Logging/
    Sampling/
    Raw Data/
    Results/
    Error Record Reports/
```

All generated outputs are written beneath `Sample Collection/Outbound/`.

---

## Requirements

- Python 3.10+
- Internet access (for FAIR-Checker API calls)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Minimum required packages:

- pandas  
- numpy  
- requests  
- openpyxl  
- kneed  
- matplotlib  

---

## Configuration

Open `data_collection.py` and confirm:

- `TARGET` — Name of Excel file located in `Inbound/`
- `SHEET_NAME` — Sheet to process
- `TXT_FILE_DIR` — Subdirectory containing zero-citation support files
- `MODE` — `test`, `full`, or `rerun`
- `ROOT` — Base working directory (`Sample Collection` by default)

---

## Running the Pipeline

```bash
python data_collection.py
```

### Modes

**test**  
Runs a limited number of API calls to validate workflow behavior.

**full**  
Executes the complete sampling, API submission, replacement logic, and aggregation workflow.

**rerun**  
Skips records where API response JSON already exists (useful after interruption).

---

## Sampling Strategy

The pipeline builds a citation-stratified sample using:

- Inverse-proportional selection across citation buckets
- Power normalization
- Elbow-point minimum sample enforcement (via `KneeLocator`)

Sampling artifacts are exported to:

`Outbound/Sampling/<sheet>/`

Including:

- Sampling strategy spreadsheet  
- Sampled dataset  
- Distribution visualization  
- Replacement and removal logs  

---

## API Submission and All-Zero Handling

Each sampled record is submitted to the FAIR-Checker legacy metrics endpoint.

If the API returns a payload where all metric scores equal `0`, the pipeline:

1. Logs the all-zero event.
2. Attempts to replace the record with a different target from the same citation bucket.
3. Re-calls the API for the replacement record.
4. Removes the record if no eligible replacement exists.

All actions are logged to preserve transparency and reproducibility.

Raw API responses are stored as:

`Outbound/Raw Data/<sheet>/api_response_<idx>.json`

---

## FAIR Metric Calculations

From the FAIR-Checker response, the pipeline computes:

- Findable (F) sum and percentage  
- Accessible (A) sum and percentage  
- Interoperable (I) sum and percentage  
- Reusable (R) sum and percentage  
- Weighted FAIR Score (0–100)

Final merged outputs are written to:

`Outbound/Results/<sheet>/final_analysis_<timestamp>.xlsx`

All outputs are timestamped.

---

## Citation

If you use this software in academic work, please cite:

Hirt, Juliana. (2026). *FAIRness Pipeline (FAIR-Checker Integrated).* Zenodo. https://doi.org/10.5281/zenodo.18732906

### BibTeX

```bibtex
@software{hirt_fairness_pipeline_2026,
  author    = {Hirt, Juliana},
  title     = {FAIRness Pipeline (FAIR-Checker Integrated)},
  year      = {2026},
  doi       = {10.5281/zenodo.18732906},
  url       = {https://doi.org/10.5281/zenodo.18732906},
  publisher = {Zenodo}
}
```

---

## License

MIT License

Copyright (c) 2026 Juliana Hirt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...

(See full MIT license in the `LICENSE` file.)
