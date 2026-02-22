# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-02-22

### Added
- Initial public repository release.
- End-to-end FAIRness evaluation pipeline integrating with the FAIR-Checker API.
- Citation-stratified sampling strategy using inverse-proportional selection and elbow-point minimum enforcement.
- Automated API submission with retry logic and rate-limit jitter handling.
- All-zero FAIR score detection with within-bucket replacement logic.
- Structured logging for sampling, API responses, replacements, and removals.
- FAIR metric aggregation and computation of component scores (Findable, Accessible, Interoperable, Reusable).
- Weighted FAIR score calculation (0–100 scale).
- Export of publication-ready outputs (sampling artifacts, metrics tables, final analysis dataset).
- Zenodo DOI archival release.
- MIT License.

### Notes
- Version 1.0.0 represents the first stable, publication-aligned release of the FAIRness Pipeline.
- This release corresponds to the archived Zenodo record associated with DOI: 10.5281/zenodo.18732906.