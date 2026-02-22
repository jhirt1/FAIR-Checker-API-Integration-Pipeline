# Contributing to FAIRness Pipeline

Thank you for your interest in contributing to the FAIRness Pipeline. This repository supports reproducible, publication-aligned FAIRness evaluation workflows. Contributions that improve reliability, transparency, documentation, or methodological rigor are especially welcome.

Please read this document before submitting issues or pull requests.

---

## Scope of Contributions

This repository centers a single production pipeline script:

`data_collection.py`

Contributions may include:

- Bug fixes
- Performance improvements
- Documentation clarifications
- Logging enhancements
- Reproducibility improvements
- Refactoring for clarity or maintainability
- Additional validation checks
- Improved configuration handling
- Expanded FAIR metric reporting

Major architectural changes should be discussed in an issue before implementation.

---

## Reporting Issues

If you encounter a bug or unexpected behavior, please open an issue including:

- A clear description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version
- Relevant log excerpts (if applicable)

If the issue involves FAIR-Checker API behavior, include the response status code and payload (with sensitive data removed).

---

## Submitting Changes

1. Fork the repository.
2. Create a new branch:
   ```
   git checkout -b feature/short-description
   ```
3. Make your changes.
4. Ensure the script runs successfully in `test` mode.
5. Commit with a clear, descriptive message.
6. Submit a pull request.

Pull requests should include:

- A summary of changes
- Rationale for the modification
- Any implications for reproducibility or outputs
- Updated documentation (if applicable)

---

## Coding Standards

Please aim for:

- Clear, readable Python
- Meaningful variable names
- Inline comments for non-obvious logic
- Structured logging (avoid `print` statements in production paths)
- Deterministic behavior where possible (e.g., fixed random seeds for sampling)

Avoid introducing unnecessary dependencies unless justified.

---

## Reproducibility Expectations

Because this project supports research workflows:

- Do not remove logging.
- Do not silently change sampling logic without documentation.
- Ensure that output file structures remain stable.
- Maintain backward compatibility where reasonable.

If a change alters output formats, document it clearly in `CHANGELOG.md`.

---

## Code of Conduct

Participation in this project is governed by the repository’s `CODE_OF_CONDUCT.md`. Please review it before contributing.

---

## Licensing

By contributing, you agree that your contributions will be licensed under the MIT License included in this repository.

---

## Questions

For methodological or research-related questions, open a discussion or issue so that responses remain transparent and archived.

Thank you for helping improve the FAIRness Pipeline.