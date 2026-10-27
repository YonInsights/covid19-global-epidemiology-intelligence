# Contributing Guidelines

Thank you for your interest in contributing to the **COVID-19 Global Epidemiology Intelligence & Health Risk Analytics Platform**.

## Code Standards
- **Python Version:** Python 3.10+
- **Style Guide:** Adhere to PEP 8 standards with descriptive variable naming.
- **Type Hints:** Use standard Python type hinting across all function signatures.
- **Testing:** Add PyTest test coverage in `tests/` for all new metric calculations or analytical features. Run `pytest -v` to ensure 100% pass rate.

## Development Workflow
1. Clone the repository and create an isolated virtual environment (`python -m venv venv`).
2. Activate the virtual environment and install requirements (`pip install -r requirements.txt`).
3. Create a feature branch (`git checkout -b feature/surveillance-enhancement`).
4. Implement your changes, run tests (`pytest -v`), and execute checks (`python scripts/run_checks.py`).
5. Submit a pull request with a descriptive summary of changes.
