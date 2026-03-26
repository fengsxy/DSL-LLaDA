# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python codebase with top-level entry points:

- `inference.py`: CLI for standard remasking, SDE generation, and error correction.
- `app.py`: Gradio demo for side-by-side interactive generation.
- `dsl_modules.py`: reusable DSL/LLaDA model components and SNR utilities.
- `requirements.txt`: runtime dependencies.

Keep new modules focused and importable from the repository root. If tests are added, place them in a top-level `tests/` directory.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate a local environment.
- `pip install -r requirements.txt`: install runtime dependencies.
- `python inference.py --model Beta2 --mode standard --prompt "Question: 2+3?\nAnswer:"`: run CLI inference.
- `python inference.py --model Beta1 --mode sde --prompt "Write a story about a robot."`: run SDE generation.
- `python app.py`: start the Gradio demo on `http://localhost:7860`.

There is no formal build step; development is centered on local inference and demo runs.

## Coding Style & Naming Conventions
Follow existing Python style:

- 4-space indentation, UTF-8 source, and descriptive docstrings for public functions.
- `snake_case` for functions and variables, `UPPER_CASE` for constants, `CamelCase` for classes.
- Keep tensor math readable; prefer short helper functions over deeply nested blocks.
- Group standard-library imports before third-party imports.

No formatter is configured yet. If you introduce one, keep the diff narrow and document it here.

## Testing Guidelines
Automated tests are not present yet. Before submitting changes, run a manual smoke test against the affected entry point:

- CLI changes: run `python inference.py` with the relevant `--mode`.
- UI changes: run `python app.py` and verify the Gradio flow loads and streams output.

When adding tests, use `pytest`, name files `tests/test_*.py`, and cover both argument handling and generation-path regressions where feasible.

## Commit & Pull Request Guidelines
Git history currently starts with a single commit, `Initial release: DSL-LLaDA inference (standard, SDE, correction)`. Match that style with short, imperative subjects and optional context after a colon.

Pull requests should include a clear summary, the commands used for validation, any model/checkpoint assumptions, and screenshots only when `app.py` UI behavior changes.
