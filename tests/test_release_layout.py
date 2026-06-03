from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_excludes_external_snapshots_and_workspace_extras():
    assert not (ROOT / "external").exists()
    assert not (ROOT / "workspace_extras").exists()


def test_readme_describes_clean_release_layout():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "external/" not in readme
    assert "workspace_extras/" not in readme
    assert "paper_figures/" not in readme
    assert "paper.tex" not in readme
    assert "eval_data/" in readme
    assert "dsl_llada/" in readme
