from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_homepage_contains_primary_project_links():
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert "DSL-LLaDA: Scaling Continuous Denoising to 8B Masked Diffusion LMs" in html
    assert "http://138.23.28.165:7860" in html
    assert "https://github.com/fengsxy/DSL-LLaDA" in html
    assert "https://huggingface.co/liddlefish/DSL-LLaDA-Beta1" in html


def test_homepage_keeps_paper_assets_out_of_release():
    html = (ROOT / "index.html").read_text(encoding="utf-8")

    assert "paper_figures/" not in html
    assert "paper.tex" not in html


def test_readme_points_to_live_demo():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "http://138.23.28.165:7860" in readme
