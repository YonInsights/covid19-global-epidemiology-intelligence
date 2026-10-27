"""Unit test for automated reporting script."""

from scripts.generate_report import generate_summary


def test_generate_summary(capsys):
    """Verify that reporting utility generates valid markdown output without errors."""
    generate_summary()
    captured = capsys.readouterr()
    assert "Global Surveillance KPIs" in captured.out
    assert "Top 10 Severest Outbreak Epicenters" in captured.out
    assert "US" in captured.out
