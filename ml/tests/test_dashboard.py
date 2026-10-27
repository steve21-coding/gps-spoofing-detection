"""Headless tests: every dashboard page renders without exceptions using built-in demo data."""
from __future__ import annotations

import pytest

from ml.src.config import ARTIFACTS_DIR, ROOT

st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not (ARTIFACTS_DIR / "metrics.json").exists(), reason="no trained artifacts")

PAGES = ["app.py", "pages/1_Model_Comparison.py", "pages/2_Data_Explorer.py", "pages/3_Spoofing_Analysis.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = st_testing.AppTest.from_file(str(ROOT / "dashboard" / page), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("scenario", ["jump", "drift", "dropout", "freeze"])
def test_analysis_page_demo_scenarios(scenario):
    at = st_testing.AppTest.from_file(str(ROOT / "dashboard/pages/3_Spoofing_Analysis.py"), default_timeout=90).run()
    at.selectbox(key="analysis_scen").select(scenario).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Potential spoofing rows" in labels


def test_analysis_attack_vs_normal_demo_counts():
    at = st_testing.AppTest.from_file(str(ROOT / "dashboard/pages/3_Spoofing_Analysis.py"), default_timeout=90).run()
    vals = {}
    for sc in ("normal", "freeze"):
        at.selectbox(key="analysis_scen").select(sc).run()
        vals[sc] = int(next(m.value for m in at.metric if m.label == "Potential spoofing rows").replace(",", ""))
    assert vals["freeze"] > 50 and vals["normal"] < 10
