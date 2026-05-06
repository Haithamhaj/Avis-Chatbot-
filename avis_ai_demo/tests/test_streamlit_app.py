import ast
from pathlib import Path


def test_streamlit_app_imports_without_rendering():
    import app_streamlit

    assert callable(app_streamlit.render_app)


def test_streamlit_adapter_has_initial_welcome_message():
    from avis_ai_demo.adapters.streamlit_adapter import WELCOME_MESSAGE

    assert "مرحبًا" in WELCOME_MESSAGE
    assert "حجز سيارة" in WELCOME_MESSAGE
    assert "الأسعار اليومية" in WELCOME_MESSAGE
    assert "الفروع" in WELCOME_MESSAGE
    assert "مساعدة على الطريق" in WELCOME_MESSAGE


def test_streamlit_import_stays_out_of_core():
    core_dir = Path(__file__).resolve().parents[1] / "core"
    for path in core_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "streamlit" for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert node.module != "streamlit", path
