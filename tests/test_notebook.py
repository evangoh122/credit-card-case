import re
from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "colab_generate_15m.ipynb"
TOKEN_PATTERNS = [
    r"hf_[A-Za-z0-9]{20,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"sk-[A-Za-z0-9]{20,}",
    r"(?i)(token|secret|password|api_?key)\s*=\s*[\"'][^\"']{8,}[\"']",
]


def _load():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_is_valid_nbformat():
    nbformat.validate(_load())


def test_notebook_has_no_outputs():
    for cell in _load().cells:
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None


def test_notebook_has_no_token_like_strings():
    text = NOTEBOOK.read_text()
    for pattern in TOKEN_PATTERNS:
        assert not re.search(pattern, text), pattern


def test_notebook_reads_token_from_secrets_or_env_only():
    text = NOTEBOOK.read_text()
    assert "userdata.get" in text and "os.environ.get" in text
    assert not re.search(r"print\([^)]*token", text)
