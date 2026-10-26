import sys
import types

import pytest

from ccc.sim import upload

SENTINEL = "hf_SENTINEL_not_a_real_token_0123456789"


class FakeApi:
    instances: list = []

    def __init__(self, token=None):
        self.token = token
        self.calls = []
        FakeApi.instances.append(self)

    def create_repo(self, *args, **kwargs):
        self.calls.append("create_repo")

    def upload_large_folder(self, **kwargs):
        self.calls.append("upload_large_folder")
        if kwargs.get("repo_id") == "boom/boom":
            raise RuntimeError("upload failed")


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    return tmp_path


@pytest.fixture(autouse=True)
def fake_hub(monkeypatch):
    FakeApi.instances = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))


def _run(folder, *extra):
    upload.main(["--dir", str(folder), "--repo-id", "u/r", *extra])


def test_token_comes_only_from_env_and_is_never_emitted(folder, monkeypatch, capsys, caplog):
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    _run(folder)
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out + captured.err + caplog.text
    assert FakeApi.instances[0].token == SENTINEL
    assert FakeApi.instances[0].calls == ["upload_large_folder"]


def test_failure_does_not_leak_token(folder, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    with pytest.raises(RuntimeError) as err:
        upload.main(["--dir", str(folder), "--repo-id", "boom/boom"])
    out = capsys.readouterr()
    assert SENTINEL not in str(err.value) + out.out + out.err


def test_missing_token_errors_clearly(folder, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="HF_TOKEN is not set"):
        _run(folder)
    assert FakeApi.instances == []


def test_no_token_cli_option(folder, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    with pytest.raises(SystemExit):
        _run(folder, "--token", "x")


def test_create_repo_only_with_flag(folder, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    _run(folder)
    assert "create_repo" not in FakeApi.instances[0].calls
    _run(folder, "--create")
    assert FakeApi.instances[1].calls == ["create_repo", "upload_large_folder"]


def test_refuses_truth_file(folder, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SENTINEL)
    (folder / upload.TRUTH_FILE).write_bytes(b"")
    with pytest.raises(ValueError, match="planted truth"):
        _run(folder)
    assert FakeApi.instances == []
