from unittest.mock import patch, MagicMock
from serving import artifacts


def test_skips_when_no_repo():
    assert artifacts.download_artifacts(None) is False


def test_downloads_when_repo_given():
    fake = MagicMock()
    with patch("serving.artifacts.snapshot_download", fake) as dl:
        result = artifacts.download_artifacts("user/recsys-artifacts", token="t")
    assert result is True
    dl.assert_called_once()
    kwargs = dl.call_args.kwargs
    assert kwargs["repo_id"] == "user/recsys-artifacts"
    assert kwargs["repo_type"] == "model"
    assert kwargs["token"] == "t"


def test_returns_false_on_download_error():
    def boom(*a, **k):
        raise RuntimeError("network down")
    with patch("serving.artifacts.snapshot_download", boom):
        assert artifacts.download_artifacts("user/recsys-artifacts") is False
