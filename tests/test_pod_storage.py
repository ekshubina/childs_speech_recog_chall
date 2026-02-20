"""
Tests for scripts/pod_storage.py

Covers:
- load_env: file parsing, comment stripping, missing-key detection
- list_subdirs: normal, empty, exception
- list_files_shallow: normal, directory-marker skipping, exception
- walk_keys: recursive accumulation
- key_exists: found / not-found paths
- delete_keys: chunked batches, error reporting
- human_size: size formatting
- cmd_list: output formatting, parallel fetch, checkpoint detail
- cmd_get: checkpoint download, log download, fallback to head_object
- cmd_clean: plan accumulation, dry-run, confirmation prompt, execution

All tests run without any real AWS/RunPod credentials.
"""

import importlib.util
import os
import sys
import textwrap
import types
from argparse import Namespace
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ── Load the module from scripts/pod_storage.py ───────────────────────────────

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pod_storage.py"
_spec = importlib.util.spec_from_file_location("pod_storage", _SCRIPT)
pod_storage: types.ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pod_storage)  # type: ignore[union-attr]

# Shortcuts
load_env = pod_storage.load_env
list_subdirs = pod_storage.list_subdirs
list_files_shallow = pod_storage.list_files_shallow
walk_keys = pod_storage.walk_keys
key_exists = pod_storage.key_exists
delete_keys = pod_storage.delete_keys
human_size = pod_storage.human_size
cmd_list = pod_storage.cmd_list
cmd_get = pod_storage.cmd_get
cmd_clean = pod_storage.cmd_clean

# ── Helpers shared across tests ───────────────────────────────────────────────

REQUIRED_VARS = [
    "RUNPOD_S3_BUCKET",
    "RUNPOD_S3_ENDPOINT",
    "RUNPOD_S3_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
]

FULL_ENV = {
    "RUNPOD_S3_BUCKET": "mybucket",
    "RUNPOD_S3_ENDPOINT": "https://s3api-eu-ro-1.runpod.io",
    "RUNPOD_S3_REGION": "eu-ro-1",
    "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
    "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
}


def env_file_content(**overrides) -> str:
    vals = {**FULL_ENV, **overrides}
    return "\n".join(f"{k}={v}" for k, v in vals.items()) + "\n"


def make_s3(bucket: str = "mybucket") -> MagicMock:
    """Return a mock S3 client pre-wired with the bucket env var."""
    os.environ["RUNPOD_S3_BUCKET"] = bucket
    return MagicMock()


# ══════════════════════════════════════════════════════════════════════════════
# load_env
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestLoadEnv:
    def test_loads_all_required_vars(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".runpod.env"
        env_file.write_text(env_file_content())
        monkeypatch.setattr(pod_storage, "ENV_FILE", env_file)
        for k in REQUIRED_VARS:
            monkeypatch.delenv(k, raising=False)
        load_env()
        for k, v in FULL_ENV.items():
            assert os.environ[k] == v

    def test_strips_inline_comments(self, tmp_path, monkeypatch):
        content = (
            "RUNPOD_S3_BUCKET=mybucket  # the bucket name\n"
            "RUNPOD_S3_ENDPOINT=https://s3api-eu-ro-1.runpod.io\n"
            "RUNPOD_S3_REGION=eu-ro-1  # adjust region\n"
            "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
        )
        env_file = tmp_path / ".runpod.env"
        env_file.write_text(content)
        monkeypatch.setattr(pod_storage, "ENV_FILE", env_file)
        for k in REQUIRED_VARS:
            monkeypatch.delenv(k, raising=False)
        load_env()
        assert os.environ["RUNPOD_S3_REGION"] == "eu-ro-1"
        assert os.environ["RUNPOD_S3_BUCKET"] == "mybucket"

    def test_ignores_comment_lines_and_blank_lines(self, tmp_path, monkeypatch):
        content = (
            "# this is a comment\n"
            "\n"
            "RUNPOD_S3_BUCKET=mybucket\n"
            "RUNPOD_S3_ENDPOINT=https://s3api-eu-ro-1.runpod.io\n"
            "RUNPOD_S3_REGION=eu-ro-1\n"
            "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
        )
        env_file = tmp_path / ".runpod.env"
        env_file.write_text(content)
        monkeypatch.setattr(pod_storage, "ENV_FILE", env_file)
        for k in REQUIRED_VARS:
            monkeypatch.delenv(k, raising=False)
        load_env()  # Should not raise
        assert os.environ["RUNPOD_S3_BUCKET"] == "mybucket"

    def test_raises_when_env_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pod_storage, "ENV_FILE", tmp_path / "nonexistent.env")
        with pytest.raises(SystemExit):
            load_env()

    def test_raises_when_var_missing(self, tmp_path, monkeypatch):
        # Write env without AWS_SECRET_ACCESS_KEY
        content = (
            "RUNPOD_S3_BUCKET=mybucket\n"
            "RUNPOD_S3_ENDPOINT=https://s3api-eu-ro-1.runpod.io\n"
            "RUNPOD_S3_REGION=eu-ro-1\n"
            "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
        )
        env_file = tmp_path / ".runpod.env"
        env_file.write_text(content)
        monkeypatch.setattr(pod_storage, "ENV_FILE", env_file)
        for k in REQUIRED_VARS:
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(SystemExit, match="missing"):
            load_env()

    def test_does_not_overwrite_existing_env(self, tmp_path, monkeypatch):
        """setdefault must not overwrite an already-set OS env var."""
        env_file = tmp_path / ".runpod.env"
        env_file.write_text(env_file_content())
        monkeypatch.setattr(pod_storage, "ENV_FILE", env_file)
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "already-set")
        load_env()
        assert os.environ["RUNPOD_S3_BUCKET"] == "already-set"

    def test_strips_quoted_values(self, tmp_path, monkeypatch):
        content = (
            'RUNPOD_S3_BUCKET="mybucket"\n'
            "RUNPOD_S3_ENDPOINT=https://s3api-eu-ro-1.runpod.io\n"
            "RUNPOD_S3_REGION=eu-ro-1\n"
            "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
        )
        env_file = tmp_path / ".runpod.env"
        env_file.write_text(content)
        monkeypatch.setattr(pod_storage, "ENV_FILE", env_file)
        for k in REQUIRED_VARS:
            monkeypatch.delenv(k, raising=False)
        load_env()
        assert os.environ["RUNPOD_S3_BUCKET"] == "mybucket"


# ══════════════════════════════════════════════════════════════════════════════
# list_subdirs
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestListSubdirs:
    def _s3(self, common_prefixes):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"CommonPrefixes": [{"Prefix": p} for p in common_prefixes]}
        return s3

    def test_returns_leaf_names(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = self._s3(["childs/checkpoints/run-1/", "childs/checkpoints/run-2/"])
        result = list_subdirs(s3, "childs/checkpoints")
        assert result == ["run-1", "run-2"]

    def test_empty_response(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = self._s3([])
        assert list_subdirs(s3, "childs/checkpoints") == []

    def test_uses_delimiter_slash(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = self._s3([])
        list_subdirs(s3, "logs")
        kwargs = s3.list_objects_v2.call_args[1]
        assert kwargs["Delimiter"] == "/"

    def test_prefix_gets_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = self._s3([])
        list_subdirs(s3, "logs")
        kwargs = s3.list_objects_v2.call_args[1]
        assert kwargs["Prefix"].endswith("/")

    def test_exception_returns_empty(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        s3.list_objects_v2.side_effect = Exception("connection refused")
        result = list_subdirs(s3, "broken/prefix")
        assert result == []
        out = capsys.readouterr().err
        assert "Warning" in out


# ══════════════════════════════════════════════════════════════════════════════
# list_files_shallow
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestListFilesShallow:
    def _s3(self, contents):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"Contents": contents}
        return s3

    def test_returns_key_size_pairs(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = self._s3([
            {"Key": "logs/current.log", "Size": 1024},
            {"Key": "logs/run_train.sh", "Size": 512},
        ])
        result = list_files_shallow(s3, "logs")
        assert ("logs/current.log", 1024) in result
        assert ("logs/run_train.sh", 512) in result

    def test_skips_directory_marker_objects(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = self._s3([
            {"Key": "logs/", "Size": 0},  # dir marker
            {"Key": "logs/current.log", "Size": 100},
        ])
        result = list_files_shallow(s3, "logs")
        keys = [k for k, _ in result]
        assert "logs/" not in keys
        assert "logs/current.log" in keys

    def test_empty_response(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = self._s3([])
        assert list_files_shallow(s3, "logs") == []

    def test_exception_returns_empty(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        s3.list_objects_v2.side_effect = Exception("timeout")
        result = list_files_shallow(s3, "logs")
        assert result == []
        assert "Warning" in capsys.readouterr().err


# ══════════════════════════════════════════════════════════════════════════════
# walk_keys
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestWalkKeys:
    def _build_s3(self, tree: dict) -> MagicMock:
        """
        tree: { prefix: {"files": [(key, sz)], "subs": [name, ...]} }
        list_objects_v2 is dispatched based on Prefix kwarg.
        """
        s3 = MagicMock()

        def list_objects_v2(Bucket, Prefix, Delimiter, MaxKeys):  # noqa: N803
            prefix_key = Prefix.rstrip("/")
            node = tree.get(prefix_key, {})
            files = node.get("files", [])
            subs = node.get("subs", [])
            contents = [{"Key": k, "Size": sz} for k, sz in files]
            common = [{"Prefix": f"{prefix_key}/{s}/"} for s in subs]
            return {"Contents": contents, "CommonPrefixes": common}

        s3.list_objects_v2.side_effect = list_objects_v2
        return s3

    def test_flat_directory(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        tree = {
            "logs": {
                "files": [("logs/current.log", 1000), ("logs/run.sh", 200)],
                "subs": [],
            }
        }
        s3 = self._build_s3(tree)
        result = walk_keys(s3, "logs")
        assert sorted(result) == sorted([("logs/current.log", 1000), ("logs/run.sh", 200)])

    def test_nested_two_levels(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        tree = {
            "ckpts": {
                "files": [],
                "subs": ["run1"],
            },
            "ckpts/run1": {
                "files": [("ckpts/run1/config.json", 500)],
                "subs": ["checkpoint-100"],
            },
            "ckpts/run1/checkpoint-100": {
                "files": [("ckpts/run1/checkpoint-100/model.pt", 100_000)],
                "subs": [],
            },
        }
        s3 = self._build_s3(tree)
        result = walk_keys(s3, "ckpts")
        assert ("ckpts/run1/config.json", 500) in result
        assert ("ckpts/run1/checkpoint-100/model.pt", 100_000) in result
        assert len(result) == 2

    def test_empty_prefix(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        tree = {"empty": {"files": [], "subs": []}}
        s3 = self._build_s3(tree)
        assert walk_keys(s3, "empty") == []

    def test_include_dir_markers_appends_subdir_slash_keys(self, monkeypatch):
        """Regression: deleted dirs leave marker objects; include_dir_markers=True collects them."""
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        tree = {
            "ckpts": {
                "files": [],
                "subs": ["run1"],
            },
            "ckpts/run1": {
                "files": [("ckpts/run1/config.json", 100)],
                "subs": ["checkpoint-50"],
            },
            "ckpts/run1/checkpoint-50": {
                "files": [],  # already deleted — only marker remains
                "subs": [],
            },
        }
        s3 = self._build_s3(tree)
        result = walk_keys(s3, "ckpts", include_dir_markers=True)
        keys = [k for k, _ in result]
        # Real file
        assert "ckpts/run1/config.json" in keys
        # Dir marker keys appended for each subdir encountered
        assert "ckpts/run1/checkpoint-50/" in keys
        assert "ckpts/run1/" in keys

    def test_include_dir_markers_false_by_default(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        tree = {
            "ckpts": {"files": [], "subs": ["run1"]},
            "ckpts/run1": {"files": [("ckpts/run1/model.pt", 1)], "subs": []},
        }
        s3 = self._build_s3(tree)
        result = walk_keys(s3, "ckpts")
        keys = [k for k, _ in result]
        assert not any(k.endswith("/") for k in keys)


# ══════════════════════════════════════════════════════════════════════════════
# key_exists
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestKeyExists:
    def test_returns_size_when_found(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        s3.head_object.return_value = {"ContentLength": 4096}
        assert key_exists(s3, "logs/current.log") == 4096

    def test_returns_none_when_not_found(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        s3.head_object.side_effect = Exception("404 Not Found")
        assert key_exists(s3, "logs/missing.log") is None

    def test_calls_correct_bucket_and_key(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "testbucket")
        s3 = MagicMock()
        s3.head_object.return_value = {"ContentLength": 1}
        key_exists(s3, "some/key.json")
        s3.head_object.assert_called_once_with(Bucket="testbucket", Key="some/key.json")


# ══════════════════════════════════════════════════════════════════════════════
# delete_keys
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDeleteKeys:

    def test_deletes_each_key_individually(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        keys = ["a/1.txt", "a/2.txt"]
        deleted = delete_keys(s3, keys)
        assert s3.delete_object.call_count == 2
        assert deleted == 2

    def test_calls_correct_bucket_and_key(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        delete_keys(s3, ["a/1.txt"])
        s3.delete_object.assert_called_once_with(Bucket="mybucket", Key="a/1.txt")

    def test_large_set_calls_once_per_key(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        keys = [f"prefix/{i}" for i in range(2500)]
        delete_keys(s3, keys)
        assert s3.delete_object.call_count == 2500

    def test_reports_errors_to_stderr(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        s3.delete_object.side_effect = Exception("AccessDenied")
        delete_keys(s3, ["a/bad.txt"], label="test-label")
        err = capsys.readouterr().err
        assert "Delete error" in err
        assert "AccessDenied" in err

    def test_returns_count_excluding_errors(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()

        def side_effect(Bucket, Key):
            if Key == "a/bad.txt":
                raise Exception("Denied")

        s3.delete_object.side_effect = side_effect
        count = delete_keys(s3, ["a/1.txt", "a/2.txt", "a/bad.txt"])
        assert count == 2

    def test_handles_all_errors_gracefully(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        s3.delete_object.side_effect = Exception("S3 down")
        count = delete_keys(s3, ["a/1.txt"])
        assert count == 0
        assert "Delete error" in capsys.readouterr().err


# ══════════════════════════════════════════════════════════════════════════════
# human_size
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHumanSize:
    @pytest.mark.parametrize("b, expected", [
        (0, "0 B"),
        (999, "999 B"),
        (1_024, "1.0 KB"),
        (1_536, "1.5 KB"),
        (1_048_576, "1.0 MB"),
        (1_572_864, "1.5 MB"),
        (1_073_741_824, "1.0 GB"),
        (16_106_127_360, "15.0 GB"),
    ])
    def test_formatting(self, b, expected):
        assert human_size(b) == expected


# ══════════════════════════════════════════════════════════════════════════════
# cmd_list
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCmdList:
    def _make_s3(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        s3 = MagicMock()
        # walk_keys → list_files_shallow + list_subdirs calls
        # We patch walk_keys and list_subdirs at module level for simplicity
        return s3

    def test_prints_bucket_and_endpoint(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        s3 = MagicMock()
        with patch.object(pod_storage, "walk_keys", return_value=[]):
            with patch.object(pod_storage, "list_subdirs", return_value=[]):
                cmd_list(s3, Namespace())
        out = capsys.readouterr().out
        assert "mybucket" in out
        assert "https://s3api.example.com" in out

    def test_shows_nonempty_prefixes_only(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        s3 = MagicMock()

        def fake_walk(s3_, prefix):
            if prefix == "logs":
                return [("logs/current.log", 1024)]
            return []

        with patch.object(pod_storage, "walk_keys", side_effect=fake_walk):
            with patch.object(pod_storage, "list_subdirs", return_value=[]):
                cmd_list(s3, Namespace())

        out = capsys.readouterr().out
        assert "logs" in out
        # audio dirs are shown as a static "not listed" line — never fetched
        assert "not listed" in out
        # managed prefixes that are empty should not appear as sized rows
        assert ".cache/huggingface" not in out

    def test_prints_checkpoint_detail(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        s3 = MagicMock()

        def fake_list_subdirs(s3_, prefix):
            if "checkpoints" in prefix and "/" not in prefix.replace("childs_speech_recog_chall/checkpoints", ""):
                return ["my-run"]
            if "my-run" in prefix:
                return ["checkpoint-50", "final_model"]
            return []

        def fake_walk(s3_, prefix):
            return [("some/file", 1_000_000)]

        with patch.object(pod_storage, "walk_keys", side_effect=fake_walk):
            with patch.object(pod_storage, "list_subdirs", side_effect=fake_list_subdirs):
                cmd_list(s3, Namespace())

        out = capsys.readouterr().out
        assert "my-run" in out
        assert "checkpoint-50" in out
        assert "final_model" in out


# ══════════════════════════════════════════════════════════════════════════════
# cmd_get
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCmdGet:
    def _args(self, **kw):
        defaults = dict(logs_only=False, checkpoints_only=False, run=None)
        return Namespace(**{**defaults, **kw})

    def test_downloads_checkpoint_files(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        monkeypatch.setattr(pod_storage, "REPO_ROOT", tmp_path)

        s3 = MagicMock()
        objs = [
            ("childs_speech_recog_chall/checkpoints/run1/checkpoint-50/model.pt", 100),
            ("childs_speech_recog_chall/checkpoints/run1/final_model/config.json", 200),
        ]

        with patch.object(pod_storage, "list_subdirs", return_value=["run1"]):
            with patch.object(pod_storage, "walk_keys", return_value=objs):
                cmd_get(s3, self._args(checkpoints_only=True))

        # download_file called once per object
        assert s3.download_file.call_count == 2

    def test_downloads_to_correct_local_path(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        monkeypatch.setattr(pod_storage, "REPO_ROOT", tmp_path)

        s3 = MagicMock()
        objs = [("childs_speech_recog_chall/checkpoints/myrun/final_model/model.pt", 1)]

        with patch.object(pod_storage, "list_subdirs", return_value=["myrun"]):
            with patch.object(pod_storage, "walk_keys", return_value=objs):
                cmd_get(s3, self._args(checkpoints_only=True))

        dest_arg = s3.download_file.call_args[0][2]
        assert "checkpoints/myrun/final_model/model.pt" in dest_arg

    def test_downloads_logs_from_walk(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        monkeypatch.setattr(pod_storage, "REPO_ROOT", tmp_path)

        s3 = MagicMock()
        log_objs = [("logs/current.log", 1300000), ("logs/run_train.sh", 500)]

        with patch.object(pod_storage, "walk_keys", return_value=log_objs):
            cmd_get(s3, self._args(logs_only=True))

        assert s3.download_file.call_count == 2

    def test_logs_fallback_to_head_object_when_walk_empty(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        monkeypatch.setattr(pod_storage, "REPO_ROOT", tmp_path)

        s3 = MagicMock()

        def fake_key_exists(s3_, key):
            return 1024 if key == "logs/current.log" else None

        with patch.object(pod_storage, "walk_keys", return_value=[]):
            with patch.object(pod_storage, "key_exists", side_effect=fake_key_exists):
                cmd_get(s3, self._args(logs_only=True))

        # Should fall back and download current.log
        assert s3.download_file.call_count == 1
        assert "logs/current.log" in s3.download_file.call_args[0][1]

    def test_specific_run_scopes_download(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        monkeypatch.setattr(pod_storage, "REPO_ROOT", tmp_path)

        s3 = MagicMock()
        objs = [("childs_speech_recog_chall/checkpoints/targetrun/model.pt", 1)]

        with patch.object(pod_storage, "list_subdirs", return_value=["targetrun"]) as mock_sd:
            with patch.object(pod_storage, "walk_keys", return_value=objs):
                cmd_get(s3, self._args(checkpoints_only=True, run="targetrun"))
            # list_subdirs should NOT be called when --run is specified
            for c in mock_sd.call_args_list:
                assert "checkpoints" not in str(c)  # no run discovery

    def test_empty_checkpoint_run_prints_message(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        monkeypatch.setenv("RUNPOD_S3_ENDPOINT", "https://s3api.example.com")
        monkeypatch.setattr(pod_storage, "REPO_ROOT", tmp_path)

        s3 = MagicMock()
        with patch.object(pod_storage, "list_subdirs", return_value=["emptyrun"]):
            with patch.object(pod_storage, "walk_keys", return_value=[]):
                cmd_get(s3, self._args(checkpoints_only=True))

        assert s3.download_file.call_count == 0
        out = capsys.readouterr().out
        assert "empty" in out.lower()


# ══════════════════════════════════════════════════════════════════════════════
# cmd_clean
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCmdClean:
    def _args(self, **kw):
        defaults = dict(
            old_checkpoints=False,
            all_checkpoints=False,
            hf_cache=False,
            pycache=False,
            logs=False,
            run=None,
            yes=False,
        )
        return Namespace(**{**defaults, **kw})

    # ── Dry-run behaviour ─────────────────────────────────────────────────────

    def test_dry_run_does_not_delete(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()

        with patch.object(pod_storage, "list_subdirs", return_value=["run1"]):
            with patch.object(pod_storage, "walk_keys", return_value=[("k/f", 1)]):
                cmd_clean(s3, self._args(all_checkpoints=True, yes=False))

        s3.delete_object.assert_not_called()
        out = capsys.readouterr().out
        assert "Dry-run" in out

    def test_dry_run_with_nothing_to_delete(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        with patch.object(pod_storage, "list_subdirs", return_value=[]):
            with patch.object(pod_storage, "walk_keys", return_value=[]):
                cmd_clean(s3, self._args(all_checkpoints=True, yes=False))
        out = capsys.readouterr().out
        assert "nothing" in out.lower()

    # ── old-checkpoints ───────────────────────────────────────────────────────

    def test_old_checkpoints_keeps_latest_and_final(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()

        def fake_list_subdirs(s3_, prefix):
            if prefix.endswith("checkpoints"):
                return ["run1"]
            if prefix.endswith("run1"):
                return ["checkpoint-10", "checkpoint-50", "final_model"]
            return []

        to_delete = [("run1/checkpoint-10/model.pt", 1)]

        def fake_walk(s3_, prefix, include_dir_markers=False):
            if "checkpoint-10" in prefix:
                return to_delete
            return []

        with patch.object(pod_storage, "list_subdirs", side_effect=fake_list_subdirs):
            with patch.object(pod_storage, "walk_keys", side_effect=fake_walk):
                cmd_clean(s3, self._args(old_checkpoints=True, yes=False))

        out = capsys.readouterr().out
        assert "Keeping" in out and "final_model" in out
        assert "Keeping" in out and "checkpoint-50" in out
        assert "Will delete" in out and "checkpoint-10" in out

    def test_old_checkpoints_all_intermediate_collected(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()

        def fake_subdirs(s3_, prefix):
            if prefix.endswith("checkpoints"):
                return ["run1"]
            if prefix.endswith("run1"):
                return ["checkpoint-10", "checkpoint-20", "checkpoint-50"]
            return []

        def fake_walk(s3_, prefix, include_dir_markers=False):
            return [("k/f", 10)]

        plan_keys = []

        def fake_delete(s3_, keys, label=""):
            plan_keys.extend(keys)
            print(f"  Deleted {len(keys)} objects from {label}.")

        with patch.object(pod_storage, "list_subdirs", side_effect=fake_subdirs):
            with patch.object(pod_storage, "walk_keys", side_effect=fake_walk):
                with patch.object(pod_storage, "delete_keys", side_effect=fake_delete):
                    with patch("builtins.input", return_value="y"):
                        cmd_clean(s3, self._args(old_checkpoints=True, yes=True))

        # checkpoint-10 and checkpoint-20 are deleted; checkpoint-50 (latest) kept.
        # Each plan entry holds 1 real file + 1 dir marker → 4 total keys across 2 calls.
        real_keys = [k for k in plan_keys if not k.endswith("/")]
        assert len(real_keys) == 2

    # ── all-checkpoints ───────────────────────────────────────────────────────

    def test_all_checkpoints_collects_entire_run(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        all_objs = [("childs/checkpoints/run1/final_model/config.json", 10), ("childs/checkpoints/run1/checkpoint-50/model.pt", 1000)]

        with patch.object(pod_storage, "list_subdirs", return_value=["run1"]):
            with patch.object(pod_storage, "walk_keys", return_value=all_objs):
                with patch.object(pod_storage, "delete_keys", return_value=3) as mock_del:
                    with patch("builtins.input", return_value="y"):
                        cmd_clean(s3, self._args(all_checkpoints=True, yes=True))

        mock_del.assert_called_once()
        deleted_keys = mock_del.call_args[0][1]
        # 2 real files + 1 run-level directory marker
        assert len(deleted_keys) == 3
        real_keys = [k for k in deleted_keys if not k.endswith("/")]
        assert len(real_keys) == 2

    # ── hf-cache ──────────────────────────────────────────────────────────────

    def test_hf_cache_collects_all_objs(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        hf_objs = [(f".cache/huggingface/models--openai--whisper-small/blobs/{i}", 10_000_000) for i in range(5)]

        with patch.object(pod_storage, "walk_keys", return_value=hf_objs):
            with patch.object(pod_storage, "delete_keys", return_value=5) as mock_del:
                with patch("builtins.input", return_value="y"):
                    cmd_clean(s3, self._args(hf_cache=True, yes=True))

        mock_del.assert_called_once()
        assert len(mock_del.call_args[0][1]) == 5

    def test_hf_cache_empty_prints_message(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        with patch.object(pod_storage, "walk_keys", return_value=[]):
            cmd_clean(s3, self._args(hf_cache=True, yes=True))
        out = capsys.readouterr().out
        assert "empty" in out.lower() or "nothing" in out.lower()

    # ── pycache ───────────────────────────────────────────────────────────────

    def test_pycache_filters_pyc_and_pycache(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        all_objs = [
            ("repo/src/__pycache__/foo.cpython-311.pyc", 1024),
            ("repo/src/models.py", 5000),
            ("repo/src/__pycache__/bar.cpython-311.pyc", 512),
        ]
        with patch.object(pod_storage, "walk_keys", return_value=all_objs):
            with patch.object(pod_storage, "delete_keys", return_value=2) as mock_del:
                with patch("builtins.input", return_value="y"):
                    cmd_clean(s3, self._args(pycache=True, yes=True))

        mock_del.assert_called_once()
        keys = mock_del.call_args[0][1]
        assert len(keys) == 2
        assert all("__pycache__" in k or k.endswith(".pyc") for k in keys)

    # ── logs ──────────────────────────────────────────────────────────────────

    def test_logs_keeps_current_log(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        log_objs = [
            ("logs/current.log", 1_300_000),
            ("logs/run_train.sh", 200),
        ]
        with patch.object(pod_storage, "walk_keys", return_value=log_objs):
            with patch.object(pod_storage, "delete_keys", return_value=1) as mock_del:
                with patch("builtins.input", return_value="y"):
                    cmd_clean(s3, self._args(logs=True, yes=True))

        mock_del.assert_called_once()
        keys = mock_del.call_args[0][1]
        assert "logs/current.log" not in keys
        assert "logs/run_train.sh" in keys

    def test_logs_only_current_log_prints_nothing_to_delete(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        with patch.object(pod_storage, "walk_keys", return_value=[("logs/current.log", 100)]):
            cmd_clean(s3, self._args(logs=True, yes=True))
        out = capsys.readouterr().out
        assert "nothing" in out.lower() or "Only" in out

    # ── Confirmation prompt ───────────────────────────────────────────────────

    def test_abort_on_non_y_input(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        with patch.object(pod_storage, "list_subdirs", return_value=["run1"]):
            with patch.object(pod_storage, "walk_keys", return_value=[("k", 1)]):
                with patch("builtins.input", return_value="n"):
                    cmd_clean(s3, self._args(all_checkpoints=True, yes=True))
        s3.delete_object.assert_not_called()
        out = capsys.readouterr().out
        assert "Aborted" in out

    def test_abort_on_empty_input(self, monkeypatch, capsys):
        monkeypatch.setenv("RUNPOD_S3_BUCKET", "mybucket")
        s3 = MagicMock()
        with patch.object(pod_storage, "list_subdirs", return_value=["run1"]):
            with patch.object(pod_storage, "walk_keys", return_value=[("k", 1)]):
                with patch("builtins.input", return_value=""):
                    cmd_clean(s3, self._args(all_checkpoints=True, yes=True))
        s3.delete_object.assert_not_called()
