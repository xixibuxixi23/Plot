import importlib.util
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).parents[1] / "scripts" / "capture_run_manifest.py"
SPEC = importlib.util.spec_from_file_location("capture_run_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sha256(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"PLOT\n")
    assert MODULE.sha256(payload) == (
        "5ed8b5b5344b62dc920c8760ed5913809d44b10ae9670c155f06ef9ca2615736"
    )


def test_dirty_repo_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "tracked.txt").write_text("clean\n")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    (repo / "tracked.txt").write_text("dirty\n")
    command = tmp_path / "command.txt"
    command.write_text("python train.py\n")
    result = subprocess.run(
        [
            "python",
            str(SCRIPT),
            "--output-dir",
            str(tmp_path / "output"),
            "--model",
            "m1",
            "--run-id",
            "test",
            "--dataset-repo",
            "example/data",
            "--dataset-revision",
            "abc123",
            "--command-file",
            str(command),
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "dirty source tree" in result.stderr
