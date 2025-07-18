import subprocess
from pathlib import Path


def test_simple_subprocess():
    """シンプルなsubprocessテスト"""
    project_root = Path(__file__).parent.parent

    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "kairei",
        "--",
        "chat",
        "--once",
        "--message",
        "Hello",
    ]

    result = subprocess.run(
        cmd, cwd=project_root, capture_output=True, text=True, timeout=10
    )

    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")

    assert result.returncode == 0
    assert "Echo: Hello" in result.stdout


if __name__ == "__main__":
    test_simple_subprocess()
