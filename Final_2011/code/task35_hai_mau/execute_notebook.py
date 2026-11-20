"""Thực thi notebook từ đầu đến cuối và lưu output ngay trong file."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    module_dir = Path(__file__).resolve().parent
    runtime_dir = module_dir / ".notebook_runtime"
    runtime_dir.mkdir(exist_ok=True)
    # Tránh ghi vào C:\\Users\\<user>\\.ipython/.jupyter trong môi trường chấm bị giới hạn.
    os.environ["IPYTHONDIR"] = str(runtime_dir / "ipython")
    os.environ["JUPYTER_CONFIG_DIR"] = str(runtime_dir / "jupyter")
    os.environ["JUPYTER_DATA_DIR"] = str(runtime_dir / "data")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_dir / "runtime")

    import nbformat
    from nbclient import NotebookClient

    notebook_path = module_dir / "two_sample_analysis.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(module_dir)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    print(f"Executed and saved: {notebook_path}")


if __name__ == "__main__":
    main()

