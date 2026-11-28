"""Catch stale notebook bundles and missing dependencies before Kaggle submission."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_matches_current_sources_and_imports_in_isolation(tmp_path):
    builder_path = ROOT / "scripts/build_lesion_crop_notebook.py"
    spec = importlib.util.spec_from_file_location("build_crop_notebook", builder_path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    notebook = json.loads((ROOT / "notebooks/kaggle_btxrd_vit_lesion_crop_ab"
                           / "vit_lesion_crop_ab.ipynb").read_text(encoding="utf-8"))
    assert notebook == builder.build_notebook(), "Rebuild the notebook after editing source files"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
    tree = ast.parse("".join(notebook["cells"][3]["source"]))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "SOURCES"
                              for target in node.targets))
    sources = ast.literal_eval(assignment.value)
    assert "src/xla/vendor/segmenter/LICENSE" in sources
    for name, content in sources.items():
        assert content == (ROOT / name).read_text(encoding="utf-8")
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    environment = dict(os.environ, PYTHONPATH=str(tmp_path / "src"))
    script = ("import runpy; from xla.data.lesion_crop import LesionCropDataset; "
              "from xla.models.segmenter_vit import SegmenterViT; "
              "runpy.run_path('scripts/improve_segmenter.py', run_name='bundle_import_check')")
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=environment, check=True,
                   capture_output=True, text=True, timeout=90)
