"""Regression checks for checkpoint selection and simulator Python discovery."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("checkpoint_assets", ROOT / "checkpoint_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class CheckpointAssets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pi05 assets ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def norm(self, name):
        p = self.root / "assets" / name / "norm_stats.json"
        p.parent.mkdir(parents=True)
        p.write_text('{"norm_stats": {}}')
        return p

    def test_standard_checkpoint(self):
        p = self.norm("robosyn_robotwin_piper")
        self.assertEqual(assets.resolve_norm_stats(self.root, p.parent.name), p)

    def test_drawer_sf_checkpoint_without_renaming(self):
        p = self.norm("robosyn_robotwin_piper_sf")
        self.assertEqual(assets.resolve_norm_stats(self.root, "robosyn_robotwin_piper"), p)
        self.assertFalse((p.parent.parent / "robosyn_robotwin_piper").exists())

    def test_preferred_and_explicit_selection(self):
        preferred = self.norm("robosyn_robotwin_piper")
        sf = self.norm("robosyn_robotwin_piper_sf")
        self.assertEqual(assets.resolve_norm_stats(self.root, preferred.parent.name), preferred)
        self.assertEqual(assets.resolve_norm_stats(self.root, preferred.parent.name, sf.parent.name), sf)

    def test_missing_explicit_selection_never_falls_back(self):
        self.norm("robosyn_robotwin_piper")
        with self.assertRaises(FileNotFoundError):
            assets.resolve_norm_stats(self.root, "robosyn_robotwin_piper", "missing")

    def test_ambiguous_and_missing_assets(self):
        with self.assertRaises(FileNotFoundError):
            assets.resolve_norm_stats(self.root, "missing")
        self.norm("a")
        self.norm("b")
        with self.assertRaisesRegex(ValueError, "PI05_NORM_ASSET_ID"):
            assets.resolve_norm_stats(self.root, "missing")


class Launcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pi05 launcher ")
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        self.repo = self.ws / "RoboSynChallenge"
        self.policy = self.repo / "policy" / "pi05_v2"
        (self.policy / "src" / "openpi").mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        (self.repo / "scripts" / "eval_policy.py").touch()
        shutil.copyfile(ROOT / "eval.sh", self.policy / "eval.sh")
        self.checkpoint = self.ws / "checkpoints" / "drawer_open_place"
        (self.checkpoint / "params").mkdir(parents=True)
        norm = self.checkpoint / "assets" / "robosyn_robotwin_piper_sf" / "norm_stats.json"
        norm.parent.mkdir(parents=True)
        norm.write_text("{}")
        self.capture = self.ws / "capture.json"
        self.env = os.environ.copy()
        for k in ("PYTHON_BIN", "ROBOSYN_VENV_DIR", "VIRTUAL_ENV", "CONDA_PREFIX",
                  "ROBOSYN_ROOT", "OPENPI_ROOT", "EMBODICHAIN_ROOT", "PI05_PYTHON"):
            self.env.pop(k, None)
        self.env["CAPTURE"] = str(self.capture)
        self.env["PI05_PYTHON"] = str(self.fake(self.policy / ".venv" / "bin" / "python"))

    def fake(self, path, available=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!" + sys.executable + "\n" +
            "import sys,json,os\n"
            f"if sys.argv[1:2] == ['-c']: sys.exit({0 if available else 1})\n"
            "with open(os.environ['CAPTURE'],'w') as f:\n"
            " json.dump({'python':sys.argv[0],'args':sys.argv[1:],'cwd':os.getcwd(),"
            "'exit_process':os.environ.get('EMBODICHAIN_SIM_EXIT_PROCESS'),"
            "'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')},f)\n")
        path.chmod(0o755)
        return path

    def run_launcher(self):
        return subprocess.run(["bash", str(self.policy / "eval.sh"), "drawer_open_place",
            "random", str(self.checkpoint), "0", "--max_episodes", "20", "--headless", "true"],
            cwd=self.ws, env=self.env, capture_output=True, text=True)

    def check_selected(self, expected):
        r = self.run_launcher()
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(self.capture.read_text())
        self.assertEqual(data["python"], str(expected))
        self.assertEqual(data["cwd"], str(self.repo))
        self.assertIn(str(self.checkpoint), data["args"])
        self.assertIn("True", data["args"])
        self.assertEqual(data["exit_process"], "0")
        self.assertIsNone(data["cuda"])

    def test_official_embodichain_layout(self):
        self.check_selected(self.fake(self.ws / "EmbodiChain" / ".venv" / "bin" / "python"))

    def test_legacy_workspace_layout(self):
        self.check_selected(self.fake(self.ws / ".venv" / "bin" / "python"))

    def test_explicit_python_overrides_other_environments(self):
        self.fake(self.ws / "EmbodiChain" / ".venv" / "bin" / "python")
        p = self.fake(self.ws / "custom env" / "bin" / "python")
        self.env["PYTHON_BIN"] = str(p)
        self.env["ROBOSYN_VENV_DIR"] = str(self.ws / "nonexistent")
        self.check_selected(p)

    def test_explicit_venv_directory(self):
        p = self.fake(self.ws / "chosen venv" / "bin" / "python")
        self.env["ROBOSYN_VENV_DIR"] = str(p.parent.parent)
        self.check_selected(p)

    def test_active_virtualenv_and_conda(self):
        for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
            with self.subTest(key=key):
                self.fake(self.ws / "EmbodiChain" / ".venv" / "bin" / "python")
                p = self.fake(self.ws / key / "bin" / "python")
                self.env[key] = str(p.parent.parent)
                self.check_selected(p)
                self.env.pop(key)

    def test_unsuitable_active_env_is_skipped(self):
        p = self.fake(self.ws / "active" / "bin" / "python", available=False)
        self.env["VIRTUAL_ENV"] = str(p.parent.parent)
        self.check_selected(self.fake(self.ws / "EmbodiChain" / ".venv" / "bin" / "python"))

    def test_bad_explicit_python_does_not_fall_back(self):
        self.fake(self.ws / "EmbodiChain" / ".venv" / "bin" / "python")
        self.env["PYTHON_BIN"] = str(self.fake(self.ws / "wrong" / "bin" / "python", False))
        r = self.run_launcher()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("cannot locate", r.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
