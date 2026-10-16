#!/usr/bin/env python3
"""Bounded real-filesystem fixture; does not claim CUDA or Trainer validation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("msw", os.environ["MSW_STORAGE_TOOL"])
msw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msw)
s = msw.Storage(os.environ["MSW_EVIDENCE_ROOT"])
oracle = Path(os.environ["MSW_READONLY_ORACLE"])
oracle_hash = msw.sha256(oracle)
s.refresh()
for backend in ("native", "shim"):
    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    for kind in ("gradient", "update"):
        name = "{}/rank0/step1/{}.npy".format(backend, kind)
        np.save(str(s.raw_path(name)), array)
        s.record(name, kind, list(array.shape), str(array.dtype))
for kind in ("gradient", "update"):
    np.testing.assert_array_equal(
        np.load(str(s.resolve("native/rank0/step1/{}.npy".format(kind)))),
        np.load(str(s.resolve("shim/rank0/step1/{}.npy".format(kind)))))
cp = s.work / "trainer" / "checkpoint-2"
cp.mkdir(parents=True)
np.save(str(cp / "model.npy"), np.arange(16, dtype=np.float32))
state = {"global_step": 2, "cursor": 2, "momentum": 0.5, "lr": 0.01, "rng": 123}
for name in ("optimizer.json", "scheduler.json", "trainer_state.json", "rng_state_0.json", "rng_state_1.json", "config.json", "tokenizer_config.json"):
    msw.atomic_json(cp / name, state)
published = s.publish("native-restore", cp)
result = subprocess.check_output([sys.executable, "-c", '''
import json, sys, numpy as np
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads((p / "trainer_state.json").read_text())
assert s["global_step"] == s["cursor"] == 2
assert json.loads((p / "optimizer.json").read_text())["momentum"] == 0.5
assert json.loads((p / "rng_state_1.json").read_text())["rng"] == 123
w = np.load(str(p / "model.npy"))
assert w.shape == (16,) and float(w.sum()) == 120.0
print("NEW_PROCESS_RESTORE_PASS")
''', str(published)]).decode().strip()
# A later stage crash must not erase the already-published native checkpoint.
failed = subprocess.run([sys.executable, "-c", "import os; os._exit(23)"])
assert failed.returncode == 23
s.verify_checkpoint("native-restore")
assert msw.sha256(oracle) == oracle_hash
assert Path(os.environ["TMPDIR"]).resolve() == s.work / "t"
msw.atomic_json(s.evidence / "chain-result.json", {
    "status": "PASS", "policy": msw.VERSION, "node": s.data["node"], "job": s.data["job"],
    "raw_elementwise_compare": "PASS", "checkpoint_publish_sha256": "PASS",
    "new_process_restore": result, "later_stage_crash_checkpoint_preserved": "PASS",
    "readonly_oracle_sha256": oracle_hash, "oracle": str(oracle),
    "tmp_real_path": str(Path(os.environ["TMPDIR"]).resolve()),
    "scope": "small fixture storage chain; not actual Trainer/CUDA convergence"})
print("STORAGE_CHAIN_PASS", result)
