"""Offline single-GPU reload check for an existing Qwen2 seq-cls checkpoint."""
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CHECKPOINT = Path(os.environ["CUDA06_SEQCLS_CHECKPOINT"])
SAMPLE = Path(os.environ["CUDA06_SEQCLS_SAMPLE"])
ARGS = CHECKPOINT / "args.json"
RECEIPT = Path(os.environ["CUDA06_SEQCLS_RECEIPT"])
EXPECTED = {
    "model.safetensors": os.environ["CUDA06_SEQCLS_MODEL_SHA256"],
    "config.json": os.environ["CUDA06_SEQCLS_CONFIG_SHA256"],
    "tokenizer.json": os.environ["CUDA06_SEQCLS_TOKENIZER_SHA256"],
    "args.json": os.environ["CUDA06_SEQCLS_ARGS_SHA256"],
    "sample": os.environ["CUDA06_SEQCLS_SAMPLE_SHA256"],
    "official_reference": os.environ["CUDA06_SEQCLS_OFFICIAL_SHA256"],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_seq_cls_checkpoint_reload_head_and_finite_logits():
    config = json.loads((CHECKPOINT / "config.json").read_text())
    args = json.loads(ARGS.read_text())
    records = [json.loads(line) for line in SAMPLE.read_text().splitlines() if line.strip()]
    assert config["architectures"] == ["Qwen2ForSequenceClassification"]
    assert config["problem_type"] == "single_label_classification"
    assert args["num_labels"] == 2
    assert args["task_type"] == "seq_cls"
    assert args["max_length"] == 64
    assert len(records) == 8 and {r["label"] for r in records} == {0, 1}
    assert all(isinstance(r.get("query"), str) and isinstance(r.get("response"), str) for r in records)

    hashes = {
        "model.safetensors": sha256(CHECKPOINT / "model.safetensors"),
        "config.json": sha256(CHECKPOINT / "config.json"),
        "tokenizer.json": sha256(CHECKPOINT / "tokenizer.json"),
        "args.json": sha256(ARGS),
        "sample": sha256(SAMPLE),
        "official_reference": sha256(os.environ["CUDA06_SEQCLS_OFFICIAL_REFERENCE"]),
    }
    assert hashes == EXPECTED, hashes
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, local_files_only=True, trust_remote_code=False, attn_implementation="eager"
    ).cuda().eval()
    assert model.config.architectures == ["Qwen2ForSequenceClassification"]
    assert model.config.num_labels == 2
    assert model.score.weight.shape[0] == 2
    texts = [f"{r['query']}\n{r['response']}" for r in records]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = model(**inputs).logits
    assert tuple(logits.shape) == (8, 2)
    assert torch.isfinite(logits).all().item()

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps({
        "status": "passed", "scope": "single_gpu_seq_cls_checkpoint_reload_head_finite_logits",
        "checkpoint": str(CHECKPOINT), "sample": str(SAMPLE), "hashes": hashes,
        "model_class": type(model).__name__, "head_shape": list(model.score.weight.shape),
        "saved_training_args": {"num_labels": args["num_labels"], "task_type": args["task_type"],
                                "max_length": args["max_length"]},
        "logits_shape": list(logits.shape), "logits_finite": True,
        "cuda_device": torch.cuda.get_device_name(0), "labels": sorted({r["label"] for r in records}),
        "coverage_limit": "does not test training or historical two-rank NCCL behavior",
    }, ensure_ascii=False, indent=2) + "\n")
