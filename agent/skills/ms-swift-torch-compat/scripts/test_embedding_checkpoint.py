"""Offline CUDA embedding inference check for the existing Qwen2 checkpoint-3."""
import hashlib
import json
import os
from pathlib import Path
import traceback

CHECKPOINT = Path(os.environ["CUDA06_EMBED_CHECKPOINT"])
SAMPLE = Path(os.environ["CUDA06_EMBED_SAMPLE"])
RECEIPT = Path(os.environ["CUDA06_EMBED_RECEIPT"])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_existing_embedding_checkpoint_public_inference():
    expected = {
        "model.safetensors": os.environ["CUDA06_EMBED_MODEL_SHA256"],
        "config.json": os.environ["CUDA06_EMBED_CONFIG_SHA256"],
        "tokenizer.json": os.environ["CUDA06_EMBED_TOKENIZER_SHA256"],
        "args.json": os.environ["CUDA06_EMBED_ARGS_SHA256"],
        "sample": os.environ["CUDA06_EMBED_SAMPLE_SHA256"],
        "official_test": os.environ["CUDA06_EMBED_OFFICIAL_TEST_SHA256"],
        "official_infer_example": os.environ["CUDA06_EMBED_OFFICIAL_INFER_SHA256"],
    }
    args_path = CHECKPOINT / "args.json"
    receipt = {
        "status": "running", "scope": "existing_qwen2_embedding_checkpoint_public_cuda_inference",
        "checkpoint": str(CHECKPOINT), "sample": str(SAMPLE),
    }
    try:
        assert CHECKPOINT.is_dir() and SAMPLE.is_file()
        saved_args = json.loads(args_path.read_text())
        assert saved_args["task_type"] == "embedding"
        assert saved_args["loss_type"] == "infonce"
        assert saved_args["max_length"] == 64
        assert saved_args["tuner_type"] == "full"
        rows = [json.loads(line) for line in SAMPLE.read_text().splitlines() if line.strip()]
        assert len(rows) == 4 and all(isinstance(row.get("messages"), list) and row["messages"] for row in rows)

        hashes = {
            "model.safetensors": sha256(CHECKPOINT / "model.safetensors"),
            "config.json": sha256(CHECKPOINT / "config.json"),
            "tokenizer.json": sha256(CHECKPOINT / "tokenizer.json"),
            "args.json": sha256(args_path),
            "sample": sha256(SAMPLE),
            "official_test": sha256(os.environ["CUDA06_EMBED_OFFICIAL_TEST"]),
            "official_infer_example": sha256(os.environ["CUDA06_EMBED_OFFICIAL_INFER_EXAMPLE"]),
        }
        assert hashes == expected, hashes
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        assert os.environ.get("MODELSCOPE_OFFLINE") == "1"

        import torch
        from swift.infer_engine import InferRequest, TransformersEngine

        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        engine = TransformersEngine(
            str(CHECKPOINT), task_type="embedding", attn_impl="eager", max_batch_size=1,
        )
        assert next(engine.model.parameters()).device.type == "cuda"
        requests = [InferRequest(messages=row["messages"]) for row in rows]
        responses = engine.infer(requests)
        assert len(responses) == len(rows)
        vectors = torch.tensor([response.data[0].embedding for response in responses], dtype=torch.float32)
        assert vectors.ndim == 2 and vectors.shape[0] == len(rows) and vectors.shape[1] > 0
        assert torch.isfinite(vectors).all().item()
        similarity = torch.nn.functional.normalize(vectors, dim=-1) @ torch.nn.functional.normalize(vectors, dim=-1).T
        assert tuple(similarity.shape) == (len(rows), len(rows))
        assert torch.isfinite(similarity).all().item()
        receipt.update({
            "status": "passed", "hashes": hashes,
            "saved_training_args": {k: saved_args[k] for k in ("task_type", "loss_type", "max_length", "tuner_type")},
            "model_class": type(engine.model).__name__, "cuda_device": torch.cuda.get_device_name(0),
            "embedding_shape": list(vectors.shape), "all_embedding_values_finite": True,
            "similarity_shape": list(similarity.shape), "all_similarity_values_finite": True,
            "coverage_limit": "inference reload only; no training, convergence, or semantic-quality claim",
        })
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt["job_id"] = os.environ.get("SLURM_JOB_ID")
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
