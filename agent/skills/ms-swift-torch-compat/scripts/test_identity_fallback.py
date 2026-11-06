"""Supplement seq-cls/embedding checkpoint inference with runtime identity and fallback counts."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

REPO = Path(os.environ["CUDA06_IDENTITY_REPO"])
SITE = Path(os.environ["CUDA06_IDENTITY_SITE"])
SEQ_CHECKPOINT = Path(os.environ["CUDA06_IDENTITY_SEQ_CHECKPOINT"])
SEQ_SAMPLE = Path(os.environ["CUDA06_IDENTITY_SEQ_SAMPLE"])
EMBED_CHECKPOINT = Path(os.environ["CUDA06_IDENTITY_EMBED_CHECKPOINT"])
EMBED_SAMPLE = Path(os.environ["CUDA06_IDENTITY_EMBED_SAMPLE"])
RECEIPT = Path(os.environ["CUDA06_IDENTITY_RECEIPT"])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_runtime_identities_and_zero_backend_fallback_for_both_cases():
    receipt = {"status": "running", "job_id": os.environ.get("SLURM_JOB_ID")}
    try:
        assert os.environ.get("backend_fallback") == "error"
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        frozen = {
            str(SEQ_CHECKPOINT / "model.safetensors"): "a8f49579b29a30d2d4440f07aee285cc072e41def3d6d34cd527177ef69f8040",
            str(SEQ_CHECKPOINT / "config.json"): "cb2a68533176bc3376815a3ede24b9edaf2c127f2242779c744870665dc6212f",
            str(SEQ_CHECKPOINT / "tokenizer.json"): "bcfe42da0a4497e8b2b172c1f9f4ec423a46dc12907f4349c55025f670422ba9",
            str(SEQ_CHECKPOINT / "args.json"): "af6b58bf1445b319494d293ba3cabd50830f7ec14ccdcfebb44a69d62fcef509",
            str(SEQ_SAMPLE): "6b4c59a4db1d485ed940882326a83d31c26273c9e193a1e81d35dda6d5bd945c",
            str(EMBED_CHECKPOINT / "model.safetensors"): "30f06b97ae7ff55cc4bcf54e5e96b209ef3e8b7a01b7c2162455fc059375f39c",
            str(EMBED_CHECKPOINT / "config.json"): "ec4375cbe65456f6ea3537993287376ca3169c856d0b5353fb3e8eb0be0ad307",
            str(EMBED_CHECKPOINT / "tokenizer.json"): "bcfe42da0a4497e8b2b172c1f9f4ec423a46dc12907f4349c55025f670422ba9",
            str(EMBED_CHECKPOINT / "args.json"): "35981ddd08a11b71baa4e4382e058cafc81c69d4e92041784c95ec5cdaa39530",
            str(EMBED_SAMPLE): "6b69fd53a87c54615855f0eef31179855c0ea7eb1a99e38b9213108d6cf39d92",
        }
        assert {p: sha256(p) for p in frozen} == frozen
        sys.path.insert(0, str(REPO))
        import jittor as jt
        import torch
        import swift

        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert Path(torch.__file__).resolve().is_relative_to(SITE.resolve())
        assert Path(jt.__file__).resolve().is_relative_to(SITE.resolve())
        assert Path(swift.__file__).resolve().is_relative_to(REPO.resolve())
        assert importlib.metadata.version("jittor") == "2.0.2"
        assert importlib.metadata.version("jittor-compat") == "2.0.2"
        assert importlib.metadata.version("jittor-ms-swift") == "2.0.0.dev1"

        identity_urls = {
            "core": json.loads((SITE / "jittor-2.0.2.dist-info/direct_url.json").read_text()),
            "compat": json.loads((SITE / "jittor_compat-2.0.2.dist-info/direct_url.json").read_text()),
            "swift_helper": json.loads((SITE / "jittor_ms_swift-2.0.0.dev1.dist-info/direct_url.json").read_text()),
        }
        core_commit = identity_urls["core"]["vcs_info"]["commit_id"]
        compat_commit = identity_urls["compat"]["vcs_info"]["commit_id"]
        helper_commit = identity_urls["swift_helper"]["vcs_info"]["commit_id"]
        assert core_commit == compat_commit == "507742d6c03fcad9a8609d078811dc5c4f7eea00"
        assert helper_commit == "fc5bc2c2d4a07267b0a91e5c3272dbca7388e2fe"
        swift_checkout = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
        ).strip()
        assert swift_checkout == os.environ["CUDA06_EXPECTED_SWIFT_CHECKOUT_COMMIT"]
        assert swift_checkout == "88d727951203256baa564c643c651b6f8d90fd7e"

        fallback_before = int(jt.core.backend_fallback_count())
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        seq_rows = [json.loads(line) for line in SEQ_SAMPLE.read_text().splitlines() if line.strip()]
        tokenizer = AutoTokenizer.from_pretrained(SEQ_CHECKPOINT, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            SEQ_CHECKPOINT, local_files_only=True, trust_remote_code=False, attn_implementation="eager"
        ).cuda().eval()
        seq_inputs = tokenizer(
            [f"{r['query']}\n{r['response']}" for r in seq_rows],
            padding=True, truncation=True, max_length=64, return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            logits = model(**seq_inputs).logits
        assert logits.shape == (8, 2) and torch.isfinite(logits).all().item()
        assert next(model.parameters()).device.type == "cuda"
        fallback_after_seq = int(jt.core.backend_fallback_count())
        del model, tokenizer, seq_inputs, logits
        torch.cuda.empty_cache()

        from swift.infer_engine import InferRequest, TransformersEngine
        embed_rows = [json.loads(line) for line in EMBED_SAMPLE.read_text().splitlines() if line.strip()]
        engine = TransformersEngine(
            str(EMBED_CHECKPOINT), task_type="embedding", attn_impl="eager", max_batch_size=1,
        )
        assert next(engine.model.parameters()).device.type == "cuda"
        responses = engine.infer([InferRequest(messages=row["messages"]) for row in embed_rows])
        embeddings = torch.tensor([response.data[0].embedding for response in responses], dtype=torch.float32)
        assert embeddings.shape == (4, 896) and torch.isfinite(embeddings).all().item()
        fallback_after_embed = int(jt.core.backend_fallback_count())
        assert fallback_before == fallback_after_seq == fallback_after_embed == 0, {
            "before": fallback_before, "after_seq_cls": fallback_after_seq,
            "after_embedding": fallback_after_embed,
        }
        receipt.update({
            "status": "passed", "jittor_core_version": importlib.metadata.version("jittor"),
            "jittor_core_commit": core_commit, "compat_version": importlib.metadata.version("jittor-compat"),
            "compat_commit": compat_commit, "swift_checkout_commit": swift_checkout,
            "swift_module_path": str(Path(swift.__file__).resolve()),
            "installed_swift_helper_version": importlib.metadata.version("jittor-ms-swift"),
            "installed_swift_helper_commit": helper_commit, "torch_shim_path": str(Path(torch.__file__).resolve()),
            "fallback_policy": "error", "backend_fallback_count": {
                "before": fallback_before, "after_seq_cls": fallback_after_seq,
                "after_embedding": fallback_after_embed,
            },
            "seq_cls_logits_shape": [8, 2], "embedding_shape": [4, 896],
            "downloaded_assets": False, "training_performed": False,
        })
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
