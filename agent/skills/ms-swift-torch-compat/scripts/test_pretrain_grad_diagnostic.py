"""One-step diagnostic of pre-clip gradients and the reported clip norm."""
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

from runtime_context import runtime

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ['CUDA06_BASE_MODEL'])
DATASET = Path(os.environ['CUDA06_DATASET'])
OUT = Path(os.environ['CUDA06_OUTPUT_DIR'])
RECEIPT = Path(os.environ['CUDA06_RECEIPT'])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def test_pretrain_grad_diagnostic():
    from runtime_context import runtime
    rec = {'status': 'running', 'job_id': os.getenv('SLURM_JOB_ID'),
           'model_sha256': sha(MODEL / 'model.safetensors'),
           'fixture_sha256': sha(DATASET), 'observations': []}
    try:
        assert rec['model_sha256'] == os.environ['CUDA06_EXPECTED_MODEL_SHA256']
        assert rec['fixture_sha256'] == os.environ['CUDA06_EXPECTED_DATASET_SHA256']
        assert not OUT.exists()
        with runtime('jittor', 'cuda', 'ms-swift', '4.5.2') as (torch, jt, identity):
            rec['runtime_identity'] = identity
            import numpy as np
            from accelerate import Accelerator
            original = Accelerator.clip_grad_norm_

            def measured(self, parameters, *args, **kwargs):
                params = list(parameters)
                sq_sum = 0.0
                finite_count = 0
                nonfinite = []
                grad_count = 0
                for i, param in enumerate(params):
                    grad = param.grad
                    if grad is None:
                        continue
                    grad_count += 1
                    array = np.asarray(grad.numpy())
                    is_finite = bool(np.isfinite(array).all())
                    if is_finite:
                        finite_count += 1
                        values = array.astype(np.float64, copy=False)
                        sq_sum += float(np.dot(values.ravel(), values.ravel()))
                    else:
                        nonfinite.append(i)
                independent_norm = math.sqrt(sq_sum) if not nonfinite else 'NaN'
                returned = original(self, params, *args, **kwargs)
                try:
                    returned_value = float(returned.item())
                except AttributeError:
                    returned_value = float(returned)
                rec['observations'].append({
                    'gradient_tensors': grad_count,
                    'finite_gradient_tensors': finite_count,
                    'nonfinite_gradient_parameter_indices': nonfinite,
                    'preclip_independent_fp64_l2_norm': independent_norm,
                    'clip_grad_norm_return': returned_value if math.isfinite(returned_value) else 'NaN',
                    'post_clip_non_null_gradient_tensors': sum(p.grad is not None for p in params),
                })
                return returned

            Accelerator.clip_grad_norm_ = measured
            try:
                from swift import PretrainArguments, pretrain_main
                result = pretrain_main(PretrainArguments(
                    model=str(MODEL), model_type='qwen2', dataset=[str(DATASET)],
                    max_steps=1, per_device_train_batch_size=1,
                    gradient_accumulation_steps=1, save_steps=1,
                    split_dataset_ratio=0.01, tuner_type='lora',
                    logging_steps=1, torch_dtype='float32', report_to='none',
                    output_dir=str(OUT), dataset_num_proc=1))
            finally:
                Accelerator.clip_grad_norm_ = original
            checkpoint = Path(result['last_model_checkpoint'])
            rec['checkpoint'] = str(checkpoint)
            rec['checkpoint_sha256'] = sha(checkpoint / 'adapter_model.safetensors')
            from safetensors.numpy import load_file
            tensors = load_file(str(checkpoint / 'adapter_model.safetensors'))
            b_tensors = {k: v for k, v in tensors.items() if 'lora_B' in k}
            rec['lora_b'] = {
                'tensor_count': len(b_tensors),
                'finite_tensors': sum(bool(np.isfinite(v).all()) for v in b_tensors.values()),
                'nonzero_tensors': sum(bool(np.count_nonzero(v)) for v in b_tensors.values()),
                'total_nonzero_elements': sum(int(np.count_nonzero(v)) for v in b_tensors.values()),
            }
            rec['fallback_before'] = identity.get('fallback_before')
            rec['fallback_after'] = identity.get('fallback_after')
            rec['status'] = 'passed'
    except BaseException as exc:
        rec.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                   traceback=traceback.format_exc())
        raise
    finally:
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(rec, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
