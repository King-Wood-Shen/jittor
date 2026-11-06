"""Compare CUDA-17 FP32 AMP-preservation patch against CUDA-06 pretrain NaNs."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np

MODEL = Path(os.environ['CUDA06_BASE_MODEL'])
DATASET = Path(os.environ['CUDA06_DATASET'])
OUT = Path(os.environ['CUDA06_OUTPUT_DIR'])
RECEIPT = Path(os.environ['CUDA06_RECEIPT'])
OVERLAY_MANIFEST = Path(os.environ['CUDA06_AMP_OVERLAY_MANIFEST'])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def direct_url(name):
    dist = importlib.metadata.distribution(name)
    raw = dist.read_text('direct_url.json')
    return {'version': dist.version, 'direct_url': json.loads(raw) if raw else None}


def test_pretrain_grad_diagnostic_with_cuda17_amp_fix():
    rec = {'status': 'running', 'job_id': os.getenv('SLURM_JOB_ID'),
           'model_sha256': sha(MODEL / 'model.safetensors'),
           'fixture_sha256': sha(DATASET), 'observations': []}
    try:
        assert rec['model_sha256'] == os.environ['CUDA06_EXPECTED_MODEL_SHA256']
        assert rec['fixture_sha256'] == os.environ['CUDA06_EXPECTED_DATASET_SHA256']
        assert not OUT.exists()
        rec['amp_overlay'] = json.loads(OVERLAY_MANIFEST.read_text())
        train_fp16 = os.environ.get('CUDA06_TRAIN_FP16', 'true').lower() == 'true'
        rec['training_options'] = {'fp16': train_fp16, 'bf16': False, 'torch_dtype': 'float32'}
        import jittor as jt
        import torch
        assert issubclass(torch.Tensor, jt.Var)
        assert Path(torch.__file__).resolve().is_relative_to(Path(os.environ['CUDA06_DIAGNOSTIC_SITE']).resolve())
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        rec['runtime'] = {
            'jittor_version': jt.__version__, 'torch_version': torch.__version__,
            'torch_file': torch.__file__, 'hardware': torch.cuda.get_device_name(0),
            'ms_swift': direct_url('ms-swift'),
            'jittor_ms_swift': direct_url('jittor-ms-swift'),
            'core_direct_url': json.loads((Path(os.environ['CUDA06_DIAGNOSTIC_SITE']) / 'jittor-2.0.2.dist-info/direct_url.json').read_text()),
            'compat_direct_url': json.loads((Path(os.environ['CUDA06_DIAGNOSTIC_SITE']) / 'jittor_compat-2.0.2.dist-info/direct_url.json').read_text()),
        }
        with jt.runtime.scope(use_cuda=1, backend_fallback='error'):
            before = int(jt.core.backend_fallback_count())
            rec['runtime']['fallback_before'] = before
            from accelerate import Accelerator
            original = Accelerator.clip_grad_norm_

            def measured(self, parameters, *args, **kwargs):
                params = list(parameters)
                name_by_id = {}
                for model in getattr(self, '_models', []):
                    try:
                        name_by_id.update({id(p): n for n, p in model.named_parameters()})
                    except Exception:
                        pass
                finite_count = 0
                nonfinite = []
                sq_sum = 0.0
                grad_count = 0
                for index, param in enumerate(params):
                    grad = param.grad
                    if grad is None:
                        continue
                    grad_count += 1
                    arr = np.asarray(grad.numpy())
                    finite = bool(np.isfinite(arr).all())
                    name = name_by_id.get(id(param), f'<unmatched:{index}>')
                    if finite:
                        finite_count += 1
                        values = arr.astype(np.float64, copy=False).ravel()
                        sq_sum += float(np.dot(values, values))
                    else:
                        nonfinite.append({'index': index, 'name': name, 'shape': list(arr.shape),
                                          'dtype': str(arr.dtype),
                                          'nonfinite_elements': int(arr.size - np.isfinite(arr).sum())})
                independent_norm = math.sqrt(sq_sum) if finite_count == grad_count else 'NaN'
                returned = original(self, params, *args, **kwargs)
                try:
                    clip_return = float(returned.item())
                except AttributeError:
                    clip_return = float(returned)
                rec['observations'].append({
                    'gradient_tensors': grad_count, 'finite_gradient_tensors': finite_count,
                    'nonfinite_gradient_tensors': len(nonfinite), 'first_nonfinite_gradients': nonfinite[:12],
                    'preclip_independent_fp64_l2_norm': independent_norm,
                    'clip_grad_norm_return': clip_return if math.isfinite(clip_return) else 'NaN',
                })
                return returned

            Accelerator.clip_grad_norm_ = measured
            try:
                from swift import PretrainArguments, pretrain_main
                result = pretrain_main(PretrainArguments(
                    model=str(MODEL), model_type='qwen2', dataset=[str(DATASET)], max_steps=1,
                    per_device_train_batch_size=1, gradient_accumulation_steps=1, save_steps=1,
                    split_dataset_ratio=0.01, tuner_type='lora', logging_steps=1,
                    torch_dtype='float32', fp16=train_fp16, bf16=False,
                    report_to='none', output_dir=str(OUT), dataset_num_proc=1))
            finally:
                Accelerator.clip_grad_norm_ = original
            checkpoint = Path(result['last_model_checkpoint'])
            from safetensors.numpy import load_file
            tensors = load_file(str(checkpoint / 'adapter_model.safetensors'))
            b = {k: v for k, v in tensors.items() if 'lora_B' in k}
            rec['checkpoint'] = str(checkpoint)
            rec['checkpoint_adapter_sha256'] = sha(checkpoint / 'adapter_model.safetensors')
            rec['lora_b'] = {'tensor_count': len(b),
                             'finite_tensors': sum(bool(np.isfinite(v).all()) for v in b.values()),
                             'nonzero_tensors': sum(bool(np.count_nonzero(v)) for v in b.values()),
                             'nonzero_elements': sum(int(np.count_nonzero(v)) for v in b.values())}
            after = int(jt.core.backend_fallback_count())
            rec['runtime']['fallback_after'] = after
            assert after == before, (before, after)
            rec['status'] = 'passed'
    except BaseException as exc:
        rec.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                   traceback=traceback.format_exc())
        raise
    finally:
        RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        RECEIPT.write_text(json.dumps(rec, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
