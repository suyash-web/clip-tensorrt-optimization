# Reproduce this project

## Supported scope

The full model-building runner targets **Linux with NVIDIA CUDA GPU**, tested on **Google Colab Tesla T4, Python 3.13, TensorRT 11.4.0.106**. It is not a macOS/CPU pipeline. Model availability, GPU allocation and upstream package compatibility are external prerequisites. Exact engine bytes, prediction ties and GPU latencies may change on rebuild.

## A completely fresh Colab run (no existing artifacts required)

1. Open a fresh Colab notebook, enable **T4 GPU** under Runtime → Change runtime type. Confirm `!nvidia-smi` reports an NVIDIA GPU.
2. Clone the source:

   ```python
   !git clone https://github.com/YOUR_USERNAME/clip-tensorrt-optimization.git /content/clip-tensorrt-optimization
   ```
3. Install once, using the notebook runtime interpreter:

   ```python
   %pip install -r /content/clip-tensorrt-optimization/requirements.txt
   ```

   If the kernel had already imported incompatible versions, restart the runtime before proceeding. The requirements do not pin the Colab-supplied CUDA-compatible PyTorch build; note the successful full environment version in the generated environment log.
4. Mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
5. Run pipeline with a **new empty Drive folder**:

   ```python
   import os, subprocess, sys
   os.environ["HF_HOME"] = "/content/hf"
   os.environ["HF_HUB_CACHE"] = "/content/hf/hub"
   proc = subprocess.run([
       sys.executable, "-u", "-m", "src.pipeline",
       "--root", "/content/drive/MyDrive/CLIP_Quantization_Clean_Test",
       "--stage", "all"
   ], cwd="/content/clip-tensorrt-optimization", check=True)
   ```

   It must complete with `Finished: ["prepare", "baseline", "quantize", "convert", "build", "evaluate"]` and a zero process exit code. The separate Drive folder protects any other project.
6. Inspect the completed results:

   ```python
   from pathlib import Path
   import pandas as pd
   root=Path("/content/drive/MyDrive/CLIP_Quantization_Clean_Test")
   for name in ["clip_baseline_fp16_metrics.json","clip_attention_fc1_int8_fp16_repaired_metrics.json"]:
       print(name, (root/"results"/name).read_text())
   display(pd.read_csv(root/"results/final_trt_comparison.csv"))
   ```

## Stages and saved files

| Stage | What it does | Key saved artifacts |
|---|---|---|
| `prepare` | Downloads pinned metadata; samples 100 train and 100 validation items per class; downloads 600 images | `data/calibration_manifest.csv`, `data/evaluation_manifest.csv`, data JPGs |
| `baseline` | Loads CLIP and computes FP32 reference predictions | `results/fp32_baseline.json` |
| `quantize` | ModelOpt max calibration on 300 training images; selective INT8; Q/DQ export | `phase2_artifacts/clip_selective_int8_modelopt.pth`, `clip_attention_fc1_int8_qdq.onnx` |
| `convert` | Converts non-Q/DQ weights to FP16; restores FP32 scales; verifies quantized integer weights | `phase2_artifacts/clip_attention_fc1_int8_fp16_repaired.onnx` |
| `build` | Creates dynamic-batch FP16 and mixed TensorRT engines on current NVIDIA GPU | `phase2_artifacts/clip_baseline_fp16.engine`, `clip_attention_fc1_int8_fp16_repaired.engine` |
| `evaluate` | Measures FP16 and mixed accuracy and repeats engine-only GPU timing 3× | `results/*_metrics.json`, `results/final_trt_comparison.csv` |

Additionally the runner writes `cpp_deployment/text_embeddings_fp32.bin` (3 × 512 float32, 6144 bytes), `cpp_deployment/labels.json`, `results/environment.json`, checksum records in `checksums/`, and numerical predictions/embeddings.

## Why saved outputs matter

Intermediate large files are generated on `/content`, transferred to Drive, checked with SHA-256, and recorded. On repeat, the runner tries to reuse validated calibration/ONNX/engine outputs; `--stage all` recomputes some lightweight stages and benchmarking. An existing file alone does not guarantee it is valid. After a Colab disconnect, reconnect/mount and run the same command. Do not manually delete the verified artifact folder unless intentionally rebuilding.

`--stage all` **does not compile or run the C++ app**. That is covered in [CPP_DEPLOYMENT.md](CPP_DEPLOYMENT.md). A successful Python clean-room run verifies Python/TensorRT components only.

## Validation criteria

A completed run must have: 300 calibration and 300 evaluation image records with no overlap, FP32 accuracy near reference, 123 quantizers, 123 Q/DQ pairs, 246 FP32 scales restored, 57,364,224 quantized weights unchanged, parsable ONNX, deserializable engines with input shape `(-1,3,224,224)` and output `(-1,512)`, and successful metric/benchmark CSV generation. Expected **historical** FP32 81%, FP16 80.67%, mixed 81.33%; the verified *fresh rebuild* mixed was 80.67%, so don"t hard-assert exact mixed predictions across engine rebuilds.

For lightweight CPU tests from the repository root after installing packages needed by tests:

```bash
python -m compileall -q src
python -m pytest -q tests
```

## Troubleshooting

| Symptom | Cause / next step |
|---|---|
| `ModuleNotFoundError: onnx` | Run `%pip install -r .../requirements.txt` in the **same Colab runtime**; if necessary restart runtime after installation. |
| Python `CalledProcessError` with little detail | Rerun command using `subprocess.run(..., stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)` and print the last 100 lines, or invoke `python -u -m src.pipeline ...` directly. |
| Missing model / interrupted download | Hugging Face cache must be local `/content/hf`, not mounted Drive. The loader uses Safetensors. Retry after confirming internet access and disk space. |
| Hugging Face unauthenticated warning or rate limit | Optional `HF_TOKEN` may improve rate limits. No token is hardcoded in the repo. |
| ModelOpt diffusers plugin or Transformers compatibility warnings | Optional plugin warnings were nonfatal in the observed run, but the untested Transformers combination may change. Record/pin a known-working lockfile when possible. |
| `FileNotFoundError` for a dataset JPEG | Confirm dataset revision, Hugging Face access, and that the preparation stage completed. Rerun `--stage prepare`. |
| Engine cannot deserialize | NVIDIA GPU/TensorRT runtime mismatch or damaged artifact. Regenerate on the target GPU; don"t assume an engine built on a different GPU can load. |
| Timing differs by several ms | Check GPU type, clocks/load, batch size, thermal/power conditions, warmup, and timer scope. Report repeated measurements, not guaranteed exact numbers. |
| Missing `text_embeddings_fp32.bin` | Run pipeline, then use `<root>/cpp_deployment/text_embeddings_fp32.bin`; it is intentionally not checked into Git. |
| macOS user cannot build | Native CUDA TensorRT requires a supported NVIDIA GPU/Linux setup; use Colab T4 or compatible NVIDIA machine. |

## Data splits and leakage

The code pins dataset revision and takes 100 images per class from the **train** split for calibration (seed 42), and 100 per class from **validation** (seed 43) for scoring. This is *post-training calibration*, not supervised training or fine-tuning. Do not tune prompts/thresholds based on the evaluation images and then report those same images as an unbiased test.
