# Technical report: selective CLIP INT8/FP16 quantization and C++ deployment

## Abstract

This project compresses a pretrained CLIP ViT-B/16 vision encoder for NVIDIA Tesla T4 inference using selective post-training INT8 quantization and FP16 execution. The final C++17/CUDA application accepts a JPEG and assigns one of three time-of-day labels by comparing image embeddings with fixed CLIP text embeddings. Relative to a 166.14 MiB FP16 TensorRT engine, the original mixed engine was 114.42 MiB (**31.1% smaller**). A clean-room rebuild on another Colab account completed all six pipeline stages successfully, reproducing 81.00% FP32 and 80.67% FP16 accuracy; mixed accuracy changed from 81.33% in an earlier engine to 80.67% in the independently rebuilt engine. Results demonstrate practical engine compression and a working C++ deployment—not a statistically established improvement in classification accuracy, nor guaranteed speedup for every batch size.

## 1. Problem and motivation

CLIP maps images and text into a comparable 512-dimensional embedding space. The three fixed prompts describe a road scene during daytime, night, or dawn/dusk. For each image, compute its projected vision embedding, normalize it, compare with normalized text embeddings, and use the largest cosine similarity as the class. The task is zero-shot classification; model weights and text prompts are not fitted on task labels. The optimization target is efficient GPU deployment of an existing model, not training a new time-of-day classifier.

## 2. Method and data

- **Model:** `openai/clip-vit-base-patch16` (CLIP ViT-B/16, vision projection 512-D).
- **Data:** `dronefreak/BDD100K-Period-Classification`; revision `47b8bde4a2d808df9208a89cdd10f68b251bcdb4`.
- **Calibration:** 300 images from training split, 100 each for daytime/night/dawn-dusk (seed 42).
- **Evaluation:** separate 300 validation images, 100 per class (seed 43). Validation data is not used for calibration.
- **Preprocessing:** RGB, bicubic resize shortest edge to 224, center crop to 224×224, map 8-bit pixels to [0,1], normalize channel means `[0.48145466,0.4578275,0.40821073]` and stds `[0.26862954,0.26130258,0.27577711]`.
- **Labels and prompts:** see [`src/common.py`](../src/common.py). `argmax` is over cosine similarities, not calibrated confidences.

### Quantization design

Using NVIDIA ModelOpt with maximum calibration, INT8 quantizers are enabled on attention query/key/value/output projections, MLP `fc1`, and selected embedding weights/inputs. MLP `fc2` and LayerNorm input quantization remain disabled. The model uses **123 enabled quantizers** and exports **123 `QuantizeLinear` plus 123 `DequantizeLinear`** ONNX nodes. This was chosen after broader INT8 approaches hurt accuracy; it is not a claim that the architecture/quantization algorithm is globally optimal.

An ONNX float16 conversion then converts eligible non-Q/DQ computation (including 12 `fc2` GEMMs) to FP16 while preserving FP32 model input/output. The converter may cast/round quantization scales, so the original **246 FP32 Q/DQ scale inputs** are restored explicitly. Quantized values of **57,364,224 weights** were compared before and after conversion with **zero changes**. The baseline FP16 ONNX graph needs an explicit FP32→FP16 Cast at its patch embedding convolution input to satisfy TensorRT type constraints.

TensorRT engines use dynamic batches min=1, opt=8, max=32. The rebuilt mixed engine is **not equivalent to a fully INT8 model**: it deliberately mixes INT8 matrix operations, FP16 layers, and FP32 I/O/boundaries.

## 3. Measured results and provenance

### Accuracy: source of truth

| Validation run | FP32 accuracy | FP16 TensorRT accuracy | Mixed TensorRT accuracy | Mixed macro F1 |
|---|---:|---:|---:|---:|
| Original development/engine run | 81.00% (243/300) | 80.67% (242/300) | 81.33% (244/300) | 0.8030 |
| New account, **empty Drive / fresh engine build** | **81.00% (243/300)** | **80.67% (242/300)** | **80.67% (242/300)** | **0.7941** |

Fresh build mixed confusion matrix (rows true labels, columns predicted, order: daytime, night, dawn/dusk):

```text
[[96, 0, 4],
 [ 0,98, 2],
 [49, 3,48]]
```

Fresh build class recalls: daytime 96%, night 98%, dawn/dusk **48%**. This is an important weakness, especially for boundary lighting conditions. The mixed-engine result differs from the earlier engine by **two correct images**; don't claim deterministic bit-for-bit accuracy from TensorRT compilation.

### Engine storage size

| Baseline | MiB |
|---|---:|
| FP16 TRT (original, subsequently revalidated) | 166.14 |
| Mixed INT8/FP16 TRT (original, subsequently revalidated) | 114.42 |
| Reduction | **31.1% / 51.72 MiB** |

Engine-file bytes are not GPU-memory usage, and TensorRT engines should not be assumed portable across GPU/runtime versions.

### GPU engine-only latency

A previously recorded **same-session** single comparison showed 4.410 vs 4.229 ms (batch 1) and 91.727 vs 84.006 ms (batch 32), favoring mixed. A later **three-repetition** same-session comparison favored mixed slightly at batch 1 but FP16 at batch 32:

| Three-run mean, Tesla T4 | FP16 | Mixed INT8/FP16 | Direction |
|---|---:|---:|---|
| Batch-1 mean latency (ms) | 4.018 | 3.908 | Mixed ~2.7% faster |
| Batch-32 mean latency (ms) | **59.397** | 62.184 | Mixed ~4.7% slower |
| Batch-32 throughput (images/s) | **~539** | ~515 | FP16 faster |

These are observations from different runtime conditions, not contradictions in accuracy. No universal TensorRT speedup is claimed. The fresh clean-room run completed three repetitions per batch/engine, but **the numeric benchmark CSV from that new account was not supplied for this report**; do not mislabel historical timings as fresh clean-room timings. Raw provided prior runs are transcribed in [`results/reference_runs.csv`](../results/reference_runs.csv).

### C++ JPEG-to-label deployment (previously validated)

| Metric | Result |
|---|---:|
| Correct predictions / accuracy | 242/300 / **80.67%** |
| Macro F1 | 0.7951 |
| Agreement with earlier Python mixed engine | 98.67% |
| Local-storage mean CPU decode + preprocess | 24.799 ms |
| Local-storage mean GPU event time | 8.708 ms |
| Local-storage mean end-to-end | **33.768 ms** |
| Local-storage median end-to-end | **32.564 ms** |
| Local-storage p95 end-to-end | **39.960 ms** |

These C++ results are from an earlier manually compiled app; `src.pipeline --stage all` does not regenerate them. Measurements exclude engine deserialization/startup and use sequential images from local disk, whereas the engine benchmark runs kernels back-to-back on a preallocated tensor. The clocks/methodology differ, so **4 ms engine-only** is not the application GPU time. The approximately 25 ms preprocessing cost dominates the C++ path.

## 4. Failure modes addressed

- Google Drive Hugging Face model caches could be corrupted during interruptions: caches now live on local Colab disk, and `use_safetensors=True` is requested.
- Original large ONNX/checkpoint/engine outputs could be partial after disconnects: large artifacts are built locally, copied to Drive with SHA-256 validation, and tracked with manifest records.
- Earlier script relied on values defined in unrelated notebook cells: the public pipeline uses ordered stages and saves checkpoints.
- `fc2` left in FP32 caused poor TensorRT performance: FP16 conversion removes that bottleneck while keeping selected INT8 quantization.
- A TensorRT parse failure caused by FP32 Conv input with FP16 weights: explicit Cast correction for the FP16 baseline.
- OpenCV bicubic did not reproduce Python CLIP interpolation accurately enough: C++ uses a custom antialiased bicubic resampler, with regression validation performed against 300 images.

## 5. Interpretation and limits

What is supported: a materially smaller mixed-engine file; accuracy on a small validation split roughly comparable to FP32/FP16; a functioning C++ application; clean-room completion and staged restart/reuse. What is **not** supported: improved robustness to weather/geographies; a trained model; statistically significant superiority; guaranteed identical engine/timing on all GPUs; a GPU-peak-memory reduction; or safety-critical road-scene suitability.

A stronger future evaluation would use a larger, genuinely held-out deployment dataset with weather/lighting breakdowns and repeated timing across controlled GPU clock states. Prompt tuning, if attempted, should use development data **separate from the evaluation set**.

## 6. Reproducibility evidence

A new Colab account on a fresh workspace completed `prepare`, `baseline`, `quantize`, `convert`, `build`, and `evaluate` (exit code 0). A second invocation restored the ModelOpt checkpoint, mixed ONNX, and both engines (skip messages), while repeating evaluation and benchmarks. This proves that the public pipeline can generate and reuse required artifacts in that environment. It does not prove failure-free execution indefinitely as dependency releases and Colab base images evolve. Dependency/driver versions should be archived via the generated `results/environment.json` and optional `pip freeze` for future replay.
