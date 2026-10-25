# Standalone C++ deployment

The C++17 application loads a previously built TensorRT vision encoder, performs JPEG decoding plus CLIP-compatible RGB/preprocessing, executes TensorRT, compares the 512-dimensional output with **three fixed normalized CLIP text embeddings**, and prints a CSV row per image. It is **not** a network service, camera driver, or probability-calibrated classifier.

## Requirements and compatibility

- **Linux with NVIDIA CUDA-capable GPU**, CUDA toolkit and CUDA runtime compatible with TensorRT build.
- TensorRT **11.4** C++ headers and matching `libnvinfer.so.11` runtime. Pip-installed Python TensorRT may provide libraries but not C++ headers.
- OpenCV **4.x** development package (includes JPEG decoding), GCC/G++ with C++17 and `pkg-config`.
- Compatible TensorRT engine and generated `text_embeddings_fp32.bin` from the Python pipeline. Engines built on a Tesla T4 may not load on other GPU/software stacks.

On Ubuntu/Colab Linux with a T4:

```bash
sudo apt-get update
sudo apt-get install -y g++ libopencv-dev pkg-config
export TRT_INCLUDE_DIR=/path/to/TensorRT-11.4/include
export TRT_LIB_DIR=/path/to/tensorrt_libs.  # optional if auto detected
bash cpp/build.sh
```

`cpp/build.sh` is designed to be invoked from the repository root; confirm the expected source file path in script if adapting. Do not copy NVIDIA proprietary headers into your public repository.

## Build artifacts needed

After running `python -m src.pipeline --stage all --root /.../CLIP_Quantization_Project`, locate:

- `<root>/phase2_artifacts/clip_attention_fc1_int8_fp16_repaired.engine`
- `<root>/cpp_deployment/text_embeddings_fp32.bin` (6144 bytes, 3 × 512 FP32, row-major)
- `<root>/cpp_deployment/labels.json` (label and prompt order)

For manual transfer, preserve binary bytes and ordering; changing prompts invalidates embeddings. `text_embeddings_fp32.bin` is generated automatically by CLIP text encoder and is omitted from the public repository intentionally.

## Run on one or multiple images

```bash
cpp/clip_classifier   /path/to/clip_attention_fc1_int8_fp16_repaired.engine   /path/to/text_embeddings_fp32.bin   /path/to/image1.jpg /path/to/image2.jpg
```

CSV columns:

```text
index,prediction,s_daytime,s_night,s_dawn_dusk,preprocess_ms,gpu_ms,end_to_end_ms
0,night,0.21725205,0.29542404,0.25177487,31.94,60.32,92.49
```

**The example timing row is a first-inference smoke test, not steady-state performance.** Similarity scores are dot products of normalized embeddings. They are not class probabilities. A higher similarity means the image embedding is more closely aligned to that prompt in CLIP space.

## Timing interpretation

`preprocess_ms` includes image loading/decoding and CPU resize/crop/normalization. `gpu_ms` measures a CUDA event around the TensorRT execution call in the C++ app. `end_to_end_ms` includes preprocessing, transfers and scoring (and excludes initial engine deserialization). Prior validated steady-state local-SSD measurements across 300 images: CPU 24.80 ms mean, GPU 8.71 ms mean, end-to-end 33.77 ms mean / 32.56 ms median / 39.96 ms p95. Those figures are not the same as the ~4-ms back-to-back Python engine-only benchmark.

The custom antialiased bicubic preprocessing approximates Pillow/CLIP closely but isn't guaranteed bitwise identical. Python mixed-engine classification achieved 81.33% on one engine, while C++ achieved 80.67%; the four differing predictions on that run are expected to be disclosed rather than hidden. The independently rebuilt clean-room mixed engine also achieved 80.67%, but a matching accuracy percentage does not mean predictions are identical.

## Smoke test and full validation

Run on one real JPEG first, confirm a valid `prediction` among the three labels, then run on a larger list of image paths using the **same fixed manifest ordering** to compare scores/predictions. The C++ evaluation and image staging are not invoked by `--stage all`; separately building the C++ app is required to validate the release. Do not compare GPU metrics collected under different hardware and scheduling conditions as if they were an apples-to-apples benchmark.
