# Release acceptance checklist (must be run on a fresh T4 GPU runtime)

- [ ] Colab selected T4 GPU and `nvidia-smi` succeeds.
- [ ] `notebooks/run_pipeline.ipynb` runs without manual edits in a fresh session.
- [ ] Dataset manifests have 300 rows each and 100 images per class; train/eval disjoint.
- [ ] CLIP Safetensors loads and text embeddings shape is `(3,512)`.
- [ ] ModelOpt selective calibration has **123** enabled quantizers.
- [ ] Export has **123** QuantizeLinear and 123 DequantizeLinear nodes.
- [ ] Restored FP32 scales match **246** original scale uses.
- [ ] **62** INT8 weight quantizers and **57,364,224** values preserved with **0** differences.
- [ ] Both TensorRT 11.4 engine variants deserialize and support batch sizes 1–32.
- [ ] FP32 and TRT accuracy checked on exactly same 300 examples; compare with historical 81.0/80.67/81.33% only as approximate references.
- [ ] Run repeated CUDA benchmarks and report mean/spread, not only one timing.
- [ ] Restart runtime and rerun workflow; verified artifacts should skip generation.
- [ ] Build C++ with TensorRT 11.4 headers and run on test images; test classifications and latency against Python.
- [ ] Capture driver version, CUDA version, GPU model, package freeze, and SHA-256 manifest.

**The repository has not passed this end-to-end fresh-runtime checklist yet.**

## Reader-facing documentation

See `docs/REPRODUCING.md`, `docs/TECHNICAL_REPORT.md` and `docs/CPP_DEPLOYMENT.md`. The clean-room Python pipeline passed on a new Colab account with no preexisting Drive files; a C++ rebuild from that new account is not yet established.
