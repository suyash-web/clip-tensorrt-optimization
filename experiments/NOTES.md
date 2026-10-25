# Earlier development experiments

The main pipeline uses the final selective INT8 / FP16 configuration only.
Earlier broad INT8 quantization reduced accuracy substantially (about 65.7% on the historical balanced subset); LayerNorm-aware variants did not yield the final chosen precision/performance balance. Keeping `fc2` in FP32 slowed the TensorRT engine, while unquantized FP16 `fc2` restored useful throughput.

These historical figures are not rerun by `run_pipeline.ipynb`. Different balanced samples were used before the final manifest-based evaluation, so historical accuracy values must not be mixed into the final comparison table.
