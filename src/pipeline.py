import argparse
import copy
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import numpy as np
import onnx
import pandas as pd
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from onnxconverter_common import float16
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from .common import (
    DATASET_ID,
    DATASET_REVISION,
    LABELS,
    MODEL_ID,
    PROMPTS,
    SEED,
    artifact_ok,
    copy_verified,
    publish,
    sha256,
    validate_csv,
    validate_engine,
    validate_onnx,
    versions,
    write_json,
)

STAGES = ["prepare", "baseline", "quantize", "convert", "build", "evaluate"]


def existing_or_verified(root, relative, validator=None):
    """Safely adopt a pre-existing user artifact after structural validation.

    Existing Colab projects predate checksum sidecars. Verify the artifact
    before recording its fingerprint.
    """
    if artifact_ok(root, relative, validator):
        return True
    path = root / relative
    if not path.is_file():
        return False
    try:
        if validator is not None:
            validator(path)
        elif path.stat().st_size == 0:
            return False
        write_json(
            root / "checksums" / (relative.replace("/", "__") + ".json"),
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "extra": {"adopted_from_existing": True},
            },
        )
        return True
    except Exception as exc:
        print("Rebuilding invalid existing artifact:", relative, repr(exc))
        return False


def load_clip(device):
    # use safetensors
    model = (
        CLIPModel.from_pretrained(
            MODEL_ID, use_safetensors=True, dtype=torch.float32, cache_dir="/content/hf"
        )
        .to(device)
        .eval()
    )
    processor = CLIPProcessor.from_pretrained(MODEL_ID, cache_dir="/content/hf")

    class VisionEncoder(nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.vision_model = clip.vision_model
            self.visual_projection = clip.visual_projection

        def forward(self, pixel_values):
            return self.visual_projection(
                self.vision_model(pixel_values=pixel_values).pooler_output
            )

    vision = VisionEncoder(model).to(device).eval()
    inputs = {
        k: v.to(device)
        for k, v in processor(text=PROMPTS, return_tensors="pt", padding=True).items()
    }
    with torch.inference_mode():
        vectors = model.get_text_features(**inputs)
        if not isinstance(vectors, torch.Tensor):
            vectors = vectors.pooler_output
        text = F.normalize(vectors.float(), dim=-1)
    assert text.shape == (3, 512)
    return vision, processor, text


def manifest(root, split):
    df = pd.read_csv(root / "data" / f"{split}_manifest.csv")
    assert len(df) == 300
    assert all(df.label.value_counts().get(k, 0) == 100 for k in LABELS)
    assert all(
        df["label_index"].to_numpy()
        == df["label"].map({k: i for i, k in enumerate(LABELS)}).to_numpy()
    )
    # CSV may come from a different user"s system: ignore old absolute local_path.
    df["local_path"] = df.repo_path.map(lambda x: str(root / "data" / x))
    return df


def prepare(root):
    root.joinpath("data").mkdir(parents=True, exist_ok=True)
    calpath, evalpath = (
        root / "data/calibration_manifest.csv",
        root / "data/evaluation_manifest.csv",
    )
    if not (
        existing_or_verified(root, "data/calibration_manifest.csv", validate_csv)
        and existing_or_verified(root, "data/evaluation_manifest.csv", validate_csv)
    ):
        files = HfApi().list_repo_files(
            repo_id=DATASET_ID, repo_type="dataset", revision=DATASET_REVISION
        )
        frames = []
        for name in sorted(f for f in files if f.endswith("/metadata.jsonl")):
            path = hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=name,
                revision=DATASET_REVISION,
                cache_dir="/content/hf",
            )
            f = pd.read_json(path, lines=True)
            f["source_metadata"] = name
            f["source_split"] = "validation" if "/val/" in name else "train"
            frames.append(f)
        all_metadata = pd.concat(frames, ignore_index=True)
        with tempfile.TemporaryDirectory(dir="/content") as tmp:
            for output, split, seed in [
                ("calibration", "train", SEED),
                ("evaluation", "validation", SEED + 1),
            ]:
                df = all_metadata[
                    (all_metadata.source_split == split)
                    & all_metadata.label.isin(LABELS)
                ].copy()
                df = (
                    df.groupby("label", group_keys=False)
                    .sample(n=100, random_state=seed)
                    .sample(frac=1, random_state=seed)
                    .reset_index(drop=True)
                )
                df["repo_path"] = df.apply(
                    lambda r: str(
                        PurePosixPath(r.source_metadata).parent / r.file_name
                    ),
                    axis=1,
                )
                df["label_index"] = df.label.map({k: i for i, k in enumerate(LABELS)})
                # Store portable relative paths, never notebook-specific mount paths.
                df["local_path"] = df.repo_path
                local = Path(tmp) / f"{output}.csv"
                df.to_csv(local, index=False)
                validate_csv(local)
                publish(
                    root,
                    f"data/{output}_manifest.csv",
                    local,
                    {
                        "dataset_revision": DATASET_REVISION,
                        "seed": seed,
                        "classes": LABELS,
                    },
                )
    cal, ev = manifest(root, "calibration"), manifest(root, "evaluation")
    assert set(cal.repo_path).isdisjoint(set(ev.repo_path))
    jobs = list(dict.fromkeys(cal.repo_path.tolist() + ev.repo_path.tolist()))

    def fetch(path):
        target = root / "data" / path
        if target.is_file():
            try:
                with Image.open(target) as img:
                    img.verify()
                return
            except (OSError, ValueError):
                target.unlink(missing_ok=True)
        remote = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=path,
            revision=DATASET_REVISION,
            cache_dir="/content/hf",
        )
        with tempfile.TemporaryDirectory(dir="/content") as tmp:
            local = Path(tmp) / Path(path).name
            shutil.copyfile(remote, local)
            with Image.open(local) as img:
                img.verify()
            # Images don't need hash sidecars individually; verify format at use.
            copy_verified(local, target)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        for future in as_completed(futures):
            future.result()
    print("Data prepared:", len(cal), "calibration;", len(ev), "evaluation")


def data_loaders(root, processor):
    class ImageDataset(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            with Image.open(self.df.iloc[i].local_path) as image:
                return processor(images=image.convert("RGB"), return_tensors="pt")[
                    "pixel_values"
                ].squeeze(0)

    cal, ev = manifest(root, "calibration"), manifest(root, "evaluation")
    return (
        DataLoader(
            ImageDataset(cal),
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        DataLoader(
            ImageDataset(ev),
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        ev.label_index.to_numpy(dtype="int64"),
    )


def eval_torch(encoder, loader, text, device):
    predictions = []
    embeddings = []
    encoder.eval()
    with torch.inference_mode():
        for x in loader:
            z = F.normalize(encoder(x.to(device, non_blocking=True)).float(), dim=-1)
            predictions.extend((z @ text.T).argmax(-1).cpu().tolist())
            embeddings.append(z.cpu())
    return np.asarray(predictions, dtype="int64"), torch.cat(embeddings).numpy()


def score(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro")),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
    }


def write_predictions(root, relative, predictions, embeddings=None):
    with tempfile.TemporaryDirectory(dir="/content") as tmp:
        p = Path(tmp) / "p.npy"
        np.save(p, predictions)
        publish(root, "results/" + relative + "_predictions.npy", p)
        if embeddings is not None:
            e = Path(tmp) / "e.npy"
            np.save(e, embeddings)
            publish(root, "results/" + relative + "_embeddings.npy", e)


def baseline(root, vision, processor, text, device):
    rel = "results/fp32_baseline.json"
    if artifact_ok(root, rel):
        print("skip: FP32 baseline")
        return
    _, ev, y = data_loaders(root, processor)
    pred, emb = eval_torch(vision, ev, text, device)
    write_predictions(root, "fp32", pred, emb)
    write_json(root / rel, score(y, pred))
    print("FP32:", score(y, pred))


def quantize(root, vision, processor, device):
    """Restore or calibrate quantized model. Returned model is on CUDA."""
    rel = "phase2_artifacts/clip_selective_int8_modelopt.pth"
    candidate = copy.deepcopy(vision).to(device).eval()
    if (root / rel).is_file():
        try:
            mto.restore(candidate, str(root / rel))
            enabled = sum(
                bool(m.is_enabled)
                for n, m in candidate.named_modules()
                if n.endswith(("input_quantizer", "weight_quantizer"))
            )
            if enabled == 123:
                existing_or_verified(root, rel)
                print("skip: restored calibrated ModelOpt checkpoint")
                return candidate
            raise RuntimeError(f"Unexpected restored quantizers: {enabled}")
        except Exception as error:
            print("Checkpoint restore failed; recalibrating:", repr(error))
            candidate = copy.deepcopy(vision).to(device).eval()
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            *[
                {
                    "quantizer_name": f"*self_attn.*_proj.{t}_quantizer",
                    "cfg": {"num_bits": 8, "axis": axis},
                }
                for t, axis in [("weight", 0), ("input", None)]
            ],
            *[
                {
                    "quantizer_name": f"*mlp.fc1.{t}_quantizer",
                    "cfg": {"num_bits": 8, "axis": axis},
                }
                for t, axis in [("weight", 0), ("input", None)]
            ],
            *[
                {
                    "quantizer_name": f"*patch_embedding.{t}_quantizer",
                    "cfg": {"num_bits": 8, "axis": axis},
                }
                for t, axis in [("weight", 0), ("input", None)]
            ],
            *[
                {
                    "quantizer_name": f"*position_embedding.{t}_quantizer",
                    "cfg": {"num_bits": 8, "axis": axis},
                }
                for t, axis in [("weight", 0), ("input", None)]
            ],
        ],
        "algorithm": "max",
    }
    cal, _, _ = data_loaders(root, processor)

    def forward_loop(model):
        with torch.inference_mode():
            for x in cal:
                model(x.to(device, non_blocking=True))

    candidate = mtq.quantize(candidate, cfg, forward_loop)
    active = [
        n
        for n, m in candidate.named_modules()
        if n.endswith(("input_quantizer", "weight_quantizer")) and bool(m.is_enabled)
    ]
    if len(active) != 123:
        raise RuntimeError(
            f"Expected 123 enabled quantizers, got {len(active)}: {active[:8]}"
        )
    with tempfile.TemporaryDirectory(dir="/content") as tmp:
        p = Path(tmp) / "modelopt.pth"
        mto.save(candidate, str(p))
        publish(root, rel, p, {"enabled_quantizers": len(active)})
    print("calibrated:", len(active), "quantizers")
    return candidate


def export_model(vision, root, relative, quantized=False):
    p = root / relative
    if existing_or_verified(root, relative, validate_onnx):
        return p
    with tempfile.TemporaryDirectory(dir="/content") as tmp:
        output = Path(tmp) / "export.onnx"
        vision.eval()
        with torch.inference_mode():
            torch.onnx.export(
                vision,
                (torch.randn(1, 3, 224, 224, device="cuda"),),
                str(output),
                input_names=["pixel_values"],
                output_names=["image_features"],
                dynamic_axes={
                    "pixel_values": {0: "batch_size"},
                    "image_features": {0: "batch_size"},
                },
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        validate_onnx(output)
        if quantized:
            counts = Counter(x.op_type for x in onnx.load(str(output)).graph.node)
            if counts["QuantizeLinear"] != 123 or counts["DequantizeLinear"] != 123:
                raise RuntimeError(f"Q/DQ pairs differ from reference: {counts}")
        publish(root, relative, output)
    return p


def resolve_constants(model):
    initial = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    producers = {o: n for n in model.graph.node for o in n.output}

    def resolve(name, seen=None):
        seen = set() if seen is None else seen
        if name in seen:
            return None
        seen.add(name)
        if name in initial:
            return initial[name]
        node = producers.get(name)
        if node is None:
            return None
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
        if node.op_type in ("Cast", "Identity"):
            value = resolve(node.input[0], seen)
            if value is None:
                return None
            if node.op_type == "Identity":
                return value
            to = next((a for a in node.attribute if a.name == "to"), None)
            if to is not None:
                return value.astype(onnx.helper.tensor_dtype_to_np_dtype(to.i))
        return None

    return resolve


def qdq_scales(model):
    resolve = resolve_constants(model)
    return {
        n.name: resolve(n.input[1])
        for n in model.graph.node
        if n.op_type in ("QuantizeLinear", "DequantizeLinear")
    }


def weight_values(node, resolve):
    w = resolve(node.input[0])
    s = resolve(node.input[1])
    if w is None or s is None:
        raise RuntimeError("Missing quantization constant: " + node.name)
    axis = next(
        (
            onnx.helper.get_attribute_value(a)
            for a in node.attribute
            if a.name == "axis"
        ),
        1,
    )

    def broadcast(x):
        x = np.asarray(x)
        if x.size == 1:
            return x.reshape(())
        if x.ndim != 1 or x.size != w.shape[axis]:
            raise ValueError("Unexpected per-channel scale shape")
        sh = [1] * w.ndim
        sh[axis] = x.size
        return x.reshape(sh)

    if len(node.input) >= 3 and node.input[2]:
        zp = resolve(node.input[2])
        dtype = zp.dtype
        zero = broadcast(zp).astype("int64")
    else:
        dtype = np.dtype("uint8")
        zero = 0
    mn, mx = (-128, 127) if dtype == np.dtype("int8") else (0, 255)
    return np.clip(
        np.rint(w.astype("float64") / broadcast(s).astype("float64")) + zero, mn, mx
    ).astype(dtype)


def convert(root):
    original = root / "phase2_artifacts/clip_attention_fc1_int8_qdq.onnx"
    target = "phase2_artifacts/clip_attention_fc1_int8_fp16_repaired.onnx"
    if existing_or_verified(root, target, validate_onnx):
        print("skip: mixed precision ONNX")
        return root / target
    orig = onnx.load(str(original))
    scales = qdq_scales(orig)
    assert len(scales) == 246 and all(x is not None for x in scales.values())
    mixed = float16.convert_float_to_float16(
        orig,
        keep_io_types=True,
        disable_shape_infer=False,
        op_block_list=["QuantizeLinear", "DequantizeLinear"],
    )
    j = 0
    for node in mixed.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            if node.name not in scales:
                raise RuntimeError("Q/DQ name changed")
            name = f"restored_qdq_scale_{j}"
            mixed.graph.initializer.append(
                numpy_helper.from_array(
                    np.asarray(scales[node.name], dtype=np.float32), name
                )
            )
            node.input[1] = name
            j += 1
    assert j == 246
    final_scales = qdq_scales(mixed)
    assert all(
        np.array_equal(value, final_scales[name]) for name, value in scales.items()
    )
    orig_nodes = {n.name: n for n in orig.graph.node}
    new_nodes = {n.name: n for n in mixed.graph.node}
    src, dst = resolve_constants(orig), resolve_constants(mixed)
    changed = 0
    total = 0
    count = 0
    for name, n in orig_nodes.items():
        if n.op_type != "QuantizeLinear" or "weight_quantizer" not in name:
            continue
        a = weight_values(n, src)
        b = weight_values(new_nodes[name], dst)
        total += a.size
        changed += np.count_nonzero(a != b)
        count += 1
    if count != 62 or total != 57364224 or changed != 0:
        raise RuntimeError(
            f"Unexpected INT8 preservation: quantizers={count} values={total} changed={changed}"
        )
    with tempfile.TemporaryDirectory(dir="/content") as tmp:
        path = Path(tmp) / "mixed.onnx"
        onnx.save(mixed, str(path))
        validate_onnx(path)
        publish(
            root,
            target,
            path,
            {"weight_values": total, "changed": changed, "scales": j},
        )
    return root / target


def make_fp16(root, vision):
    relative = "phase2_artifacts/clip_baseline_fp16_fixed.onnx"
    if existing_or_verified(root, relative, validate_onnx):
        return root / relative
    original = export_model(vision, root, "phase2_artifacts/clip_baseline_fp32.onnx")
    mixed = float16.convert_float_to_float16(
        onnx.load(str(original)), keep_io_types=True, disable_shape_infer=False
    )
    matches = [
        n
        for n in mixed.graph.node
        if n.op_type == "Conv" and "patch_embedding" in n.name
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one patch embedding Conv")
    conv = matches[0]
    original_input = conv.input[0]
    cast = helper.make_node(
        "Cast",
        inputs=[original_input],
        outputs=["patch_embedding_input_fp16"],
        to=TensorProto.FLOAT16,
        name="patch_embedding_input_cast_fp16",
    )
    index = next(i for i, n in enumerate(mixed.graph.node) if n is conv)
    mixed.graph.node.insert(index, cast)
    conv.input[0] = "patch_embedding_input_fp16"
    with tempfile.TemporaryDirectory(dir="/content") as tmp:
        path = Path(tmp) / "fp16.onnx"
        onnx.save(mixed, str(path))
        validate_onnx(path)
        publish(root, relative, path)
    return root / relative


def build_one(root, onnx_path, name):
    relative = "phase2_artifacts/" + name + ".engine"
    if existing_or_verified(root, relative, validate_engine):
        print("skip: engine", name)
        return root / relative
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    network = builder.create_network()
    parser = trt.OnnxParser(network, log)
    with open(onnx_path, "rb") as f:
        valid = parser.parse(f.read())
    if not valid:
        raise RuntimeError(
            "ONNX parser error: "
            + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "pixel_values",
        min=(1, 3, 224, 224),
        opt=(8, 3, 224, 224),
        max=(32, 3, 224, 224),
    )
    assert bool(profile)
    config.add_optimization_profile(profile)
    with tempfile.TemporaryDirectory(dir="/content") as tmp:
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT builder failed")
        path = Path(tmp) / "engine.plan"
        path.write_bytes(engine)
        validate_engine(path)
        publish(
            root,
            relative,
            path,
            {"trt_version": trt.__version__, "gpu": versions()["gpu"]},
        )
    return root / relative


def get_engine(path):
    log = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(log).deserialize_cuda_engine(Path(path).read_bytes())
    assert engine is not None
    return engine


def eval_engine(engine, loader, text, device):
    ctx = engine.create_execution_context()
    stream = torch.cuda.current_stream()
    preds = []
    emb = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(
                device=device, dtype=torch.float32, non_blocking=True
            ).contiguous()
            assert ctx.set_input_shape("pixel_values", tuple(images.shape))
            out = torch.empty(
                tuple(ctx.get_tensor_shape("image_features")),
                device=device,
                dtype=torch.float32,
            )
            assert ctx.set_tensor_address("pixel_values", images.data_ptr())
            assert ctx.set_tensor_address("image_features", out.data_ptr())
            assert ctx.execute_async_v3(stream_handle=stream.cuda_stream)
            norm = F.normalize(out.float(), dim=-1)
            preds.extend((norm @ text.T).argmax(-1).cpu().tolist())
            emb.append(norm.cpu())
    stream.synchronize()
    return np.asarray(preds, dtype="int64"), torch.cat(emb).numpy()


def bench(engine, batch, warmup=50, iters=150):
    ctx = engine.create_execution_context()
    assert ctx.set_input_shape("pixel_values", (batch, 3, 224, 224))
    x = torch.randn(batch, 3, 224, 224, device="cuda")
    y = torch.empty(tuple(ctx.get_tensor_shape("image_features")), device="cuda")
    assert ctx.set_tensor_address("pixel_values", x.data_ptr())
    assert ctx.set_tensor_address("image_features", y.data_ptr())
    stream = torch.cuda.current_stream()
    stream.synchronize()
    for _ in range(warmup):
        assert ctx.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record(stream)
        assert ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        stop.record(stream)
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    times = np.asarray(times)
    return {
        "batch_size": batch,
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "throughput_img_s": float(batch * 1000 / times.mean()),
    }


def evaluate(root, processor, text, device):
    _, ev, y = data_loaders(root, processor)
    allresults = []
    for label, name in [
        ("FP16 TensorRT", "clip_baseline_fp16"),
        ("Mixed INT8/FP16 TensorRT", "clip_attention_fc1_int8_fp16_repaired"),
    ]:
        eng = get_engine(root / "phase2_artifacts" / (name + ".engine"))
        pred, embedding = eval_engine(eng, ev, text, device)
        write_predictions(
            root,
            "fp16_trt" if name == "clip_baseline_fp16" else "mixed_trt",
            pred,
            embedding,
        )
        stats = score(y, pred)
        write_json(root / "results" / (name + "_metrics.json"), stats)
        for bs in (1, 32):
            for run in range(3):
                r = bench(eng, bs)
                r.update({"model": label, "run": run})
                allresults.append(r)
        print(name, stats)
    with tempfile.TemporaryDirectory(dir="/content") as tmp:
        p = Path(tmp) / "comparison.csv"
        pd.DataFrame(allresults).to_csv(p, index=False)
        publish(root, "results/final_trt_comparison.csv", p)
    print("Completed benchmark (3 repetitions per engine and batch).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/content/drive/MyDrive/CLIP_Quantization_Project"),
    )
    p.add_argument("--stage", choices=["all"] + STAGES, default="all")
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required: select T4 GPU runtime in Colab.")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    write_json(root / "results/environment.json", versions())
    selected = STAGES if args.stage == "all" else [args.stage]
    if "prepare" in selected:
        prepare(root)
    # Other stages require both manifests and verified images; run prepare first.
    if selected == ["prepare"]:
        return
    vision, processor, text = load_clip("cuda")
    with tempfile.TemporaryDirectory(dir="/content") as tmp:
        q = Path(tmp) / "text.bin"
        text.detach().cpu().numpy().astype("<f4").tofile(q)
        publish(root, "cpp_deployment/text_embeddings_fp32.bin", q)
    write_json(
        root / "cpp_deployment/labels.json",
        {
            "labels": LABELS,
            "prompts": PROMPTS,
            "embedding_dimension": 512,
            "text_embedding_layout": "row_major_float32",
        },
    )
    if "baseline" in selected:
        baseline(root, vision, processor, text, "cuda")
    quant = None
    if "quantize" in selected:
        quant = quantize(root, vision, processor, "cuda")
        export_model(
            quant,
            root,
            "phase2_artifacts/clip_attention_fc1_int8_qdq.onnx",
            quantized=True,
        )
    if "convert" in selected:
        convert(root)
    if "build" in selected:
        make_fp16(root, vision)
        build_one(
            root,
            root / "phase2_artifacts/clip_baseline_fp16_fixed.onnx",
            "clip_baseline_fp16",
        )
        build_one(
            root,
            root / "phase2_artifacts/clip_attention_fc1_int8_fp16_repaired.onnx",
            "clip_attention_fc1_int8_fp16_repaired",
        )
    if "evaluate" in selected:
        evaluate(root, processor, text, "cuda")
    print("Finished:", selected)


if __name__ == "__main__":
    main()
