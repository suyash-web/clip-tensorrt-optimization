from pathlib import Path
import pandas as pd
import torch
import transformers
import onnx
import tensorrt
import hashlib
import json
import os
import sys
import shutil
import tempfile
from importlib.metadata import version

os.environ.setdefault("HF_HOME", "/content/hf")
os.environ.setdefault("HF_HUB_CACHE", "/content/hf/hub")

MODEL_ID = "openai/clip-vit-base-patch16"
DATASET_ID = "dronefreak/BDD100K-Period-Classification"
DATASET_REVISION = "47b8bde4a2d808df9208a89cdd10f68b251bcdb4"
LABELS = ["daytime", "night", "dawn/dusk"]
PROMPTS = ["A photograph of a road during daytime",
           "A photograph of a road at night",
           "A photograph of a road at dawn or dusk"]
SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def copy_verified(src: Path, dest: Path) -> None:
    """Copy from local disk to persistent storage and verify before publishing."""
    src, dest = Path(src), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".partial")
    shutil.copyfile(src, temp)
    if sha256(src) != sha256(temp):
        temp.unlink(missing_ok=True)
        raise OSError("Checksum mismatch while transferring " + str(dest))
    os.replace(temp, dest)
    if sha256(src) != sha256(dest):
        raise OSError("Checksum mismatch after publishing " + str(dest))


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / path.name
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        copy_verified(temporary, path)


def artifact_ok(root: Path, relative: str, validator=None) -> bool:
    """A hash-matched artifact may be reused; existence alone is not enough."""
    path = root / relative
    proof = root / "checksums" / (relative.replace("/", "__") + ".json")
    if not path.is_file() or not proof.is_file():
        return False
    try:
        record = json.loads(proof.read_text())
        if record["sha256"] != sha256(path) or record["bytes"] != path.stat().st_size:
            return False
        if validator is not None:
            validator(path)
        return True
    except (ValueError, OSError, KeyError, RuntimeError, AssertionError, ImportError):
        return False


def publish(root: Path, relative: str, source: Path, extra=None) -> Path:
    target = root / relative
    copy_verified(source, target)
    record = {"relative_path": relative, "bytes": target.stat().st_size,
              "sha256": sha256(target), "extra": extra or {}}
    write_json(root / "checksums" / (relative.replace("/", "__") + ".json"), record)
    return target


def validate_onnx(path: Path) -> None:
    onnx.checker.check_model(str(path))


def validate_engine(path: Path) -> None:
    log = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine = tensorrt.Runtime(log).deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError("Unable to deserialize engine: " + str(path))
    assert engine.get_tensor_shape("pixel_values")[0] == -1


def validate_csv(path: Path, count=300) -> None:
    frame = pd.read_csv(path)
    if len(frame) != count:
        raise ValueError(f"Expected {count} rows in {path}, got {len(frame)}")


def versions() -> dict:
    r = {"python": sys.version, "torch": torch.__version__,
         "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
         "onnx": onnx.__version__, "tensorrt": tensorrt.__version__,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}
    try:
        r["nvidia-modelopt"] = version("nvidia-modelopt")
    except Exception:
        pass
    return r
