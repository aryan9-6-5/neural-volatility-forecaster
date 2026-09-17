import json
import os
import torch

from models.architectures import StackedLSTM, ConvLSTM, TransformerEncoderModel, HARRVLSTMHybrid

MODEL_REGISTRY = {
    "StackedLSTM": StackedLSTM,
    "ConvLSTM": ConvLSTM,
    "TransformerEncoderModel": TransformerEncoderModel,
    "HARRVLSTMHybrid": HARRVLSTMHybrid,
}

# Extra (non-parameter, non-buffer) attributes models carry for serving/reproducibility.
# These live in the JSON sidecar since they don't survive state_dict().
_EXTRA_ATTR_KEYS = [
    "data_contract_version",
    "lookback_window",
    "horizons",
    "tensor_orientation",
    "train_mean",
    "train_std",
]


def _sidecar_path(checkpoint_path: str) -> str:
    return checkpoint_path + ".json"


def save_checkpoint(model: torch.nn.Module, checkpoint_path: str, model_class_name: str,
                     constructor_kwargs: dict, extra_meta: dict = None) -> None:
    """
    Save a model as a plain state_dict plus a JSON sidecar describing how to
    reconstruct it. Avoids pickling the module itself (torch.save(model, ...)),
    which is fragile across code refactors and requires weights_only=False
    (arbitrary code execution risk) to load back.
    """
    if model_class_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class '{model_class_name}'. Known: {list(MODEL_REGISTRY.keys())}")

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    sidecar = {
        "model_class": model_class_name,
        "constructor_kwargs": constructor_kwargs,
    }
    if extra_meta:
        sidecar.update(extra_meta)

    with open(_sidecar_path(checkpoint_path), "w") as f:
        json.dump(sidecar, f, indent=2)


def load_checkpoint(checkpoint_path: str, map_location="cpu") -> torch.nn.Module:
    """
    Reconstruct a model from its class registry + constructor kwargs (from the
    JSON sidecar), then load the state_dict (weights_only=True — safe, it's
    just tensors). Restores any extra serving attributes (train_mean/std,
    data contract metadata) that don't live in the state_dict.
    """
    sidecar_path = _sidecar_path(checkpoint_path)
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(
            f"No sidecar metadata found at {sidecar_path}. This checkpoint was likely saved with the "
            f"legacy torch.save(model, ...) full-module format and is incompatible with load_checkpoint. "
            f"Retrain to regenerate it in the new state_dict + sidecar format."
        )

    with open(sidecar_path, "r") as f:
        sidecar = json.load(f)

    model_class_name = sidecar["model_class"]
    if model_class_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class '{model_class_name}' in {sidecar_path}. Known: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_class_name]
    model = model_cls(**sidecar["constructor_kwargs"])

    state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)

    for key in _EXTRA_ATTR_KEYS:
        if key in sidecar:
            setattr(model, key, sidecar[key])

    model.to(map_location)
    model.eval()
    return model
