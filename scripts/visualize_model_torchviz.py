#!/usr/bin/env python3
"""
Generate a TorchViz graph for the thesis student model (Improved_Phi_GRU_ATT).

Default target:
  /Users/leonardomannini/Progetti/tesi/backups/rf4422/models.py
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path


def _inject_optional_stubs() -> None:
    """Provide tiny stubs for optional imports used only for typing/helpers."""
    try:
        import omegaconf  # noqa: F401
    except ModuleNotFoundError:
        m = types.ModuleType("omegaconf")

        class DictConfig(dict):
            pass

        m.DictConfig = DictConfig
        sys.modules["omegaconf"] = m

    try:
        import torchsummary  # noqa: F401
    except ModuleNotFoundError:
        m = types.ModuleType("torchsummary")

        def summary(*_args, **_kwargs):
            return None

        m.summary = summary
        sys.modules["torchsummary"] = m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize thesis model with torchviz")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("../backups/rf4422"),
        help="Path containing models.py, modules.py, differentiable_spec_torch.py",
    )
    parser.add_argument(
        "--model-module",
        type=str,
        default="models",
        help="Python module containing the model class (e.g. models or birds_distillation_edge.models.legacy_models)",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default="Improved_Phi_GRU_ATT",
        help="Model class name to instantiate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("images/model_graph_torchviz"),
        help="Output path without extension (e.g. images/model_graph_torchviz)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Graph output format",
    )
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--num-classes", type=int, default=71)  # 70(+1)
    parser.add_argument(
        "--spectrogram-type",
        choices=[
            "mel",
            "linear_stft",
            "linear_triangular",
            "combined_log_linear",
            "fully_learnable",
        ],
        default="combined_log_linear",
    )
    parser.add_argument("--n-linear-filters", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=320)
    parser.add_argument("--base-filters", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = args.model_root.resolve()

    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")

    _inject_optional_stubs()

    # Hard dependency check (runtime, not stubs).
    try:
        import torch
        from torchviz import make_dot
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        print(f"Missing dependency: {missing}")
        print("Install in the active environment, then rerun:")
        print("  python3 -m pip install torch torchaudio torchviz")
        print("Graphviz binary `dot` is also required.")
        return 1

    # Ensure local imports resolve to rf4422 implementation.
    sys.path.insert(0, str(model_root))

    try:
        model_module = importlib.import_module(args.model_module)
        model_cls = getattr(model_module, args.model_class)
    except Exception as exc:
        print(f"Cannot import model class {args.model_class} from {args.model_module}")
        print(f"Model root on sys.path: {model_root}")
        print(f"Import error: {exc}")
        return 1

    model = model_cls(
        num_classes=args.num_classes,
        spectrogram_type=args.spectrogram_type,
        sample_rate=args.sample_rate,
        n_linear_filters=args.n_linear_filters,
        hidden_dim=args.hidden_dim,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        matchbox={"base_filters": args.base_filters},
    )
    model.eval()

    n_samples = int(args.sample_rate * args.clip_seconds)
    x = torch.randn(1, n_samples, requires_grad=True)
    y = model(x)

    dot = make_dot(
        y,
        params=dict(model.named_parameters()),
    )
    dot.format = args.format

    out_no_ext = args.output
    out_no_ext.parent.mkdir(parents=True, exist_ok=True)
    rendered = Path(dot.render(filename=out_no_ext.name, directory=str(out_no_ext.parent), cleanup=True))
    print(f"Graph written to: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
