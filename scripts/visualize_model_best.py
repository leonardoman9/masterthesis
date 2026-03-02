#!/usr/bin/env python3
"""
Create a readable model architecture image for thesis documentation.

Priority:
1) torchview (if installed): module-level graph.
2) Fallback: Graphviz block diagram generated from the instantiated model.

Also writes an optional torchinfo summary text file.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import types
from pathlib import Path


def _inject_optional_stubs() -> None:
    """Provide tiny stubs for optional imports used only in helper code."""
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
    p = argparse.ArgumentParser(description="Generate readable model architecture image")
    p.add_argument(
        "--model-root",
        type=Path,
        default=Path("../backups/rf4422"),
        help="Path containing models.py, modules.py, differentiable_spec_torch.py",
    )
    p.add_argument("--model-module", type=str, default="models")
    p.add_argument("--model-class", type=str, default="Improved_Phi_GRU_ATT")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("images/model_architecture_best"),
        help="Output path without extension",
    )
    p.add_argument("--format", choices=["png", "svg", "pdf"], default="png")
    p.add_argument("--sample-rate", type=int, default=32000)
    p.add_argument("--clip-seconds", type=float, default=1.0)
    p.add_argument("--num-classes", type=int, default=71)
    p.add_argument(
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
    p.add_argument("--n-linear-filters", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=320)
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument(
        "--summary-output",
        type=Path,
        default=Path("images/model_architecture_summary.txt"),
        help="Optional torchinfo summary output file",
    )
    p.add_argument(
        "--no-torchview",
        action="store_true",
        help="Skip torchview even if installed and directly generate block diagram",
    )
    p.add_argument(
        "--torchview-depth",
        type=int,
        default=3,
        help="Depth used by torchview when available",
    )
    return p.parse_args()


def count_trainable_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def safe_node_id(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def build_block_diagram_dot(model, args: argparse.Namespace) -> str:
    total_params = count_trainable_params(model)

    frontend_name = f"Spectrogram Front-end ({model.spectrogram_type})"
    frontend_params = 0
    if getattr(model, "combined_log_linear_spec", None) is not None:
        frontend_params = count_trainable_params(model.combined_log_linear_spec)
    elif getattr(model, "mel_transform", None) is not None:
        frontend_params = 0
    elif getattr(model, "stft_transform", None) is not None:
        frontend_params = 0

    blocks = [
        ("input", f"Input waveform\\n[1, {int(args.sample_rate * args.clip_seconds)}]"),
        ("front", f"{frontend_name}\\nparams: {frontend_params:,}"),
        ("dbnorm", "AmplitudeToDB + sample-wise normalization\\nparams: 0"),
        ("phi", f"MatchboxNetSkip encoder\\nparams: {count_trainable_params(model.phi):,}"),
        ("gru", f"GRU temporal layer\\nparams: {count_trainable_params(model.gru):,}"),
        ("proj", f"Projection Linear\\nparams: {count_trainable_params(model.projection):,}"),
        ("att", f"AttentionLayer\\nparams: {count_trainable_params(model.keyword_attention):,}"),
        ("fc", f"Classifier FC ({args.num_classes} classes)\\nparams: {count_trainable_params(model.fc):,}"),
        ("out", "Output logits"),
    ]

    lines = []
    lines.append("digraph WrenNetArchitecture {")
    lines.append("  rankdir=LR;")
    lines.append('  graph [fontname="Helvetica", fontsize=14, label="WrenNet Student Architecture (Readable View)", labelloc=t, ranksep=1.1, nodesep=0.55];')
    lines.append('  node [shape=box, style="rounded,filled", fillcolor="#EEF6FF", color="#2B4C7E", fontname="Helvetica", fontsize=11, penwidth=1.1, margin="0.12,0.08"];')
    lines.append('  edge [color="#4D4D4D", arrowsize=0.8, penwidth=1.0];')
    lines.append(
        '  model_meta [shape=note, fillcolor="#FFF7E6", color="#A67C00", label="Model: '
        + f'{args.model_class}\\nTotal trainable params: {total_params:,}\\n'
        + f'Sample rate: {args.sample_rate} Hz\\nFront-end: {args.spectrogram_type}"];'
    )

    for block_id, label in blocks:
        nid = safe_node_id(block_id)
        lines.append(f'  {nid} [label="{label}"];')

    for i in range(len(blocks) - 1):
        src = safe_node_id(blocks[i][0])
        dst = safe_node_id(blocks[i + 1][0])
        lines.append(f"  {src} -> {dst};")

    lines.append("  model_meta -> front [style=dashed, color=\"#A67C00\"];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def try_torchview(model, x, out_no_ext: Path, fmt: str, depth: int) -> Path | None:
    try:
        from torchview import draw_graph
    except ModuleNotFoundError:
        return None

    graph = draw_graph(
        model,
        input_data=x,
        expand_nested=False,
        depth=depth,
        graph_name="WrenNet Student Architecture",
        roll=False,
    )
    gv = graph.visual_graph
    gv.format = fmt
    out_no_ext.parent.mkdir(parents=True, exist_ok=True)
    rendered = Path(gv.render(filename=out_no_ext.name, directory=str(out_no_ext.parent), cleanup=True))
    return rendered


def write_torchinfo_summary(model, x, out_path: Path) -> None:
    try:
        from torchinfo import summary
    except ModuleNotFoundError:
        return

    info = summary(
        model,
        input_data=x,
        depth=3,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        verbose=0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(str(info) + "\n", encoding="utf-8")


def render_block_diagram(model, args: argparse.Namespace, out_no_ext: Path) -> Path:
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        raise RuntimeError("Graphviz binary 'dot' not found in PATH.")

    dot_text = build_block_diagram_dot(model, args)
    out_no_ext.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_no_ext.with_suffix(".dot")
    dot_path.write_text(dot_text, encoding="utf-8")

    rendered = out_no_ext.with_suffix("." + args.format)
    cmd = [dot_binary, f"-T{args.format}", str(dot_path), "-o", str(rendered)]
    subprocess.run(cmd, check=True)
    return rendered


def main() -> int:
    args = parse_args()
    model_root = args.model_root.resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")

    _inject_optional_stubs()

    try:
        import torch
    except ModuleNotFoundError:
        print("Missing dependency: torch")
        return 1

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
    x = torch.randn(1, n_samples)

    if args.summary_output:
        try:
            write_torchinfo_summary(model, x, args.summary_output)
            print(f"Summary written to: {args.summary_output}")
        except Exception as exc:
            print(f"Skipping summary generation: {exc}")

    out_no_ext = args.output
    if not args.no_torchview:
        try:
            rendered = try_torchview(model, x, out_no_ext, args.format, args.torchview_depth)
            if rendered is not None:
                print(f"Architecture image written with torchview: {rendered}")
                return 0
            print("torchview not available, using Graphviz block diagram fallback.")
        except Exception as exc:
            print(f"torchview failed, using Graphviz block diagram fallback: {exc}")

    rendered = render_block_diagram(model, args, out_no_ext)
    print(f"Architecture image written (block diagram): {rendered}")
    print(f"DOT source written to: {out_no_ext.with_suffix('.dot')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
