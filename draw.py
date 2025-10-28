#!/usr/bin/env python3
"""
Draw the TCN model structure using torchview (Graphviz).

Usage:
  python -m n2n.draw --out tcn_structure --directory . --length 1000 --format png

Requires:
  pip install torchview graphviz
  (and Graphviz system package: e.g., apt-get install graphviz)
"""

import argparse
import os
import torch
from torchinfo import summary

try:
    from torchview import draw_graph
except Exception as e:
    raise RuntimeError(
        "torchview is required. Install with: pip install torchview graphviz"
    ) from e

try:
    from n2n.tcn import Noise2Noise1DTCN
except Exception as e:
    # Fallback import path if package layout differs
    from .tcn import Noise2Noise1DTCN  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw TCN model graph with torchview")
    p.add_argument("--out", default="tcn_structure", help="Output filename (without extension)")
    p.add_argument("--directory", default=".", help="Directory to save the image")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output image format")
    p.add_argument("--length", type=int, default=1000, help="Example input length L for the 1D waveform")
    p.add_argument("--in_channels", type=int, default=1, help="Number of input channels (C)")
    p.add_argument("--kernel_size", type=int, default=3, help="Kernel size for TCN")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout for TCN")
    p.add_argument("--block_index", type=int, default=None, help="If set, visualize only this TemporalBlock (0-based index)")
    # Visualization controls
    p.add_argument("--expand_nested", action="store_true", help="Expand nested modules (e.g., show TemporalBlock internals)")
    p.add_argument("--depth", type=int, default=6, help="Module nesting depth to visualize")
    p.add_argument("--graph_dir", default="TB", choices=["TB", "LR", "BT", "RL"], help="Graph layout direction")
    p.add_argument("--show_module_functions", action="store_true", help="Show module functions (functional ops) in graph")
    p.add_argument("--show_inner_tensors", action="store_true", help="Show inner tensors and intermediate edges")
    p.add_argument("--no_shapes", action="store_true", help="Hide tensor shapes on edges")
    p.add_argument("--collect_attributes", action="store_true", help="Collect and display module attributes (e.g., dilation, padding)")
    # Graphviz layout tuning
    p.add_argument("--ranksep", type=float, default=None, help="Vertical separation between ranks (inches)")
    p.add_argument("--nodesep", type=float, default=None, help="Separation between nodes on the same rank (inches)")
    p.add_argument("--ratio", default=None, choices=["auto", "compress", "expand", "fill"], help="Graph aspect ratio hint")
    p.add_argument("--size", default=None, help="Graph size WxH, e.g., '10,14!' for fixed page")
    p.add_argument("--dpi", type=int, default=None, help="Dots-per-inch for rendered image")
    p.add_argument("--splines", default=None, choices=["spline", "polyline", "curved", "ortho", "line"], help="Edge routing style")
    p.add_argument("--concentrate", action="store_true", help="Merge parallel edges when possible")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.directory, exist_ok=True)

    model = Noise2Noise1DTCN(
        in_channels=args.in_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).eval()

    if args.block_index is None:
        # Full model view
        summary(model, input_size=(1,1,int(args.length)))
        example_input = (torch.randn(1, args.in_channels, int(args.length)),)
        target_module = model
        graph_name = "tcn"
    else:
        # Single TemporalBlock view
        try:
            target_module = model.tcn.network[args.block_index]
        except Exception as e:
            raise ValueError(f"Invalid --block_index={args.block_index}. Available range: 0..{len(model.tcn.network)-1}") from e
        # Infer required input channels from the block itself
        try:
            block_in_channels = int(getattr(target_module.conv1, 'in_channels'))
        except Exception:
            block_in_channels = args.in_channels
        example_input = (torch.randn(1, block_in_channels, int(args.length)),)
        graph_name = f"tcn_block_{args.block_index}"

    # Some torchview versions forward unknown kwargs to model.forward; avoid passing 'format' here.
    g = draw_graph(
        target_module,
        input_data=example_input,
        graph_name=graph_name,
        save_graph=False,
        expand_nested=args.expand_nested,
        depth=args.depth,
        graph_dir=args.graph_dir,
        hide_module_functions=(not args.show_module_functions),
        hide_inner_tensors=(not args.show_inner_tensors),
        show_shapes=(not args.no_shapes),
        collect_attributes=args.collect_attributes,
    )
    # Apply additional Graphviz layout attributes for compactness/orientation
    gv = g.visual_graph
    if args.ranksep is not None:
        gv.graph_attr["ranksep"] = str(args.ranksep)
    if args.nodesep is not None:
        gv.graph_attr["nodesep"] = str(args.nodesep)
    if args.ratio is not None:
        gv.graph_attr["ratio"] = args.ratio
    if args.size is not None:
        gv.graph_attr["size"] = args.size
    if args.dpi is not None:
        gv.graph_attr["dpi"] = str(args.dpi)
    if args.splines is not None:
        gv.graph_attr["splines"] = args.splines
    if args.concentrate:
        gv.graph_attr["concentrate"] = "true"
    # Use graphviz render directly
    output_path = g.visual_graph.render(filename=args.out, directory=args.directory, format=args.format, cleanup=True)
    print(f"Saved model graph to {output_path}", flush=True)


if __name__ == "__main__":
    main()


