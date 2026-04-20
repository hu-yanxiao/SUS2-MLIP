#!/usr/bin/env python3
"""Plot a publication-style error comparison chart for different models.

Examples
--------
Plot directly from command-line values in input order:
    python3 plot_model_errors.py \
        --models NEP DeepMD SUS2 \
        --values 7.8 7.6 3.6 \
        --ylabel "Energy MAE (meV/atom)" \
        --output energy_mae.pdf

Plot from a JSON file:
    python3 plot_model_errors.py --input model_errors.json --output energy_mae.pdf

Supported JSON shapes:
    {
      "ylabel": "Energy MAE (meV/atom)",
      "models": [
        {"name": "NEP", "value": 7.8, "color": "#A9BDD6"},
        {"name": "DeepMD", "value": 7.6},
        {"name": "SUS2", "value": 3.6}
      ]
    }

    [
      {"name": "NEP", "value": 7.8},
      {"name": "DeepMD", "value": 7.6},
      {"name": "SUS2", "value": 3.6}
    ]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NATURE_COLORS = [
    "#A9BDD6",
    "#C8D6C1",
    "#EBC3A8",
    "#C7CDD9",
    "#D8CDBF",
    "#B8D0C9",
    "#D6C0D8",
    "#D8D8D8",
]

CLASSIC_COLORS = [
    "#7AA6C2",
    "#B8D8BA",
    "#F2B880",
    "#D98C95",
    "#8E9AAF",
    "#C9ADA7",
    "#9CC5A1",
    "#E4C1F9",
]

SERIF_FALLBACKS = [
    "Times New Roman",
    "Times New Roman PS MT",
    "Times",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
]


@dataclass
class Record:
    name: str
    value: float
    color: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a comparison chart of model errors. "
            "Data can be passed through --models/--values or loaded from JSON/CSV."
        )
    )
    parser.add_argument(
        "--input",
        help="Optional JSON or CSV file containing model-error data.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model names, for example: --models NEP DeepMD SUS2",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        help="Error values matching --models, for example: --values 7.8 7.6 3.6",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        help=(
            "Optional bar colors. Provide either one color for all models or one color per model. "
            "Hex codes such as #3A7CA5 are recommended."
        ),
    )
    parser.add_argument(
        "--theme",
        choices=("nature", "default"),
        default="nature",
        help="Visual theme. Default: nature.",
    )
    parser.add_argument(
        "--font-family",
        default="Times New Roman",
        help="Primary serif font family. Default: Times New Roman.",
    )
    parser.add_argument(
        "--base-font-size",
        type=float,
        default=9.0,
        help="Base font size in points. Default: 9.",
    )
    parser.add_argument(
        "--label-size",
        type=float,
        default=10.0,
        help="Axis-label and title size in points. Default: 10.",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=9.0,
        help="Tick-label size in points. Default: 9.",
    )
    parser.add_argument(
        "--annotation-size",
        type=float,
        default=8.5,
        help="Value-label size in points. Default: 8.5.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title. Default: none for publication-style output.",
    )
    parser.add_argument(
        "--xlabel",
        default=None,
        help="X-axis label. Default: Model for vertical bars, Error for horizontal bars.",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Y-axis label. Default: Error for vertical bars, Model for horizontal bars.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. PDF/SVG are recommended for publication. Default: model_errors.pdf",
    )
    parser.add_argument(
        "--style",
        choices=("bar", "barh"),
        default="bar",
        help="Chart style: vertical bars (bar) or horizontal bars (barh). Default: bar.",
    )
    parser.add_argument(
        "--sort",
        choices=("none", "asc", "desc"),
        default="none",
        help="Sort models by error value. Default: none, which preserves the input order.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output DPI for raster export. Default: 600.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(3.4, 2.8),
        help="Figure size in inches. Default: 3.4 2.8 for a compact single-column figure.",
    )
    parser.add_argument(
        "--bar-width",
        type=float,
        default=0.62,
        help="Bar width. Default: 0.62.",
    )
    parser.add_argument(
        "--padding-factor",
        type=float,
        default=0.14,
        help="Extra headroom fraction added for annotations. Default: 0.14.",
    )
    parser.add_argument(
        "--value-format",
        default=".1f",
        help="Python format specifier for value labels. Default: .1f",
    )
    parser.add_argument(
        "--annotation-suffix",
        default="",
        help="Suffix appended to numeric annotations, for example: ' meV/atom'.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable value labels on bars.",
    )
    parser.add_argument(
        "--highlight-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Highlight the lowest-error model. Default: enabled.",
    )
    parser.add_argument(
        "--tight-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tight_layout before saving. Default: enabled.",
    )
    return parser.parse_args()


def load_records_from_json(path: Path) -> tuple[list[Record], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {}

    if isinstance(payload, dict):
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"models", "records", "data", "items"}
        }
        raw_records = (
            payload.get("models")
            or payload.get("records")
            or payload.get("data")
            or payload.get("items")
        )
    else:
        raw_records = payload

    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("JSON input must contain a non-empty list of records.")

    return parse_raw_records(raw_records), metadata


def load_records_from_csv(path: Path) -> tuple[list[Record], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV input is empty.")

    return parse_raw_records(rows), {}


def parse_raw_records(raw_records: list[dict[str, Any]]) -> list[Record]:
    records: list[Record] = []
    for index, item in enumerate(raw_records, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} is not a JSON object / CSV row.")

        name = item.get("name") or item.get("model") or item.get("label")
        raw_value = item.get("value")
        if raw_value is None:
            raw_value = item.get("error")
        if raw_value is None:
            raw_value = item.get("mae")

        if name is None or raw_value is None:
            raise ValueError(
                f"Record {index} must contain a model name and numeric value "
                "(accepted keys: name/model/label and value/error/mae)."
            )

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Record {index} value is not numeric: {raw_value}") from exc

        color = item.get("color")
        records.append(Record(str(name), value, str(color) if color else None))

    return records


def load_records(args: argparse.Namespace) -> tuple[list[Record], dict[str, Any]]:
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        suffix = input_path.suffix.lower()
        if suffix == ".json":
            return load_records_from_json(input_path)
        if suffix == ".csv":
            return load_records_from_csv(input_path)
        raise ValueError("Unsupported input format. Please use a .json or .csv file.")

    if not args.models or not args.values:
        raise ValueError("Provide --models and --values, or use --input.")
    if len(args.models) != len(args.values):
        raise ValueError("--models and --values must have the same length.")

    records = [Record(name, value) for name, value in zip(args.models, args.values)]
    return records, {}


def apply_cli_colors(records: list[Record], colors: list[str] | None) -> None:
    if not colors:
        return
    if len(colors) not in {1, len(records)}:
        raise ValueError(
            "--colors must provide either one color for all models or one color per model."
        )

    if len(colors) == 1:
        for record in records:
            record.color = colors[0]
        return

    for record, color in zip(records, colors):
        record.color = color


def sort_records(records: list[Record], mode: str) -> list[Record]:
    if mode == "none":
        return list(records)
    reverse = mode == "desc"
    return sorted(records, key=lambda record: record.value, reverse=reverse)


def assign_default_colors(records: list[Record], theme: str) -> list[str]:
    palette = NATURE_COLORS if theme == "nature" else CLASSIC_COLORS
    assigned: list[str] = []
    for index, record in enumerate(records):
        assigned.append(record.color or palette[index % len(palette)])
    return assigned


def format_value(value: float, value_format: str, suffix: str) -> str:
    return f"{format(value, value_format)}{suffix}"


def resolve_text(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, str]:
    style = args.style
    title = args.title if args.title is not None else metadata.get("title", "")
    output = args.output or metadata.get("output") or "model_errors.pdf"

    if style == "bar":
        xlabel = args.xlabel if args.xlabel is not None else metadata.get("xlabel", "Model")
        ylabel = args.ylabel if args.ylabel is not None else metadata.get("ylabel", "Error")
    else:
        xlabel = args.xlabel if args.xlabel is not None else metadata.get("xlabel", "Error")
        ylabel = args.ylabel if args.ylabel is not None else metadata.get("ylabel", "Model")

    return {
        "title": str(title),
        "xlabel": str(xlabel),
        "ylabel": str(ylabel),
        "output": str(output),
    }


def configure_matplotlib(args: argparse.Namespace):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "font.family": "serif",
            "font.serif": [args.font_family]
            + [font for font in SERIF_FALLBACKS if font != args.font_family],
            "font.size": args.base_font_size,
            "axes.labelsize": args.label_size,
            "axes.titlesize": args.label_size,
            "xtick.labelsize": args.tick_size,
            "ytick.labelsize": args.tick_size,
            "legend.fontsize": args.base_font_size,
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def apply_axis_style(ax, style: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    if style == "bar":
        ax.yaxis.grid(True, linestyle=(0, (2.0, 2.0)), linewidth=0.7, color="#D5D5D5")
    else:
        ax.xaxis.grid(True, linestyle=(0, (2.0, 2.0)), linewidth=0.7, color="#D5D5D5")
    ax.set_axisbelow(True)


def plot_records(records: list[Record], args: argparse.Namespace, labels: dict[str, str]) -> Path:
    configure_matplotlib(args)
    import matplotlib.pyplot as plt

    colors = assign_default_colors(records, args.theme)
    names = [record.name for record in records]
    values = [record.value for record in records]
    best_index = min(range(len(records)), key=lambda i: values[i]) if records else None

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    if args.style == "bar":
        bars = ax.bar(
            names,
            values,
            color=colors,
            edgecolor="#555555",
            linewidth=0.75,
            width=args.bar_width,
            zorder=3,
        )
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.tick_params(axis="x", rotation=0)
    else:
        bars = ax.barh(
            names,
            values,
            color=colors,
            edgecolor="#555555",
            linewidth=0.75,
            height=args.bar_width,
            zorder=3,
        )
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
    apply_axis_style(ax, args.style)

    if labels["title"]:
        ax.set_title(labels["title"], pad=6.0)

    if args.highlight_best and best_index is not None:
        bars[best_index].set_edgecolor("#111111")
        bars[best_index].set_linewidth(1.1)
        bars[best_index].set_alpha(1.0)

    if not args.no_annotate:
        max_value = max(values) if values else 0.0
        offset = max(max_value * 0.03, 0.04)
        for index, (bar, value) in enumerate(zip(bars, values)):
            label = format_value(value, args.value_format, args.annotation_suffix)
            if args.style == "bar":
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=args.annotation_size,
                    fontweight="bold" if args.highlight_best and index == best_index else "normal",
                    color="#111111",
                )
            else:
                ax.text(
                    bar.get_width() + offset,
                    bar.get_y() + bar.get_height() / 2.0,
                    label,
                    ha="left",
                    va="center",
                    fontsize=args.annotation_size,
                    fontweight="bold" if args.highlight_best and index == best_index else "normal",
                    color="#111111",
                )

    if args.style == "bar" and values:
        upper = max(values) * (1.0 + args.padding_factor)
        ax.set_ylim(0.0, upper)
    if args.style == "barh" and values:
        upper = max(values) * (1.0 + args.padding_factor)
        ax.set_xlim(0.0, upper)

    output_path = Path(labels["output"]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.tight_layout:
        fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def main() -> int:
    try:
        args = parse_args()
        records, metadata = load_records(args)
        apply_cli_colors(records, args.colors)
        records = sort_records(records, args.sort)
        labels = resolve_text(args, metadata)
        output_path = plot_records(records, args, labels)
    except ImportError as exc:
        print("This script requires matplotlib. Install it with: pip install matplotlib", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Failed to parse JSON input: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved figure to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
