#!/usr/bin/env python3
"""Plot a customizable error comparison chart for different models.

Examples
--------
Plot directly from command-line values:
    python3 plot_model_errors.py \
        --models NEP DeepMD SUS2 \
        --values 7.8 7.6 3.6 \
        --ylabel "Energy MAE (meV/atom)" \
        --title "Example Dataset" \
        --output energy_mae.png

Plot from a JSON file:
    python3 plot_model_errors.py --input model_errors.json --output energy_mae.png

Supported JSON shapes:
    {
      "title": "Example Dataset",
      "ylabel": "Energy MAE (meV/atom)",
      "models": [
        {"name": "NEP", "value": 7.8, "color": "#7AA6C2"},
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

DEFAULT_COLORS = [
    "#7AA6C2",
    "#B8D8BA",
    "#F2B880",
    "#D98C95",
    "#8E9AAF",
    "#C9ADA7",
    "#9CC5A1",
    "#E4C1F9",
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
        "--title",
        default=None,
        help="Figure title. Default: Model Error Comparison",
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
        help="Output image path. Default: model_errors.png",
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
        help="Sort models by error value. Default: none.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI. Default: 300.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(8.0, 5.0),
        help="Figure size in inches. Default: 8 5.",
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


def assign_default_colors(records: list[Record]) -> list[str]:
    assigned: list[str] = []
    for index, record in enumerate(records):
        assigned.append(record.color or DEFAULT_COLORS[index % len(DEFAULT_COLORS)])
    return assigned


def format_value(value: float, value_format: str, suffix: str) -> str:
    return f"{format(value, value_format)}{suffix}"


def resolve_text(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, str]:
    style = args.style
    title = args.title or metadata.get("title") or "Model Error Comparison"
    output = args.output or metadata.get("output") or "model_errors.png"

    if style == "bar":
        xlabel = args.xlabel or metadata.get("xlabel") or "Model"
        ylabel = args.ylabel or metadata.get("ylabel") or "Error"
    else:
        xlabel = args.xlabel or metadata.get("xlabel") or "Error"
        ylabel = args.ylabel or metadata.get("ylabel") or "Model"

    return {
        "title": str(title),
        "xlabel": str(xlabel),
        "ylabel": str(ylabel),
        "output": str(output),
    }


def plot_records(records: list[Record], args: argparse.Namespace, labels: dict[str, str]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = assign_default_colors(records)
    names = [record.name for record in records]
    values = [record.value for record in records]
    best_index = min(range(len(records)), key=lambda i: values[i]) if records else None

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    if args.style == "bar":
        bars = ax.bar(names, values, color=colors, edgecolor="#2F3E46", linewidth=0.8)
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=0)
    else:
        bars = ax.barh(names, values, color=colors, edgecolor="#2F3E46", linewidth=0.8)
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    ax.set_title(labels["title"])

    if args.highlight_best and best_index is not None:
        bars[best_index].set_edgecolor("#B22222")
        bars[best_index].set_linewidth(2.0)
        bars[best_index].set_alpha(0.95)

    if not args.no_annotate:
        max_value = max(values) if values else 0.0
        offset = max(max_value * 0.02, 0.05)
        for index, (bar, value) in enumerate(zip(bars, values)):
            label = format_value(value, args.value_format, args.annotation_suffix)
            if args.highlight_best and index == best_index:
                label += " (best)"
            if args.style == "bar":
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            else:
                ax.text(
                    bar.get_width() + offset,
                    bar.get_y() + bar.get_height() / 2.0,
                    label,
                    ha="left",
                    va="center",
                    fontsize=10,
                )

    output_path = Path(labels["output"]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.tight_layout:
        fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
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
