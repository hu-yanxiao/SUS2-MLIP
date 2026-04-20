#!/usr/bin/env python3
"""Plot publication-style error charts for different models.

Examples
--------
Single dataset:
    python3 plot_model_errors.py \
        --models NEP DeepMD SUS2 \
        --values 7.8 7.6 3.6 \
        --ylabel "Energy MAE (meV/atom)" \
        --output energy_mae.pdf

Grouped bar chart with multiple datasets:
    python3 plot_model_errors.py \
        --datasets SiC GaN Al2O3 \
        --models NEP DeepMD SUS2 \
        --value-matrix 7.8,7.6,3.6 8.4,8.1,4.2 6.9,6.5,3.1 \
        --xlabel "" \
        --ylabel "Energy MAE (meV/atom)" \
        --output grouped_energy_mae.pdf

Grouped JSON shape:
    {
      "ylabel": "Energy MAE (meV/atom)",
      "datasets": [
        {
          "name": "SiC",
          "models": [
            {"name": "NEP", "value": 7.8},
            {"name": "DeepMD", "value": 7.6},
            {"name": "SUS2", "value": 3.6}
          ]
        },
        {
          "name": "GaN",
          "models": [
            {"name": "NEP", "value": 8.4},
            {"name": "DeepMD", "value": 8.1},
            {"name": "SUS2", "value": 4.2}
          ]
        }
      ]
    }

Grouped CSV columns:
    dataset,model,value,color
    SiC,NEP,7.8,#A9BDD6
    SiC,DeepMD,7.6,#C8D6C1
    SiC,SUS2,3.6,#EBC3A8
    GaN,NEP,8.4,#A9BDD6
    GaN,DeepMD,8.1,#C8D6C1
    GaN,SUS2,4.2,#EBC3A8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_FIGSIZE = (3.4, 2.8)

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


@dataclass
class GroupedDataset:
    name: str
    records: list[Record]


@dataclass
class PlotPayload:
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)
    records: list[Record] = field(default_factory=list)
    datasets: list[GroupedDataset] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot publication-style model error charts. "
            "Supports single-dataset bars and grouped bars for multiple datasets."
        )
    )
    parser.add_argument(
        "--input",
        help="Optional JSON or CSV file containing chart data.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names for grouped bar charts, for example: --datasets SiC GaN Al2O3",
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
        help="Values for a single dataset, aligned with --models.",
    )
    parser.add_argument(
        "--value-matrix",
        nargs="+",
        help=(
            "Grouped values aligned with --datasets. Each item is one comma-separated row "
            "aligned with --models, for example: --value-matrix 7.8,7.6,3.6 8.4,8.1,4.2"
        ),
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        help=(
            "Optional colors. For single-dataset mode, provide one color for all bars or one per model. "
            "For grouped mode, provide one color for all models or one color per model."
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
        help="X-axis label. Defaults depend on chart mode.",
    )
    parser.add_argument(
        "--ylabel",
        default=None,
        help="Y-axis label. Defaults depend on chart mode.",
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
        help="Chart style. Grouped plots currently support only bar. Default: bar.",
    )
    parser.add_argument(
        "--ylog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a logarithmic y-axis. Supported for vertical bar charts. Default: disabled.",
    )
    parser.add_argument(
        "--sort",
        choices=("none", "asc", "desc"),
        default="none",
        help="Sort bars by value in single-dataset mode. Default: none, which preserves input order.",
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
        default=DEFAULT_FIGSIZE,
        help="Figure size in inches. Default: 3.4 2.8.",
    )
    parser.add_argument(
        "--bar-width",
        type=float,
        default=0.62,
        help="Bar width in single-dataset mode. Default: 0.62.",
    )
    parser.add_argument(
        "--group-width",
        type=float,
        default=0.82,
        help="Total width occupied by one dataset group in grouped mode. Default: 0.82.",
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
        help="Highlight the lowest-error bar. In grouped mode, applied within each dataset. Default: enabled.",
    )
    parser.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show legend in grouped mode. Default: enabled.",
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=None,
        help="Legend column count in grouped mode. Default: min(number of models, 4).",
    )
    parser.add_argument(
        "--tight-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tight_layout before saving. Default: enabled.",
    )
    return parser.parse_args()


def normalize_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().lower(): value for key, value in mapping.items()}


def get_first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def current_palette(theme: str) -> list[str]:
    return NATURE_COLORS if theme == "nature" else CLASSIC_COLORS


def parse_raw_records(raw_records: list[dict[str, Any]]) -> list[Record]:
    records: list[Record] = []
    for index, item in enumerate(raw_records, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} is not a JSON object / CSV row.")

        normalized = normalize_mapping(item)
        name = get_first(normalized, "name", "model", "label")
        raw_value = get_first(normalized, "value", "error", "mae")
        color = get_first(normalized, "color")

        if name is None or raw_value is None:
            raise ValueError(
                f"Record {index} must contain a model name and numeric value "
                "(accepted keys: name/model/label and value/error/mae)."
            )

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Record {index} value is not numeric: {raw_value}") from exc

        records.append(Record(str(name), value, str(color) if color else None))

    return records


def validate_grouped_datasets(datasets: list[GroupedDataset]) -> None:
    if not datasets:
        raise ValueError("Grouped input must contain at least one dataset.")

    reference_models = [record.name for record in datasets[0].records]
    if not reference_models:
        raise ValueError("Grouped input datasets must contain at least one model.")

    if len(set(reference_models)) != len(reference_models):
        raise ValueError("Model names must be unique within each grouped dataset.")

    for dataset in datasets[1:]:
        model_names = [record.name for record in dataset.records]
        if model_names != reference_models:
            raise ValueError(
                "All grouped datasets must contain the same models in the same input order."
            )


def parse_raw_grouped_datasets(raw_datasets: list[dict[str, Any]]) -> list[GroupedDataset]:
    datasets: list[GroupedDataset] = []
    for index, item in enumerate(raw_datasets, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset {index} is not a JSON object.")

        normalized = normalize_mapping(item)
        dataset_name = get_first(normalized, "name", "dataset", "group", "label")
        raw_records = get_first(normalized, "models", "records", "data", "items")
        if dataset_name is None:
            raise ValueError(f"Dataset {index} is missing a dataset name.")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(
                f"Dataset {index} must contain a non-empty list under models/records/data/items."
            )

        datasets.append(GroupedDataset(str(dataset_name), parse_raw_records(raw_records)))

    validate_grouped_datasets(datasets)
    return datasets


def load_payload_from_json(path: Path) -> PlotPayload:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            probe = normalize_mapping(payload[0])
            if get_first(probe, "models", "records", "data", "items") is not None:
                return PlotPayload(mode="grouped", datasets=parse_raw_grouped_datasets(payload))
        return PlotPayload(mode="single", records=parse_raw_records(payload))

    if not isinstance(payload, dict):
        raise ValueError("JSON input must be an object or a list.")

    normalized_top = normalize_mapping(payload)
    grouped_raw = get_first(normalized_top, "datasets", "groups", "series")
    excluded = {"models", "records", "data", "items", "datasets", "groups", "series"}
    metadata = {
        str(key): value
        for key, value in payload.items()
        if str(key).strip().lower() not in excluded
    }

    if grouped_raw is not None:
        if not isinstance(grouped_raw, list):
            raise ValueError("The datasets/groups/series field must be a list.")
        return PlotPayload(
            mode="grouped",
            metadata=metadata,
            datasets=parse_raw_grouped_datasets(grouped_raw),
        )

    raw_records = get_first(normalized_top, "models", "records", "data", "items")
    if raw_records is None:
        raise ValueError(
            "JSON input must contain grouped datasets or a record list under models/records/data/items."
        )
    if not isinstance(raw_records, list):
        raise ValueError("The models/records/data/items field must be a list.")

    return PlotPayload(
        mode="single",
        metadata=metadata,
        records=parse_raw_records(raw_records),
    )


def load_payload_from_csv(path: Path) -> PlotPayload:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [normalize_mapping(row) for row in reader]

    if not rows:
        raise ValueError("CSV input is empty.")

    if any(get_first(row, "dataset", "group") is not None for row in rows):
        grouped_rows: dict[str, list[dict[str, Any]]] = {}
        for index, row in enumerate(rows, start=1):
            dataset_name = get_first(row, "dataset", "group")
            if dataset_name is None:
                raise ValueError(
                    f"CSV row {index} is missing dataset/group while grouped rows are present."
                )
            grouped_rows.setdefault(str(dataset_name), []).append(row)

        datasets = [
            GroupedDataset(name=dataset_name, records=parse_raw_records(dataset_rows))
            for dataset_name, dataset_rows in grouped_rows.items()
        ]
        validate_grouped_datasets(datasets)
        return PlotPayload(mode="grouped", datasets=datasets)

    return PlotPayload(mode="single", records=parse_raw_records(rows))


def parse_matrix_row(raw_row: str, model_count: int) -> list[float]:
    cleaned = raw_row.replace(",", " ").replace(";", " ").replace("|", " ")
    tokens = [token for token in cleaned.split() if token]
    if len(tokens) != model_count:
        raise ValueError(
            f"Each --value-matrix row must contain exactly {model_count} numeric values, got {len(tokens)}."
        )

    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Failed to parse --value-matrix row: {raw_row}") from exc


def build_single_payload_from_cli(args: argparse.Namespace) -> PlotPayload:
    if not args.models or not args.values:
        raise ValueError("Provide --models and --values, or use --input.")
    if len(args.models) != len(args.values):
        raise ValueError("--models and --values must have the same length.")

    records = [Record(name, value) for name, value in zip(args.models, args.values)]
    return PlotPayload(mode="single", records=records)


def build_grouped_payload_from_cli(args: argparse.Namespace) -> PlotPayload:
    if not args.datasets or not args.models or not args.value_matrix:
        raise ValueError(
            "Grouped mode requires --datasets, --models, and --value-matrix, or use a grouped JSON/CSV file."
        )
    if len(args.datasets) != len(args.value_matrix):
        raise ValueError("--datasets and --value-matrix must have the same length.")

    matrix = [
        parse_matrix_row(raw_row, len(args.models))
        for raw_row in args.value_matrix
    ]
    datasets = [
        GroupedDataset(
            name=dataset_name,
            records=[Record(model_name, row[index]) for index, model_name in enumerate(args.models)],
        )
        for dataset_name, row in zip(args.datasets, matrix)
    ]
    validate_grouped_datasets(datasets)
    return PlotPayload(mode="grouped", datasets=datasets)


def load_plot_payload(args: argparse.Namespace) -> PlotPayload:
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        suffix = input_path.suffix.lower()
        if suffix == ".json":
            return load_payload_from_json(input_path)
        if suffix == ".csv":
            return load_payload_from_csv(input_path)
        raise ValueError("Unsupported input format. Please use a .json or .csv file.")

    if args.datasets or args.value_matrix:
        return build_grouped_payload_from_cli(args)

    return build_single_payload_from_cli(args)


def apply_cli_colors_single(records: list[Record], colors: list[str] | None) -> None:
    if not colors:
        return
    if len(colors) not in {1, len(records)}:
        raise ValueError(
            "--colors must provide either one color for all bars or one color per model."
        )

    if len(colors) == 1:
        for record in records:
            record.color = colors[0]
        return

    for record, color in zip(records, colors):
        record.color = color


def apply_cli_colors_grouped(datasets: list[GroupedDataset], colors: list[str] | None) -> None:
    if not colors:
        return

    model_count = len(datasets[0].records)
    if len(colors) not in {1, model_count}:
        raise ValueError(
            "--colors must provide either one color for all models or one color per model in grouped mode."
        )

    model_names = [record.name for record in datasets[0].records]
    if len(colors) == 1:
        color_map = {name: colors[0] for name in model_names}
    else:
        color_map = {name: color for name, color in zip(model_names, colors)}

    for dataset in datasets:
        for record in dataset.records:
            record.color = color_map[record.name]


def apply_cli_colors(payload: PlotPayload, colors: list[str] | None) -> None:
    if payload.mode == "grouped":
        apply_cli_colors_grouped(payload.datasets, colors)
    else:
        apply_cli_colors_single(payload.records, colors)


def sort_plot_payload(payload: PlotPayload, mode: str) -> None:
    if mode == "none":
        return
    if payload.mode == "grouped":
        raise ValueError("Grouped charts preserve input order. Do not use --sort with grouped input.")

    reverse = mode == "desc"
    payload.records = sorted(payload.records, key=lambda record: record.value, reverse=reverse)


def format_value(value: float, value_format: str, suffix: str) -> str:
    return f"{format(value, value_format)}{suffix}"


def ensure_positive_for_ylog(values: list[float]) -> None:
    if any(value <= 0.0 for value in values):
        raise ValueError("--ylog requires all plotted values to be positive.")


def annotation_height(value: float, additive_offset: float, ylog: bool, stagger: float = 0.0) -> float:
    if ylog:
        return value * (1.08 + stagger)
    return value + additive_offset * (1.0 + stagger)


def apply_y_axis_scale(ax, values: list[float], args: argparse.Namespace) -> None:
    if not values:
        return

    if args.ylog:
        ensure_positive_for_ylog(values)
        min_positive = min(values)
        lower = 10 ** math.floor(math.log10(min_positive))
        upper = max(values) * max(1.0 + args.padding_factor, 1.25)
        if upper <= lower:
            upper = lower * 10.0
        ax.set_yscale("log")
        ax.set_ylim(lower, upper)
        return

    ax.set_ylim(0.0, max(values) * (1.0 + args.padding_factor))


def resolve_text(args: argparse.Namespace, metadata: dict[str, Any], mode: str) -> dict[str, str]:
    title = args.title if args.title is not None else str(metadata.get("title", ""))
    output = args.output or str(metadata.get("output", "model_errors.pdf"))

    if mode == "grouped":
        default_xlabel = "Dataset"
        default_ylabel = "Error"
    elif args.style == "bar":
        default_xlabel = "Model"
        default_ylabel = "Error"
    else:
        default_xlabel = "Error"
        default_ylabel = "Model"

    xlabel = args.xlabel if args.xlabel is not None else str(metadata.get("xlabel", default_xlabel))
    ylabel = args.ylabel if args.ylabel is not None else str(metadata.get("ylabel", default_ylabel))

    return {
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "output": output,
    }


def resolve_figure_size(args: argparse.Namespace, mode: str, item_count: int) -> tuple[float, float]:
    figsize = tuple(args.figsize)
    if mode != "grouped" or figsize != DEFAULT_FIGSIZE:
        return figsize

    width = max(DEFAULT_FIGSIZE[0], 1.15 + 0.85 * item_count)
    return (width, DEFAULT_FIGSIZE[1])


def configure_matplotlib(args: argparse.Namespace) -> None:
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
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.top": False,
            "ytick.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def apply_axis_style(ax, style: str) -> None:
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#333333")
    if style == "bar":
        ax.yaxis.grid(True, linestyle=(0, (2.0, 2.0)), linewidth=0.7, color="#D5D5D5")
    else:
        ax.xaxis.grid(True, linestyle=(0, (2.0, 2.0)), linewidth=0.7, color="#D5D5D5")
    ax.tick_params(axis="x", which="both", direction="in", top=False, bottom=True)
    ax.tick_params(axis="y", which="both", direction="in", right=False, left=True)
    ax.set_axisbelow(True)


def assign_single_colors(records: list[Record], theme: str) -> list[str]:
    palette = current_palette(theme)
    colors: list[str] = []
    for index, record in enumerate(records):
        colors.append(record.color or palette[index % len(palette)])
    return colors


def assign_grouped_colors(datasets: list[GroupedDataset], theme: str) -> list[str]:
    palette = current_palette(theme)
    model_names = [record.name for record in datasets[0].records]
    explicit: dict[str, str] = {}
    for dataset in datasets:
        for record in dataset.records:
            if record.color and record.name not in explicit:
                explicit[record.name] = record.color

    return [
        explicit.get(model_name, palette[index % len(palette)])
        for index, model_name in enumerate(model_names)
    ]


def save_figure(fig, args: argparse.Namespace, output: str) -> Path:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.tight_layout:
        fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    return output_path


def plot_single_records(records: list[Record], args: argparse.Namespace, labels: dict[str, str]) -> Path:
    configure_matplotlib(args)
    import matplotlib.pyplot as plt

    colors = assign_single_colors(records, args.theme)
    names = [record.name for record in records]
    values = [record.value for record in records]
    best_index = min(range(len(records)), key=lambda index: values[index]) if records else None

    if args.ylog and args.style != "bar":
        raise ValueError("--ylog is only supported with vertical bar charts.")

    fig, ax = plt.subplots(figsize=resolve_figure_size(args, "single", len(records)))
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

    if not args.no_annotate and values:
        max_value = max(values)
        offset = max(max_value * 0.03, 0.04)
        for index, (bar, value) in enumerate(zip(bars, values)):
            label = format_value(value, args.value_format, args.annotation_suffix)
            if args.style == "bar":
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    annotation_height(value, offset, args.ylog),
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
        apply_y_axis_scale(ax, values, args)
    if args.style == "barh" and values:
        ax.set_xlim(0.0, max(values) * (1.0 + args.padding_factor))

    output_path = save_figure(fig, args, labels["output"])
    plt.close(fig)
    return output_path


def plot_grouped_datasets(
    datasets: list[GroupedDataset],
    args: argparse.Namespace,
    labels: dict[str, str],
) -> Path:
    if args.style != "bar":
        raise ValueError("Grouped charts currently support only --style bar.")
    if args.ylog:
        ensure_positive_for_ylog(
            [record.value for dataset in datasets for record in dataset.records]
        )

    configure_matplotlib(args)
    import matplotlib.pyplot as plt

    model_names = [record.name for record in datasets[0].records]
    model_colors = assign_grouped_colors(datasets, args.theme)
    dataset_names = [dataset.name for dataset in datasets]
    dataset_count = len(datasets)
    model_count = len(model_names)
    total_group_width = max(0.1, min(args.group_width, 0.95))
    bar_width = total_group_width / model_count
    centers = list(range(dataset_count))
    max_value = max(record.value for dataset in datasets for record in dataset.records)
    best_model_indices = [
        min(range(model_count), key=lambda index: dataset.records[index].value)
        for dataset in datasets
    ]

    fig, ax = plt.subplots(figsize=resolve_figure_size(args, "grouped", dataset_count))
    containers = []
    for model_index, model_name in enumerate(model_names):
        x_positions = [
            center - total_group_width / 2.0 + (model_index + 0.5) * bar_width
            for center in centers
        ]
        y_values = [dataset.records[model_index].value for dataset in datasets]
        container = ax.bar(
            x_positions,
            y_values,
            width=bar_width * 0.92,
            color=model_colors[model_index],
            edgecolor="#555555",
            linewidth=0.75,
            label=model_name,
            zorder=3,
        )
        containers.append(container)

    ax.set_xticks(centers)
    ax.set_xticklabels(dataset_names)
    ax.set_xlabel(labels["xlabel"])
    ax.set_ylabel(labels["ylabel"])
    apply_axis_style(ax, "bar")
    ax.margins(x=0.05)

    if labels["title"]:
        ax.set_title(labels["title"], pad=6.0)

    if args.highlight_best:
        for dataset_index, best_model_index in enumerate(best_model_indices):
            best_patch = containers[best_model_index].patches[dataset_index]
            best_patch.set_edgecolor("#111111")
            best_patch.set_linewidth(1.1)
            best_patch.set_alpha(1.0)

    if args.show_legend and model_count > 1:
        legend_columns = args.legend_columns or min(model_count, 4)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=legend_columns,
            frameon=False,
            columnspacing=1.0,
            handlelength=1.2,
            handletextpad=0.5,
        )

    if not args.no_annotate and max_value > 0.0:
        offset = max(max_value * 0.03, 0.04)
        for model_index, container in enumerate(containers):
            stagger = 0.15 * (model_index % 3)
            for dataset_index, patch in enumerate(container.patches):
                value = datasets[dataset_index].records[model_index].value
                ax.text(
                    patch.get_x() + patch.get_width() / 2.0,
                    annotation_height(value, offset, args.ylog, stagger),
                    format_value(value, args.value_format, args.annotation_suffix),
                    ha="center",
                    va="bottom",
                    fontsize=args.annotation_size,
                    fontweight=(
                        "bold"
                        if args.highlight_best and best_model_indices[dataset_index] == model_index
                        else "normal"
                    ),
                    color="#111111",
                )

    if max_value > 0.0:
        apply_y_axis_scale(
            ax,
            [record.value for dataset in datasets for record in dataset.records],
            args,
        )

    output_path = save_figure(fig, args, labels["output"])
    plt.close(fig)
    return output_path


def main() -> int:
    try:
        args = parse_args()
        payload = load_plot_payload(args)
        apply_cli_colors(payload, args.colors)
        sort_plot_payload(payload, args.sort)
        labels = resolve_text(args, payload.metadata, payload.mode)

        if payload.mode == "grouped":
            output_path = plot_grouped_datasets(payload.datasets, args, labels)
        else:
            output_path = plot_single_records(payload.records, args, labels)
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
