#!/usr/bin/env python3
"""Convert ASE-readable structures to a LAMMPS data file.

Examples
--------
Default type order follows atomic number:
    python ase_to_lammps_data.py 622-16ion.cif -o 622-16ion.data

Use an explicit global species set:
    python ase_to_lammps_data.py 622-16ion.cif -o 622-16ion.data \
        --species-order H,C,N,O

Read a POSCAR-like file without an extension:
    python ase_to_lammps_data.py POSCAR -o structure.data --input-format vasp
"""

from __future__ import annotations

import argparse
import io
import inspect
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a structure file readable by ASE into a LAMMPS data file. "
            "Default atom types follow atomic-number order. "
            "An explicit species list can be supplied to reserve type IDs."
        )
    )
    parser.add_argument("input", help="Input structure file, such as CIF, POSCAR, XYZ, or extxyz.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output LAMMPS data file. Defaults to <input>.data.",
    )
    parser.add_argument(
        "--input-format",
        help=(
            "Optional ASE input format, such as cif, vasp, extxyz, xyz. "
            "Useful for POSCAR-like files without a standard extension."
        ),
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Configuration index for multi-frame files. Default: 0.",
    )
    parser.add_argument(
        "--species-order",
        nargs="+",
        help=(
            "Explicit species order used for LAMMPS type IDs. "
            "Accepts space-separated or comma-separated symbols, for example "
            "'--species-order H C N O' or '--species-order H,C,N,O'."
        ),
    )
    parser.add_argument(
        "--atom-style",
        default="atomic",
        choices=("atomic", "charge", "full"),
        help="LAMMPS atom_style written into the data file. Default: atomic.",
    )
    parser.add_argument(
        "--units",
        default="metal",
        help="LAMMPS units passed to ASE when supported. Default: metal.",
    )
    parser.add_argument(
        "--reduce-cell",
        action="store_true",
        help="Ask ASE to reduce the triclinic cell before writing.",
    )
    parser.add_argument(
        "--force-skew",
        action="store_true",
        help="Force ASE to write the box as triclinic when supported.",
    )
    return parser.parse_args()


def normalize_species_order(raw_values: list[str] | None, atomic_numbers: dict[str, int]) -> list[str] | None:
    if not raw_values:
        return None

    species: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for token in raw.replace(",", " ").split():
            symbol = token.strip()
            if not symbol:
                continue
            if symbol not in atomic_numbers:
                raise ValueError(f"Unknown chemical symbol in --species-order: {symbol}")
            if symbol in seen:
                raise ValueError(f"Duplicate chemical symbol in --species-order: {symbol}")
            seen.add(symbol)
            species.append(symbol)

    if not species:
        raise ValueError("--species-order was provided but no valid symbols were parsed.")
    return species


def default_species_order(symbols: list[str], atomic_numbers: dict[str, int]) -> list[str]:
    return sorted(set(symbols), key=lambda symbol: atomic_numbers[symbol])


def infer_input_format(input_path: Path, explicit_format: str | None) -> str | None:
    if explicit_format:
        return explicit_format
    if input_path.name in {"POSCAR", "CONTCAR"}:
        return "vasp"
    return None


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_suffix(".data")
    return input_path.with_name(f"{input_path.name}.data")


def build_write_kwargs(write_lammps_data, args: argparse.Namespace, species_order: list[str]) -> dict:
    signature = inspect.signature(write_lammps_data)
    kwargs = {
        "specorder": species_order,
        "atom_style": args.atom_style,
    }

    if "masses" in signature.parameters:
        kwargs["masses"] = True
    if "units" in signature.parameters:
        kwargs["units"] = args.units
    if "reduce_cell" in signature.parameters:
        kwargs["reduce_cell"] = args.reduce_cell
    if "force_skew" in signature.parameters:
        kwargs["force_skew"] = args.force_skew
    if "atom_type_labels" in signature.parameters:
        kwargs["atom_type_labels"] = True
    if "bonds" in signature.parameters and args.atom_style != "full":
        kwargs["bonds"] = False

    return kwargs


def format_masses_section(
    species_order: list[str],
    atomic_numbers: dict[str, int],
    atomic_masses,
) -> str:
    lines = ["Masses", ""]
    for type_id, symbol in enumerate(species_order, start=1):
        mass = atomic_masses[atomic_numbers[symbol]]
        lines.append(f"{type_id:6d} {mass:20.10f} # {symbol}")
    lines.append("")
    return "\n".join(lines)


def ensure_masses_section(
    data_text: str,
    species_order: list[str],
    atomic_numbers: dict[str, int],
    atomic_masses,
) -> str:
    if "\nMasses\n" in data_text:
        return data_text

    masses_block = format_masses_section(species_order, atomic_numbers, atomic_masses)
    atoms_index = data_text.find("\nAtoms")
    if atoms_index == -1:
        return data_text.rstrip() + "\n\n" + masses_block + "\n"
    return data_text[:atoms_index] + "\n" + masses_block + data_text[atoms_index:]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(input_path)

    try:
        from ase.data import atomic_masses, atomic_numbers
        from ase.io import read
        from ase.io.lammpsdata import write_lammps_data
    except ImportError as exc:
        print("This script requires ASE. Install it with: pip install ase", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 1

    input_format = infer_input_format(input_path, args.input_format)

    try:
        atoms = read(str(input_path), format=input_format, index=args.index)
    except Exception as exc:
        print(f"Failed to read input structure: {input_path}", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1

    if isinstance(atoms, list):
        if len(atoms) != 1:
            print(
                "LAMMPS data files only support one structure. "
                "Use --index to select a single frame.",
                file=sys.stderr,
            )
            return 1
        atoms = atoms[0]

    symbols = atoms.get_chemical_symbols()
    try:
        explicit_species = normalize_species_order(args.species_order, atomic_numbers)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if explicit_species is None:
        species_order = default_species_order(symbols, atomic_numbers)
    else:
        missing = sorted(set(symbols) - set(explicit_species), key=lambda symbol: atomic_numbers[symbol])
        if missing:
            print(
                "The structure contains species missing from --species-order: "
                + ", ".join(missing),
                file=sys.stderr,
            )
            return 1
        species_order = explicit_species

    write_kwargs = build_write_kwargs(write_lammps_data, args, species_order)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text_buffer = io.StringIO()
        write_lammps_data(text_buffer, atoms, **write_kwargs)
        data_text = ensure_masses_section(
            text_buffer.getvalue(),
            species_order,
            atomic_numbers,
            atomic_masses,
        )
        with output_path.open("w", encoding="ascii") as handle:
            handle.write(data_text)
    except Exception as exc:
        print(f"Failed to write LAMMPS data file: {output_path}", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1

    present_species = set(symbols)
    print(f"Wrote LAMMPS data file: {output_path}")
    print(f"Source structure: {input_path}")
    print("LAMMPS type mapping:")
    for type_id, symbol in enumerate(species_order, start=1):
        status = "present" if symbol in present_species else "reserved"
        print(f"  {type_id:>3} -> {symbol:<2} ({status})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
