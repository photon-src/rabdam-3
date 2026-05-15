"""
Prepare structure data for RABDAM 3 calculations.

This module takes raw StructureData from input/reader.py and prepares the
asymmetric-unit atom records for later RABDAM stages.

It does the following:
    - remove hydrogens
    - remove atoms with invalid occupancy
    - remove atoms with non-positive B-factors
    - resolve alternate conformers
    - classify atoms as protein, nucleic acid, solvent, HETATM, etc.
    - select atoms for BDamage calculation
"""

from dataclasses import dataclass

from input.reader import AtomRecord, StructureData, StructureMetadata


AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


NUCLEOTIDES = frozenset(
    {
        "A",
        "C",
        "G",
        "U",
        "T",
        "DA",
        "DC",
        "DG",
        "DT",
        "DU",
    }
)


SOLVENTS = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
    }
)


class StructurePreparationError(ValueError):
    """Raised when RABDAM 3 cannot prepare structure data for calculation."""


@dataclass(frozen=True)
class StructurePreparationOptions:
    """
    Options controlling structure preparation.
    """

    remove_hydrogens: bool = True
    require_valid_occupancy: bool = True
    require_positive_b_factor: bool = True
    resolve_altlocs: bool = True

    include_hetatm_in_selection: bool = False
    include_nucleic_acid_in_selection: bool = False
    require_protein_selection: bool = True

    remove_atom_serials: frozenset[int] = frozenset()
    add_atom_serials: frozenset[int] = frozenset()

    remove_component_names: frozenset[str] = frozenset()
    add_component_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PreparedAtom:
    """
    An atom record with structure-preparation annotations.
    """

    record: AtomRecord
    is_hydrogen: bool
    is_protein: bool
    is_nucleic_acid: bool
    is_solvent: bool
    is_hetatm: bool


@dataclass(frozen=True)
class StructurePreparationReport:
    """
    Counts and warnings produced during structure preparation.
    """

    input_atom_count: int
    cleaned_atom_count: int
    selected_atom_count: int
    removed_hydrogen_count: int
    removed_invalid_occupancy_count: int
    removed_invalid_b_factor_count: int
    removed_altloc_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedStructure:
    """
    Prepared structure data for later RABDAM 3 stages.

    cleaned_atoms:
        All valid cleaned atoms retained from the asymmetric unit.

    selected_atoms:
        The subset of cleaned atoms selected for BDamage calculation.

    metadata:
        Structure-level metadata from the reader.

    report:
        Counts and warnings from the preparation stage.
    """

    cleaned_atoms: tuple[PreparedAtom, ...]
    selected_atoms: tuple[PreparedAtom, ...]
    metadata: StructureMetadata
    report: StructurePreparationReport


def prepare_structure(
    structure_data: StructureData,
    options: StructurePreparationOptions | None = None,
) -> PreparedStructure:
    """
    Prepare raw structure data for RABDAM calculations.

    This is the main entry point for the structure-preparation stage.
    """

    if options is None:
        options = StructurePreparationOptions()

    raw_atoms = tuple(structure_data.atoms)

    cleaned_atoms, filter_counts, filter_warnings = _filter_clean_atoms(
        raw_atoms,
        options,
    )

    altloc_atoms, removed_altloc_count, altloc_warnings = _select_altlocs(
        cleaned_atoms,
        options,
    )

    prepared_atoms = tuple(_classify_atom(atom) for atom in altloc_atoms)

    selected_atoms = _select_bdamage_atoms(
        prepared_atoms,
        options,
    )

    _check_prepared_structure(
        cleaned_atoms=prepared_atoms,
        selected_atoms=selected_atoms,
        options=options,
    )

    warnings = tuple(filter_warnings + altloc_warnings)

    report = StructurePreparationReport(
        input_atom_count=len(raw_atoms),
        cleaned_atom_count=len(prepared_atoms),
        selected_atom_count=len(selected_atoms),
        removed_hydrogen_count=filter_counts["hydrogen"],
        removed_invalid_occupancy_count=filter_counts["invalid_occupancy"],
        removed_invalid_b_factor_count=filter_counts["invalid_b_factor"],
        removed_altloc_count=removed_altloc_count,
        warnings=warnings,
    )

    return PreparedStructure(
        cleaned_atoms=prepared_atoms,
        selected_atoms=selected_atoms,
        metadata=structure_data.metadata,
        report=report,
    )


def _filter_clean_atoms(
    atoms: tuple[AtomRecord, ...],
    options: StructurePreparationOptions,
) -> tuple[tuple[AtomRecord, ...], dict[str, int], list[str]]:
    """
    Apply basic atom-cleaning rules.

    This removes atoms that should not enter the cleaned asymmetric-unit atom
    list used by later RABDAM stages.
    """

    kept_atoms: list[AtomRecord] = []

    counts = {
        "hydrogen": 0,
        "invalid_occupancy": 0,
        "invalid_b_factor": 0,
    }

    warnings: list[str] = []

    for atom in atoms:
        if options.remove_hydrogens and _is_hydrogen(atom):
            counts["hydrogen"] += 1
            continue

        if options.require_valid_occupancy and not _has_valid_occupancy(atom):
            counts["invalid_occupancy"] += 1
            continue

        if options.require_positive_b_factor and not _has_positive_b_factor(atom):
            counts["invalid_b_factor"] += 1
            continue

        kept_atoms.append(atom)

    return tuple(kept_atoms), counts, warnings


def _select_altlocs(
    atoms: tuple[AtomRecord, ...],
    options: StructurePreparationOptions,
) -> tuple[tuple[AtomRecord, ...], int, list[str]]:
    """
    Resolve alternate conformers.

    For atoms that appear to represent the same atom site, keep the highest
    occupancy version. If occupancies tie, keep the one encountered first in
    the input order.
    """

    if not options.resolve_altlocs:
        return atoms, 0, []

    grouped_atoms: dict[tuple[object, ...], list[tuple[int, AtomRecord]]] = {}

    for index, atom in enumerate(atoms):
        key = _atom_site_key(atom)
        grouped_atoms.setdefault(key, []).append((index, atom))

    kept_indexes: set[int] = set()
    removed_count = 0
    residues_with_altlocs: dict[tuple[object, ...], AtomRecord] = {}

    for key, group in grouped_atoms.items():
        if len(group) == 1:
            kept_indexes.add(group[0][0])
            continue

        selected_index, selected_atom = max(
            group,
            key=lambda item: (item[1].occupancy, -item[0]),
        )

        kept_indexes.add(selected_index)
        removed_count += len(group) - 1
        residues_with_altlocs.setdefault(_residue_key(selected_atom), selected_atom)

    kept_atoms = tuple(
        atom
        for index, atom in enumerate(atoms)
        if index in kept_indexes
    )

    warnings = [
        "Alternate conformers detected in "
        f"{_format_residue_label(atom)}. "
        "Highest-occupancy atom sites were retained."
        for atom in residues_with_altlocs.values()
    ]

    return kept_atoms, removed_count, warnings


def _classify_atom(atom: AtomRecord) -> PreparedAtom:
    """
    Classify one atom record for RABDAM preparation.
    """

    component_name = atom.residue_name.strip().upper()
    record_type = atom.record_type.strip().upper()

    return PreparedAtom(
        record=atom,
        is_hydrogen=_is_hydrogen(atom),
        is_protein=component_name in AMINO_ACIDS,
        is_nucleic_acid=component_name in NUCLEOTIDES,
        is_solvent=component_name in SOLVENTS,
        is_hetatm=record_type == "HETATM",
    )


def _select_bdamage_atoms(
    atoms: tuple[PreparedAtom, ...],
    options: StructurePreparationOptions,
) -> tuple[PreparedAtom, ...]:
    """
    Select atoms that should receive BDamage values.

    Default behavior selects standard protein atoms from the cleaned atom list.
    """

    selected_atoms: list[PreparedAtom] = []
    selected_keys: set[int] = set()

    remove_component_names = _normalize_component_name_set(options.remove_component_names)
    add_component_names = _normalize_component_name_set(options.add_component_names)

    for atom in atoms:
        if _should_remove_from_selection(atom, options, remove_component_names):
            continue

        if _is_default_bdamage_selection(atom, options):
            _append_selected_atom(atom, selected_atoms, selected_keys)

    for atom in atoms:
        if _should_force_add_to_selection(atom, options, add_component_names):
            _append_selected_atom(atom, selected_atoms, selected_keys)

    return tuple(selected_atoms)


def _check_prepared_structure(
    *,
    cleaned_atoms: tuple[PreparedAtom, ...],
    selected_atoms: tuple[PreparedAtom, ...],
    options: StructurePreparationOptions,
) -> None:
    """
    Validate the prepared structure before later RABDAM stages run.
    """

    if not cleaned_atoms:
        raise StructurePreparationError(
            "No atoms remain after structure preparation."
        )

    if not selected_atoms:
        raise StructurePreparationError(
            "No atoms were selected for BDamage calculation."
        )

    if options.require_protein_selection:
        if not any(atom.is_protein for atom in selected_atoms):
            raise StructurePreparationError(
                "No protein atoms were selected for BDamage calculation."
            )


def _is_default_bdamage_selection(
    atom: PreparedAtom,
    options: StructurePreparationOptions,
) -> bool:
    """
    Return True if an atom should be selected by default for BDamage.
    """

    if atom.is_protein and not atom.is_hetatm:
        return True

    if options.include_hetatm_in_selection and atom.is_hetatm and not atom.is_solvent:
        return True

    if options.include_nucleic_acid_in_selection and atom.is_nucleic_acid:
        return True

    return False


def _should_remove_from_selection(
    atom: PreparedAtom,
    options: StructurePreparationOptions,
    remove_component_names: frozenset[str],
) -> bool:
    """
    Return True if an atom should be excluded from BDamage selection.
    """

    atom_serial = atom.record.atom_serial
    component_name = atom.record.residue_name.strip().upper()

    if atom_serial is not None and atom_serial in options.remove_atom_serials:
        return True

    if component_name in remove_component_names:
        return True

    return False


def _should_force_add_to_selection(
    atom: PreparedAtom,
    options: StructurePreparationOptions,
    add_component_names: frozenset[str],
) -> bool:
    """
    Return True if an atom should be force-added to BDamage selection.
    """

    atom_serial = atom.record.atom_serial
    component_name = atom.record.residue_name.strip().upper()

    if atom_serial is not None and atom_serial in options.add_atom_serials:
        return True

    if component_name in add_component_names:
        return True

    return False


def _append_selected_atom(
    atom: PreparedAtom,
    selected_atoms: list[PreparedAtom],
    selected_keys: set[int],
) -> None:
    """
    Append an atom to the selected list if it is not already present.
    """

    key = _prepared_atom_key(atom)

    if key in selected_keys:
        return

    selected_atoms.append(atom)
    selected_keys.add(key)


def _is_hydrogen(atom: AtomRecord) -> bool:
    """
    Return True if an atom is hydrogen or deuterium.
    """

    element = atom.element.strip().upper()

    if element:
        return element in {"H", "D"}

    atom_name = atom.atom_name.strip().upper()

    return atom_name.startswith(("H", "D"))


def _has_valid_occupancy(atom: AtomRecord) -> bool:
    """
    Return True if occupancy is in the valid range for this preparation stage.
    """

    return 0.0 < atom.occupancy <= 1.0


def _has_positive_b_factor(atom: AtomRecord) -> bool:
    """
    Return True if an atom has a positive B-factor.
    """

    return atom.b_factor > 0.0


def _normalize_component_name_set(component_names: frozenset[str]) -> frozenset[str]:
    """
    Normalize component names to uppercase.
    """

    return frozenset(component_name.strip().upper() for component_name in component_names)


def _atom_site_key(atom: AtomRecord) -> tuple[object, ...]:
    """
    Return a key representing a unique atom site before altloc selection.

    Alternate conformers of the same atom site should share this key.
    """

    return (
        atom.model_number,
        atom.chain_id,
        atom.residue_name.strip().upper(),
        atom.residue_number,
        atom.insertion_code.strip(),
        atom.atom_name.strip().upper(),
    )


def _residue_key(atom: AtomRecord) -> tuple[object, ...]:
    """
    Return a key representing a residue.
    """

    return (
        atom.model_number,
        atom.chain_id,
        atom.residue_name.strip().upper(),
        atom.residue_number,
        atom.insertion_code.strip(),
    )


def _format_residue_label(atom: AtomRecord) -> str:
    """
    Return a compact human-readable residue label for warnings.
    """

    chain_id = atom.chain_id.strip() or "?"
    residue_name = atom.residue_name.strip().upper() or "?"
    residue_number = "?" if atom.residue_number is None else str(atom.residue_number)
    insertion_code = atom.insertion_code.strip()

    return f"chain {chain_id} residue {residue_name}{residue_number}{insertion_code}"


def _prepared_atom_key(atom: PreparedAtom) -> int:
    """
    Return a stable key representing a prepared atom.
    """

    return atom.record.source_atom_index
