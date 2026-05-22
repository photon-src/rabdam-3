"""
Packing-density calculation for BDamage.

The trimmed crystal block contains the local neighbour cloud around the selected
asymmetric-unit atoms. For each selected atom, packing density is the number of
trimmed crystal atoms whose Cartesian distance from that selected atom is less
than the packing-density threshold, minus one to remove the selected atom's
central-cell copy.

This module performs the exact distance-counting step after the broader crystal
block has already been reduced by crystal.trim.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np

from crystal.translate import TranslatedAtom
from crystal.trim import ArrayTrimmedCrystalBlock, TrimmedNeighbourBlock
from structure.models import PreparedAtom, PreparedStructure


class PackingDensityError(ValueError):
    """Raised when RABDAM cannot calculate packing density."""


SpatialCellKey = tuple[int, int, int]
SpatialIndex = dict[SpatialCellKey, list[TranslatedAtom]]
ArraySpatialIndex = dict[SpatialCellKey, list[int]]


@dataclass(frozen=True)
class PackingDensityAtomResult:
    """
    Packing-density result for one selected asymmetric-unit atom.

    packing_density_atom_index:
        One-based position of this atom in the packing-density result list.

    source_atom_index:
        Zero-based reader index of the selected asymmetric-unit atom.

    atom_serial:
        Atom serial number from the input structure, when available.

    neighbour_count:
        Number of trimmed crystal atoms within the packing-density threshold of
        this selected atom, after subtracting one for the central-cell copy of
        the atom itself.
    """

    packing_density_atom_index: int
    source_atom_index: int
    atom_serial: int | None
    neighbour_count: int


@dataclass(frozen=True)
class PackingDensityResult:
    """
    Packing-density counts for the selected BDamage atoms.

    atom_results:
        One result per selected asymmetric-unit atom, in selected-atom order.

    packing_density_threshold:
        Distance cutoff in Angstroms used for neighbour counting.

    selected_atom_count:
        Number of selected asymmetric-unit atoms that were scored.

    neighbour_atom_count:
        Number of trimmed crystal atoms searched for each selected atom.
    """

    atom_results: tuple[PackingDensityAtomResult, ...]
    packing_density_threshold: float
    selected_atom_count: int
    neighbour_atom_count: int


def calculate_bdamage_packing_density(
    *,
    prepared_structure: PreparedStructure,
    trimmed_block: TrimmedNeighbourBlock,
    packing_density_threshold: float,
) -> PackingDensityResult:
    """
    Calculate packing density for the BDamage-selected atoms.

    This convenience wrapper uses prepared_structure.selected_atoms as the atoms
    that receive packing-density counts and the trimmed block as the local
    crystal neighbour cloud.
    """

    if isinstance(trimmed_block, ArrayTrimmedCrystalBlock):
        return calculate_packing_density_from_arrays(
            selected_atoms=prepared_structure.selected_atoms,
            neighbour_coordinates=trimmed_block.coordinates,
            source_atom_indices=trimmed_block.source_atom_indices,
            is_identity_symmetry_operation=(
                trimmed_block.is_identity_symmetry_operation
            ),
            translation_offsets=trimmed_block.translation_offsets,
            packing_density_threshold=packing_density_threshold,
        )

    return calculate_packing_density(
        selected_atoms=prepared_structure.selected_atoms,
        neighbour_atoms=trimmed_block.atoms,
        packing_density_threshold=packing_density_threshold,
    )


def calculate_packing_density(
    *,
    selected_atoms: Iterable[PreparedAtom],
    neighbour_atoms: Iterable[TranslatedAtom],
    packing_density_threshold: float,
) -> PackingDensityResult:
    """
    Count neighbour atoms within packing_density_threshold of each selected atom.
    """

    if (
        not math.isfinite(packing_density_threshold)
        or packing_density_threshold <= 0
    ):
        raise PackingDensityError(
            "packing_density_threshold must be a finite positive number, "
            f"got {packing_density_threshold!r}."
        )

    selected_atom_tuple = tuple(selected_atoms)
    if not selected_atom_tuple:
        raise PackingDensityError(
            "Cannot calculate packing density for an empty selected-atom list."
        )

    neighbour_atom_tuple = tuple(neighbour_atoms)
    if not neighbour_atom_tuple:
        raise PackingDensityError(
            "Cannot calculate packing density with an empty neighbour-atom list."
        )

    threshold_squared = float(packing_density_threshold) ** 2
    spatial_index = _build_spatial_index(
        neighbour_atom_tuple,
        cell_size=float(packing_density_threshold),
    )

    atom_results = tuple(
        PackingDensityAtomResult(
            packing_density_atom_index=selected_atom_index,
            source_atom_index=selected_atom.record.source_atom_index,
            atom_serial=selected_atom.record.atom_serial,
            neighbour_count=_count_neighbours_excluding_selected_atom_self_copy(
                selected_atom=selected_atom,
                neighbour_atoms=_nearby_neighbour_atoms(
                    selected_atom=selected_atom,
                    spatial_index=spatial_index,
                    cell_size=float(packing_density_threshold),
                ),
                threshold_squared=threshold_squared,
            ),
        )
        for selected_atom_index, selected_atom in enumerate(
            selected_atom_tuple,
            start=1,
        )
    )

    return PackingDensityResult(
        atom_results=atom_results,
        packing_density_threshold=float(packing_density_threshold),
        selected_atom_count=len(selected_atom_tuple),
        neighbour_atom_count=len(neighbour_atom_tuple),
    )


def calculate_packing_density_from_arrays(
    *,
    selected_atoms: Iterable[PreparedAtom],
    neighbour_coordinates: np.ndarray,
    source_atom_indices: np.ndarray,
    is_identity_symmetry_operation: np.ndarray,
    translation_offsets: np.ndarray,
    packing_density_threshold: float,
) -> PackingDensityResult:
    """
    Count neighbours using array-backed retained translated atoms.
    """

    if (
        not math.isfinite(packing_density_threshold)
        or packing_density_threshold <= 0
    ):
        raise PackingDensityError(
            "packing_density_threshold must be a finite positive number, "
            f"got {packing_density_threshold!r}."
        )

    selected_atom_tuple = tuple(selected_atoms)
    if not selected_atom_tuple:
        raise PackingDensityError(
            "Cannot calculate packing density for an empty selected-atom list."
        )

    coordinates = np.asarray(neighbour_coordinates, dtype=np.float64)
    source_indices = np.asarray(source_atom_indices, dtype=np.int64)
    identity_flags = np.asarray(is_identity_symmetry_operation, dtype=np.bool_)
    offsets = np.asarray(translation_offsets, dtype=np.int64)

    _validate_neighbour_arrays(
        coordinates=coordinates,
        source_atom_indices=source_indices,
        is_identity_symmetry_operation=identity_flags,
        translation_offsets=offsets,
    )

    threshold = float(packing_density_threshold)
    threshold_squared = threshold**2
    spatial_index = _build_array_spatial_index(
        coordinates,
        cell_size=threshold,
    )

    atom_results = tuple(
        PackingDensityAtomResult(
            packing_density_atom_index=selected_atom_index,
            source_atom_index=selected_atom.record.source_atom_index,
            atom_serial=selected_atom.record.atom_serial,
            neighbour_count=_count_array_neighbours_excluding_selected_atom_self_copy(
                selected_atom=selected_atom,
                neighbour_coordinates=coordinates,
                source_atom_indices=source_indices,
                is_identity_symmetry_operation=identity_flags,
                translation_offsets=offsets,
                neighbour_indices=_nearby_neighbour_indices(
                    selected_atom=selected_atom,
                    spatial_index=spatial_index,
                    cell_size=threshold,
                ),
                threshold_squared=threshold_squared,
            ),
        )
        for selected_atom_index, selected_atom in enumerate(
            selected_atom_tuple,
            start=1,
        )
    )

    return PackingDensityResult(
        atom_results=atom_results,
        packing_density_threshold=threshold,
        selected_atom_count=len(selected_atom_tuple),
        neighbour_atom_count=int(coordinates.shape[0]),
    )


def _validate_neighbour_arrays(
    *,
    coordinates: np.ndarray,
    source_atom_indices: np.ndarray,
    is_identity_symmetry_operation: np.ndarray,
    translation_offsets: np.ndarray,
) -> None:
    """Validate array-backed neighbour-cloud shapes."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise PackingDensityError(
            "neighbour_coordinates must have shape (n, 3)."
        )

    neighbour_count = coordinates.shape[0]
    if neighbour_count == 0:
        raise PackingDensityError(
            "Cannot calculate packing density with an empty neighbour-atom list."
        )

    if source_atom_indices.shape != (neighbour_count,):
        raise PackingDensityError(
            "source_atom_indices must have shape (n,) matching coordinates."
        )

    if is_identity_symmetry_operation.shape != (neighbour_count,):
        raise PackingDensityError(
            "is_identity_symmetry_operation must have shape (n,) matching "
            "coordinates."
        )

    if translation_offsets.shape != (neighbour_count, 3):
        raise PackingDensityError(
            "translation_offsets must have shape (n, 3) matching coordinates."
        )


def _build_spatial_index(
    neighbour_atoms: Iterable[TranslatedAtom],
    *,
    cell_size: float,
) -> SpatialIndex:
    """Bucket neighbour atoms into cubic cells with edge length cell_size."""

    spatial_index: SpatialIndex = {}
    for neighbour_atom in neighbour_atoms:
        cell_key = _spatial_cell_key(
            x=neighbour_atom.x,
            y=neighbour_atom.y,
            z=neighbour_atom.z,
            cell_size=cell_size,
        )
        spatial_index.setdefault(cell_key, []).append(neighbour_atom)

    return spatial_index


def _build_array_spatial_index(
    neighbour_coordinates: np.ndarray,
    *,
    cell_size: float,
) -> ArraySpatialIndex:
    """Bucket array-backed neighbour atoms into cubic cells."""

    spatial_index: ArraySpatialIndex = {}
    cell_coordinates = np.floor(neighbour_coordinates / cell_size).astype(np.int64)
    for neighbour_index, cell_coordinate in enumerate(cell_coordinates):
        cell_key = (
            int(cell_coordinate[0]),
            int(cell_coordinate[1]),
            int(cell_coordinate[2]),
        )
        spatial_index.setdefault(cell_key, []).append(neighbour_index)

    return spatial_index


def _nearby_neighbour_atoms(
    *,
    selected_atom: PreparedAtom,
    spatial_index: SpatialIndex,
    cell_size: float,
) -> tuple[TranslatedAtom, ...]:
    """
    Return atoms from spatial cells that can contain threshold-distance neighbours.
    """

    selected_cell_key = _spatial_cell_key(
        x=selected_atom.record.x,
        y=selected_atom.record.y,
        z=selected_atom.record.z,
        cell_size=cell_size,
    )
    selected_cell_x, selected_cell_y, selected_cell_z = selected_cell_key

    nearby_atoms: list[TranslatedAtom] = []
    for cell_x in range(selected_cell_x - 1, selected_cell_x + 2):
        for cell_y in range(selected_cell_y - 1, selected_cell_y + 2):
            for cell_z in range(selected_cell_z - 1, selected_cell_z + 2):
                nearby_atoms.extend(spatial_index.get((cell_x, cell_y, cell_z), ()))

    return tuple(nearby_atoms)


def _nearby_neighbour_indices(
    *,
    selected_atom: PreparedAtom,
    spatial_index: ArraySpatialIndex,
    cell_size: float,
) -> np.ndarray:
    """
    Return array indices from cells that can contain threshold-distance neighbours.
    """

    selected_cell_key = _spatial_cell_key(
        x=selected_atom.record.x,
        y=selected_atom.record.y,
        z=selected_atom.record.z,
        cell_size=cell_size,
    )
    selected_cell_x, selected_cell_y, selected_cell_z = selected_cell_key

    nearby_indices: list[int] = []
    for cell_x in range(selected_cell_x - 1, selected_cell_x + 2):
        for cell_y in range(selected_cell_y - 1, selected_cell_y + 2):
            for cell_z in range(selected_cell_z - 1, selected_cell_z + 2):
                nearby_indices.extend(spatial_index.get((cell_x, cell_y, cell_z), ()))

    return np.asarray(nearby_indices, dtype=np.int64)


def _spatial_cell_key(
    *,
    x: float,
    y: float,
    z: float,
    cell_size: float,
) -> SpatialCellKey:
    """Return the spatial-grid cell containing one Cartesian coordinate."""

    return (
        math.floor(x / cell_size),
        math.floor(y / cell_size),
        math.floor(z / cell_size),
    )


def _count_neighbours_excluding_selected_atom_self_copy(
    *,
    selected_atom: PreparedAtom,
    neighbour_atoms: Iterable[TranslatedAtom],
    threshold_squared: float,
) -> int:
    """
    Count neighbours, then remove the selected atom's central-cell copy.
    """

    raw_count = _count_neighbours_within_threshold_squared(
        selected_atom=selected_atom,
        neighbour_atoms=neighbour_atoms,
        threshold_squared=threshold_squared,
    )
    if not _selected_atom_self_copy_is_counted(
        selected_atom=selected_atom,
        neighbour_atoms=neighbour_atoms,
        threshold_squared=threshold_squared,
    ):
        raise PackingDensityError(
            "Cannot subtract the selected atom's central-cell copy from the "
            "packing-density count because that copy was not counted. Check "
            "that the neighbour cloud contains the selected atom's central-cell "
            "image."
        )

    return raw_count - 1


def _count_array_neighbours_excluding_selected_atom_self_copy(
    *,
    selected_atom: PreparedAtom,
    neighbour_coordinates: np.ndarray,
    source_atom_indices: np.ndarray,
    is_identity_symmetry_operation: np.ndarray,
    translation_offsets: np.ndarray,
    neighbour_indices: np.ndarray,
    threshold_squared: float,
) -> int:
    """
    Count array-backed neighbours, then remove the selected atom's self copy.
    """

    if not math.isfinite(threshold_squared) or threshold_squared < 0:
        raise PackingDensityError(
            "threshold_squared must be a finite non-negative number, "
            f"got {threshold_squared!r}."
        )

    if neighbour_indices.size == 0:
        raise PackingDensityError(
            "Cannot subtract the selected atom's central-cell copy from the "
            "packing-density count because that copy was not counted. Check "
            "that the neighbour cloud contains the selected atom's central-cell "
            "image."
        )

    selected_coordinates = np.asarray(
        (
            selected_atom.record.x,
            selected_atom.record.y,
            selected_atom.record.z,
        ),
        dtype=np.float64,
    )
    nearby_coordinates = neighbour_coordinates[neighbour_indices]
    coordinate_deltas = selected_coordinates - nearby_coordinates
    distances_squared = np.einsum(
        "ij,ij->i",
        coordinate_deltas,
        coordinate_deltas,
    )
    is_within_threshold = distances_squared < threshold_squared
    raw_count = int(np.count_nonzero(is_within_threshold))

    zero_translation_offsets = np.all(
        translation_offsets[neighbour_indices] == 0,
        axis=1,
    )
    selected_atom_self_copies = (
        (
            source_atom_indices[neighbour_indices]
            == selected_atom.record.source_atom_index
        )
        & is_identity_symmetry_operation[neighbour_indices]
        & zero_translation_offsets
        & is_within_threshold
    )
    if not bool(np.any(selected_atom_self_copies)):
        raise PackingDensityError(
            "Cannot subtract the selected atom's central-cell copy from the "
            "packing-density count because that copy was not counted. Check "
            "that the neighbour cloud contains the selected atom's central-cell "
            "image."
        )

    return raw_count - 1


def _selected_atom_self_copy_is_counted(
    *,
    selected_atom: PreparedAtom,
    neighbour_atoms: Iterable[TranslatedAtom],
    threshold_squared: float,
) -> bool:
    """
    Return True when the selected atom's central-cell copy is counted.
    """

    selected_x = selected_atom.record.x
    selected_y = selected_atom.record.y
    selected_z = selected_atom.record.z

    return any(
        neighbour_atom.source_atom_index == selected_atom.record.source_atom_index
        and neighbour_atom.is_identity_symmetry_operation
        and neighbour_atom.translation_a == 0
        and neighbour_atom.translation_b == 0
        and neighbour_atom.translation_c == 0
        and squared_distance_to_translated_atom(
            selected_x=selected_x,
            selected_y=selected_y,
            selected_z=selected_z,
            neighbour_atom=neighbour_atom,
        )
        < threshold_squared
        for neighbour_atom in neighbour_atoms
    )


def _count_neighbours_within_threshold_squared(
    *,
    selected_atom: PreparedAtom,
    neighbour_atoms: Iterable[TranslatedAtom],
    threshold_squared: float,
) -> int:
    """
    Count neighbours whose squared Cartesian distance is < threshold_squared.

    Squared distances are used to avoid square-root calculations while producing
    the same inclusion result as comparing true Euclidean distances.
    """

    if not math.isfinite(threshold_squared) or threshold_squared < 0:
        raise PackingDensityError(
            "threshold_squared must be a finite non-negative number, "
            f"got {threshold_squared!r}."
        )

    selected_x = selected_atom.record.x
    selected_y = selected_atom.record.y
    selected_z = selected_atom.record.z

    count = 0
    for neighbour_atom in neighbour_atoms:
        if squared_distance_to_translated_atom(
            selected_x=selected_x,
            selected_y=selected_y,
            selected_z=selected_z,
            neighbour_atom=neighbour_atom,
        ) < threshold_squared:
            count += 1

    return count


def squared_distance_to_translated_atom(
    *,
    selected_x: float,
    selected_y: float,
    selected_z: float,
    neighbour_atom: TranslatedAtom,
) -> float:
    """
    Return squared Cartesian distance from a selected atom to a neighbour atom.
    """

    dx = selected_x - neighbour_atom.x
    dy = selected_y - neighbour_atom.y
    dz = selected_z - neighbour_atom.z

    return float(dx * dx + dy * dy + dz * dz)


def packing_density_counts_as_tuple(
    result: PackingDensityResult,
) -> tuple[int, ...]:
    """Return only neighbour counts from a packing-density result."""

    return tuple(atom_result.neighbour_count for atom_result in result.atom_results)
