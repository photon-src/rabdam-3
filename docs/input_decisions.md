# Input design decisions

- The input folder is called `input/`.
- Local mmCIF files are the first supported input type.
- Local PDB files are supported for compatibility.
- RCSB/PDB accession-code downloads are optional convenience features.
- Downloaded accession-code inputs are resolved to local mmCIF files before calculation.
- The calculation pipeline never depends directly on internet access.
- The parser backend is Gemmi.
- Gemmi objects are converted immediately into RABDAM-owned `StructureData` objects.
- RABDAM-specific filtering is not done in the input reader.
- TOML is used for optional configuration files, not for structure coordinates.


#################################################
Folder name:
  input/

Parser backend:
  Gemmi

Python floor:
  3.11+

Accepted structure inputs:
  local .cif
  local .mmcif
  local .pdb
  optional PDB/RCSB ID download

Input priority:
  local path first
  PDB/RCSB ID second
  clear error third

Config format:
  optional TOML

First module in pipeline:
  input/resolver.py

First file to create:
  input/sources.py