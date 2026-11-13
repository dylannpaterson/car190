# Portable optical and reader inputs

The local pipeline resolves these assets from the repository root:

- WFI LED SED and detector QE ECSV files at this directory's top level;
- per-detector `Roman_effarea_v8_SCA##_20240301.ecsv` files under
  `roman_effarea_v8_20240301/`, each with a provenance sidecar;
- Eric's unchanged-reader source archive under `eric_reader/`.

`PORTABLE_INPUTS.json` records sizes and SHA256 hashes. The per-SCA effective
areas are pinned to Roman technical-information commit
`14a9b0c98b0da723d0f2e4c6277956183c1fdd06`.
