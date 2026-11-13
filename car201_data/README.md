# CAR201 local pipeline inputs

These repository-local copies remove the runtime dependency on the sibling
`CAR201/` workspace. The active joint CAR190/Amber pipeline uses:

- `ambers_photometry/photometry-all-star-epochs.parquet`;
- `zanes_photometry/20261106/` pass photometry;
- `sed/v2/` faint/original stellar posteriors and the BT-Settl spectral grid.

The earlier `sed/v1/` reference files and root-level Zane pass files are also
retained for the other in-package CAR201 utilities. File sizes and SHA256 hashes
are recorded in `INPUTS.json`. Unrelated dashboard code, `.env` credentials,
pilot products, and dashboard archives were not copied.
