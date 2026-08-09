# Release checklist

- [x] Record Kisung You as the sole author in `CITATION.cff` and package metadata. No DOI is assigned.
- [x] License the public software under the MIT License and document dataset terms separately.
- [x] Include one executable notebook for every empirical or controlled study in the manuscript.
- [x] Retain compact, legally redistributable inputs with provenance and checksums.
- [x] Execute all notebooks and retain their embedded tables and figures.
- [x] Run `python tests/test_autocorr.py` and `python autocorr.py`.
- [x] Audit the public tree for absolute user paths, credentials, caches, build products, and private files.
- [ ] Create the remote GitHub repository from this folder.
- [ ] Add the article DOI and repository URL after they exist.
- [ ] Tag the first archived release if versioned preservation is desired.

