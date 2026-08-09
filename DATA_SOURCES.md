# Data sources and redistribution

The controlled spherical notebook generates its observations from fixed seeds.
The three geographic notebooks use the public files described below. The
software license in `LICENSE` does not replace the terms attached to those
data.

## Chicago community-area compositions and boundaries

`data/chicago_community_area_composition.csv` contains five mutually exclusive
race and ethnicity counts for Chicago's 77 official community areas. The table
was consolidated from the Chicago Metropolitan Agency for Planning (CMAP)
Community Data Snapshots released in June 2026, which primarily summarize
2020--2024 American Community Survey five-year estimates. Source fields
`WHITE`, `HISP`, `BLACK`, `ASIAN`, and `OTHER` were rounded independently to
the nearest integer; `category_total` is their sum.

The official CMAP ArcGIS item marks the data as "Not restricted" and asks users
to cite the Chicago Metropolitan Agency for Planning as the source. The
derived table has SHA-256 digest
`d01a19f09fa3c068a9f061270768023b5de6cd0eea71ec1bb17e51f7df6a0434`.

The `Chicago77` shapefile components are byte-identical to the copies
distributed with PySAL/libpysal at the immutable source recorded in
`data/chicago_source.json`. Retain `data/Chicago77_LICENSE.txt` and
`data/Chicago77_README.md` when redistributing them. The shapes support the
community-area queen-contiguity analysis; they are not tract boundaries.

Primary links:

- CMAP item: https://www.arcgis.com/home/item.html?id=97e9fe4ef14c42da9d26e6fc31bee5e0
- CMAP community-area layer: https://services5.arcgis.com/LcMXE3TFhi1BSaCY/arcgis/rest/services/Community_Data_Snapshots_2026/FeatureServer/11
- Chicago77 immutable directory: https://github.com/pysal/libpysal/tree/b808e7c0df8ddf6b62fb4f92d4ec0e39f9ec32e5/libpysal/examples/chicago

## California commuting-displacement tensors

`data/california_commuting_input.npz` is a compact, pickle-free derivative of
official 2016--2020 ACS residence-county to workplace-county commuting flows
and the 2020 California county Gazetteer. The preparation retained California
residence counties and California workplace destinations, normalized each
origin's retained worker counts, and formed a weighted second moment of planar
displacements in EPSG:3310. A common ridge equal to `1e-4` times the median raw
tensor trace makes every two-by-two tensor strictly positive definite.

The archive contains all 58 county tensors plus the county names, GEOIDs,
representative-point coordinates, retained worker totals, destination counts,
and the common ridge. Its SHA-256 digest is
`eafa2920b2016d4756332f61c830c9e11df1958dab59a07040854092812a016d`.
The source URLs, raw-file checksums, array schema, and processing steps are in
`data/california_source.json`.

The underlying flow and Gazetteer files are U.S. Census Bureau public-use
federal data. Any redistributed derivative should continue to identify the
U.S. Census Bureau, the ACS commuting-flow table, and the 2020 Gazetteer as its
sources. See the Census Bureau citation policy linked in the source registry.

Primary links:

- ACS commuting flows: https://www2.census.gov/programs-surveys/demo/tables/metro-micro/2020/commuting-flows-2020/table1.xlsx
- California county Gazetteer: https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_gaz_counties_06.txt

## MeteoSwiss phenological observations

`data/meteoswiss.csv` contains historical Swiss peak-bloom observations. Its
SHA-256 digest is
`3c77697278844f7e286a57c6e2d52f5d39806cfb475f993d0b29ba4033dfd310`
and matches the public file used by the GMU Cherry Blossom competition. The
official opendata.swiss record permits commercial and noncommercial reuse but
requires the source to be named.

Any redistributed copy must retain this attribution: **Source: MeteoSwiss.**

The notebook retains stations with at least eight observations before 1990
and at least eight from 1990 onward, then computes an intrinsic circular
Fréchet mean for each era. The exact acquisition link, official metadata,
terms, schema, and checksum are in `data/meteoswiss_source.json`.

Primary links:

- Official dataset: https://opendata.swiss/en/dataset/phanologische-beobachtungen1
- Official documentation: https://opendatadocs.meteoswiss.ch/a-data-groundbased/a9-phenological-observations
- Terms: https://opendata.swiss/en/terms-of-use#terms_by
