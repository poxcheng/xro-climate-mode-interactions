# data/

The input data (~836 MB, 196 files) is not stored in git. Download the dataset archive from the
Zenodo record referenced in the paper's Data Availability statement (private reviewer link during
peer review; public DOI on publication) and unpack it into this directory so that the layout matches
`../data_manifest.tsv`. Verify with:

    cd data && shasum -a 256 -c ../data_manifest.tsv
