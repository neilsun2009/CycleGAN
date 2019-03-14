# Compression and Decompression of the Dataset

## Compression
zip cat2dog.zip --out dataset.zip -s 80m

## Decompression
zip -s 0 dataset.zip --out dataset_cat2dog.zip
unzip dataset_cat2dog.zip