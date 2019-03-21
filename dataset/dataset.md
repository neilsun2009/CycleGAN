# Cat2dog_clean
The dataset has cleaned non-face images

# Cat2dog
Origin dataset

# Men2women
Testing dataset - a baseline dataset

# Git LFS
## Use Git LFS to upload datasets
### Install
brew install git-lfs  
git lfs install  
(Go to the repo directory)  
git lfs track "*.zip"  
git add .gitattributes

# Compression and Decompression of the Dataset (only if git lfs doesn't work)
## Compression
zip cat2dog.zip --out dataset.zip -s 80m
## Decompression
zip -s 0 dataset.zip --out dataset_cat2dog.zip
unzip dataset_cat2dog.zip