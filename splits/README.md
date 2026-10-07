# FoodSense splits

The image-level train/val/test split used for every result in the paper. Each
file lists image filenames (the `file_name` column of the HuggingFace
`metadata.csv`), one per line, sorted.

| File | Images | Annotations |
|:-----|-------:|------------:|
| `train_images.txt` | 2,185 | 43,758 |
| `val_images.txt` | 292 | 5,834 |
| `test_images.txt` | 438 | 8,851 |
| `excluded_images.txt` | 72 | 1,335 |

Annotation counts include only rows where all four `CanInfer_*` flags are 1.

## How the split was made

`create_image_level_splits` in `dataset.py` with `test_size=0.15`,
`val_size=0.10`, `random_state=42`: images are stratified by their binned mean
rating. It ran on the 2,915 images whose `Image_Name` matched a file in our
image directory. The other 72 images are stored on HuggingFace as
`Copy of <Image_Name>`, so they never matched and were left out of training and
evaluation. They are listed in `excluded_images.txt`.

The 438 test images are the ones evaluated in the paper.

## Usage

`create_image_level_splits` reads these files by default, so `train.py`,
`evaluate.py` and `benchmark.py` all use this split. Images in none of the
lists are dropped. Pass `split_dir=None` to recompute a stratified split
instead. Recomputing on all 2,987 images gives a different split.

Verify the files with `sha256sum -c SHA256SUMS`.
