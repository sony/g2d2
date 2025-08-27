import os
import csv
import ast
import argparse
from typing import Optional


def prepare_imagenet_csv(
    image_root: str,
    labels_txt: str,
    output_csv: str = "dataset/imagenet_val_captions.csv",
    image_ext: str = ".png",
    filename_prefix: str = "img",
    index_start: int = 0,
    index_width: int = 8,
) -> None:
    """
    Generate a CSV (image_path, caption) for ImageNet-like images using class labels.

    Assumptions:
    - image filenames contain a zero-padded index that maps to ImageNet class id
      e.g., img00000042.png -> class id 42.
    - labels_txt is a python dict literal mapping {class_id: label}.

    Parameters:
    - image_root: directory containing images.
    - labels_txt: path to imagenet1000_clsidx_to_labels.txt
    - output_csv: where to write the resulting CSV.
    - image_ext: image extension to filter files (default .png)
    - filename_prefix: prefix used before the numeric id (default 'img')
    - index_start: offset index in the stem before reading the zero-padded id (default 0)
    - index_width: width of the zero-padded numeric id (default 8)
    """

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"image_root not found: {image_root}")
    if not os.path.isfile(labels_txt):
        raise FileNotFoundError(f"labels file not found: {labels_txt}")

    with open(labels_txt, 'r') as f:
        imagenet_labels = ast.literal_eval(f.read())

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image_path", "caption"])  # header

        for filename in sorted(os.listdir(image_root)):
            if not filename.endswith(image_ext):
                continue
            stem = os.path.splitext(filename)[0]
            # extract zero-padded numeric id
            start = index_start + len(filename_prefix)
            end = start + index_width
            numeric = stem[start:end]
            try:
                class_index = int(numeric)
            except ValueError:
                raise ValueError(f"Cannot parse class index from filename: {filename}")

            label = imagenet_labels.get(class_index)
            if label is None:
                raise KeyError(f"No label for class index {class_index} in {labels_txt}")

            image_path = os.path.join(image_root, filename)
            caption = f"a photo of {label}"
            csvwriter.writerow([image_path, caption])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ImageNet captions CSV (image_path, caption)")
    p.add_argument("--image-root", required=True, help="Directory containing images (e.g., imagenet_val_1k/00000)")
    p.add_argument("--labels-txt", required=True, help="Path to imagenet1000_clsidx_to_labels.txt")
    p.add_argument("--output-csv", default="dataset/imagenet_val_captions.csv", help="Output CSV path")
    p.add_argument("--image-ext", default=".png", help="Image extension to filter (default: .png)")
    p.add_argument("--filename-prefix", default="img", help="Filename prefix before numeric id (default: img)")
    p.add_argument("--index-start", type=int, default=0, help="Offset before numeric id (default: 0)")
    p.add_argument("--index-width", type=int, default=8, help="Zero-padded width of numeric id (default: 8)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_imagenet_csv(
        image_root=args.image_root,
        labels_txt=args.labels_txt,
        output_csv=args.output_csv,
        image_ext=args.image_ext,
        filename_prefix=args.filename_prefix,
        index_start=args.index_start,
        index_width=args.index_width,
    )
