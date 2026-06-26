import argparse
import math
import os
from pathlib import Path

from PIL import Image


def resolve_template_dir(args):
    if args.template_dir:
        return Path(args.template_dir).expanduser().resolve()

    root_dir = Path(args.root_dir).expanduser().resolve()
    object_id = int(args.object_id)
    return root_dir / "templates" / args.custom_dataset_name / f"{object_id:06d}"


def make_contact_sheet(template_dir, output_path, max_images, tile_size, ncols):
    paths = sorted(
        path
        for path in template_dir.glob("*.png")
        if not path.name.endswith("_depth.png")
    )
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise FileNotFoundError(f"No template PNGs found in {template_dir}")

    tiles = []
    for path in paths:
        image = Image.open(path).convert("RGBA")
        image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 255))
        x = (tile_size - image.width) // 2
        y = (tile_size - image.height) // 2
        tile.alpha_composite(image, (x, y))
        tiles.append(tile.convert("RGB"))

    ncols = min(ncols, len(tiles))
    nrows = math.ceil(len(tiles) / ncols)
    sheet = Image.new("RGB", (ncols * tile_size, nrows * tile_size), "black")
    for idx, tile in enumerate(tiles):
        x = (idx % ncols) * tile_size
        y = (idx // ncols) * tile_size
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return len(paths), output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a contact sheet for rendered GigaPose templates."
    )
    parser.add_argument("--template_dir", help="Rendered object template directory")
    parser.add_argument("--root_dir", default="datasets", help="Dataset root containing templates/")
    parser.add_argument("--custom_dataset_name", help="Dataset name under <root_dir>/templates")
    parser.add_argument("--object_id", default=1, help="Object id, e.g. 1 for 000001")
    parser.add_argument("--output_path", help="Where to save the contact sheet")
    parser.add_argument("--max_images", type=int, default=42, help="Maximum templates to show")
    parser.add_argument("--tile_size", type=int, default=224, help="Tile size in pixels")
    parser.add_argument("--ncols", type=int, default=7, help="Number of columns")
    args = parser.parse_args()

    if not args.template_dir and not args.custom_dataset_name:
        parser.error("Pass either --template_dir or --custom_dataset_name")

    template_dir = resolve_template_dir(args)
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else template_dir / "templates.png"
    )
    count, output_path = make_contact_sheet(
        template_dir=template_dir,
        output_path=output_path,
        max_images=args.max_images,
        tile_size=args.tile_size,
        ncols=args.ncols,
    )
    print(f"Saved {count} templates to {output_path}")


if __name__ == "__main__":
    main()
