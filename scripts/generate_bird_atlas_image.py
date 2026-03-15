#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


SPECIES = [
    ("Accipiter", "gentilis"),
    ("Aegithalos", "caudatus"),
    ("Aegolius", "funereus"),
    ("Apus", "apus"),
    ("Ardea", "cinerea"),
    ("Asio", "otus"),
    ("Athene", "noctua"),
    ("Bubo", "bubo"),
    ("Buteo", "buteo"),
    ("Caprimulgus", "europaeus"),
    ("Carduelis", "carduelis"),
    ("Certhia", "brachydactyla"),
    ("Certhia", "familiaris"),
    ("Chloris", "chloris"),
    ("Coccothraustes", "coccothraustes"),
    ("Columba", "palumbus"),
    ("Corvus", "corax"),
    ("Cuculus", "canorus"),
    ("Curruca", "communis"),
    ("Cyanistes", "caeruleus"),
    ("Delichon", "urbicum"),
    ("Dendrocopos", "major"),
    ("Dryobates", "minor"),
    ("Dryocopus", "martius"),
    ("Emberiza", "cia"),
    ("Emberiza", "cirlus"),
    ("Erithacus", "rubecula"),
    ("Falco", "tinnunculus"),
    ("Ficedula", "hypoleuca"),
    ("Fringilla", "coelebs"),
    ("Garrulus", "glandarius"),
    ("Hirundo", "rustica"),
    ("Jynx", "torquilla"),
    ("Lanius", "collurio"),
    ("Lophophanes", "cristatus"),
    ("Milvus", "milvus"),
    ("Motacilla", "alba"),
    ("Motacilla", "cinerea"),
    ("Muscicapa", "striata"),
    ("Oriolus", "oriolus"),
    ("Otus", "scops"),
    ("Parus", "major"),
    ("Passer", "montanus"),
    ("Periparus", "ater"),
    ("Pernis", "apivorus"),
    ("Phoenicurus", "ochruros"),
    ("Phoenicurus", "phoenicurus"),
    ("Phylloscopus", "bonelli"),
    ("Phylloscopus", "collybita"),
    ("Phylloscopus", "sibilatrix"),
    ("Picus", "canus"),
    ("Picus", "viridis"),
    ("Poecile", "montanus"),
    ("Poecile", "palustris"),
    ("Prunella", "modularis"),
    ("Pyrrhula", "pyrrhula"),
    ("Regulus", "ignicapilla"),
    ("Regulus", "regulus"),
    ("Serinus", "serinus"),
    ("Sitta", "europaea"),
    ("Spinus", "spinus"),
    ("Strix", "aluco"),
    ("Sturnus", "vulgaris"),
    ("Sylvia", "atricapilla"),
    ("Troglodytes", "troglodytes"),
    ("Turdus", "merula"),
    ("Turdus", "philomelos"),
    ("Turdus", "pilaris"),
    ("Turdus", "viscivorus"),
    ("Upupa", "epops"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a compressed single-image photographic atlas for the thesis."
    )
    parser.add_argument(
        "--photos-dir",
        default="thesis/images/birdphotos",
        help="Directory containing per-species photos.",
    )
    parser.add_argument(
        "--output",
        default="thesis/images/generated/bird_species_atlas.jpg",
        help="Output JPEG path.",
    )
    parser.add_argument("--columns", type=int, default=7, help="Number of columns.")
    parser.add_argument("--cell-width", type=int, default=210, help="Cell width in px.")
    parser.add_argument("--image-height", type=int, default=140, help="Photo height in px.")
    parser.add_argument("--label-height", type=int, default=52, help="Label area height in px.")
    parser.add_argument("--gap-x", type=int, default=18, help="Horizontal gap in px.")
    parser.add_argument("--gap-y", type=int, default=18, help="Vertical gap in px.")
    parser.add_argument("--margin", type=int, default=28, help="Outer margin in px.")
    parser.add_argument("--quality", type=int, default=70, help="JPEG quality.")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/Library/Fonts/Times New Roman Italic.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for candidate in font_candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    center_x: int,
    top_y: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    draw.text((center_x - width / 2, top_y), text, font=font, fill=fill)


def open_or_placeholder(path: Path, width: int, height: int) -> Image.Image:
    if path.exists():
        image = Image.open(path).convert("RGB")
        # Preserve the full bird photo and pad with white margins when aspect ratios differ.
        return ImageOps.pad(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
            centering=(0.5, 0.5),
        )

    placeholder = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(placeholder)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(160, 160, 160), width=2)
    font = load_font(18)
    draw_centered_text(draw, "Missing", font, width // 2, height // 2 - 10, fill=(90, 90, 90))
    return placeholder


def main() -> int:
    args = parse_args()

    photos_dir = Path(args.photos_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = args.columns
    rows = (len(SPECIES) + cols - 1) // cols

    canvas_width = args.margin * 2 + cols * args.cell_width + (cols - 1) * args.gap_x
    cell_height = args.image_height + args.label_height
    canvas_height = args.margin * 2 + rows * cell_height + (rows - 1) * args.gap_y

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(24)

    for index, (genus, species) in enumerate(SPECIES):
        row = index // cols
        col = index % cols
        x = args.margin + col * (args.cell_width + args.gap_x)
        y = args.margin + row * (cell_height + args.gap_y)

        filename = f"{genus}_{species}.jpg"
        photo = open_or_placeholder(photos_dir / filename, args.cell_width, args.image_height)
        canvas.paste(photo, (x, y))

        center_x = x + args.cell_width // 2
        label_top = y + args.image_height + 8
        draw_centered_text(draw, genus, label_font, center_x, label_top)
        draw_centered_text(draw, species, label_font, center_x, label_top + 21)

    canvas.save(
        output_path,
        format="JPEG",
        quality=max(1, min(args.quality, 95)),
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )

    print(f"Saved atlas to {output_path}")
    print(f"Canvas: {canvas_width}x{canvas_height}px")
    print(f"Species: {len(SPECIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
