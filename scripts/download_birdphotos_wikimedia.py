#!/usr/bin/env python3
"""
Download one Wikimedia Commons bird photo per species.

Default output directory:
    thesis/images/birdphotos

The script:
1. searches Wikimedia Commons for each scientific name
2. picks the first raster image result with usable metadata
3. downloads it using the scientific name as local filename

Example:
    python3 thesis/scripts/download_birdphotos_wikimedia.py
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "tesi-birdphoto-downloader/1.0 (academic use)"
THUMB_WIDTH_CHOICES = {120, 250, 330, 500, 960, 1280, 1920}

SPECIES = [
    "Accipiter gentilis",
    "Aegithalos caudatus",
    "Aegolius funereus",
    "Apus apus",
    "Ardea cinerea",
    "Asio otus",
    "Athene noctua",
    "Bubo bubo",
    "Buteo buteo",
    "Caprimulgus europaeus",
    "Carduelis carduelis",
    "Certhia brachydactyla",
    "Certhia familiaris",
    "Chloris chloris",
    "Coccothraustes coccothraustes",
    "Columba palumbus",
    "Corvus corax",
    "Cuculus canorus",
    "Curruca communis",
    "Cyanistes caeruleus",
    "Delichon urbicum",
    "Dendrocopos major",
    "Dryobates minor",
    "Dryocopus martius",
    "Emberiza cia",
    "Emberiza cirlus",
    "Erithacus rubecula",
    "Falco tinnunculus",
    "Ficedula hypoleuca",
    "Fringilla coelebs",
    "Garrulus glandarius",
    "Hirundo rustica",
    "Jynx torquilla",
    "Lanius collurio",
    "Lophophanes cristatus",
    "Milvus milvus",
    "Motacilla alba",
    "Motacilla cinerea",
    "Muscicapa striata",
    "Oriolus oriolus",
    "Otus scops",
    "Parus major",
    "Passer montanus",
    "Periparus ater",
    "Pernis apivorus",
    "Phoenicurus ochruros",
    "Phoenicurus phoenicurus",
    "Phylloscopus bonelli",
    "Phylloscopus collybita",
    "Phylloscopus sibilatrix",
    "Picus canus",
    "Picus viridis",
    "Poecile montanus",
    "Poecile palustris",
    "Prunella modularis",
    "Pyrrhula pyrrhula",
    "Regulus ignicapilla",
    "Regulus regulus",
    "Serinus serinus",
    "Sitta europaea",
    "Spinus spinus",
    "Strix aluco",
    "Sturnus vulgaris",
    "Sylvia atricapilla",
    "Troglodytes troglodytes",
    "Turdus merula",
    "Turdus philomelos",
    "Turdus pilaris",
    "Turdus viscivorus",
    "Upupa epops",
]

RASTER_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/tiff": ".tif",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "thesis" / "images" / "birdphotos",
        help="Directory where the downloaded images will be saved.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay in seconds between species downloads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional species limit for testing.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save a download report as JSON.",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=500,
        help="Commons thumbnail width to request. Use a standard Wikimedia size such as 250, 330, 500, or 960.",
    )
    parser.add_argument(
        "--retry-429",
        type=int,
        default=4,
        help="Maximum number of retries for rate-limited requests.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=60.0,
        help="Base wait time in seconds after a 429 response.",
    )
    return parser.parse_args()


def scientific_name_to_stem(name: str) -> str:
    return name.replace(" ", "_")


def api_get(params: dict[str, str], retry_429: int = 0, retry_wait: float = 0.0) -> dict:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{COMMONS_API}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except HTTPError as exc:
            if exc.code != 429 or attempt >= retry_429:
                raise
            wait_s = retry_wait * (2**attempt)
            print(f"429 from Commons API, waiting {wait_s:.0f}s before retrying...", file=sys.stderr)
            time.sleep(wait_s)
            attempt += 1


def build_search_queries(species: str) -> list[str]:
    return [
        f'"{species}"',
        f'"{species}" bird',
        f'intitle:"{species}"',
        f'"{species}" filetype:bitmap',
    ]


def search_file_candidates(
    species: str,
    limit: int = 8,
    thumb_width: int = 500,
    retry_429: int = 0,
    retry_wait: float = 0.0,
) -> list[dict]:
    seen_titles: set[str] = set()
    candidates: list[dict] = []

    for query in build_search_queries(species):
        data = api_get(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": str(limit),
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "iiurlwidth": str(thumb_width),
            },
            retry_429=retry_429,
            retry_wait=retry_wait,
        )
        pages = data.get("query", {}).get("pages", {})
        for page in sorted(pages.values(), key=lambda p: p.get("index", 10**9)):
            title = page.get("title", "")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            imageinfo = page.get("imageinfo", [])
            if not imageinfo:
                continue
            info = imageinfo[0]
            mime = info.get("mime", "")
            if mime not in RASTER_MIME_TYPES:
                continue
            width = int(info.get("width", 0) or 0)
            height = int(info.get("height", 0) or 0)
            if width < 400 or height < 300:
                continue
            candidates.append(
                {
                    "title": title,
                    "url": info.get("thumburl", info["url"]),
                    "mime": mime,
                    "width": width,
                    "height": height,
                    "source_url": info["url"],
                }
            )
        if candidates:
            break

    return candidates


def pick_candidate(species: str, candidates: list[dict]) -> dict | None:
    species_lower = species.lower()

    def score(candidate: dict) -> tuple[int, int, int]:
        title = candidate["title"].lower()
        exact = 1 if species_lower in title else 0
        bitmap_bonus = 1 if candidate["mime"] == "image/jpeg" else 0
        pixels = candidate["width"] * candidate["height"]
        return (exact, bitmap_bonus, pixels)

    if not candidates:
        return None
    return max(candidates, key=score)


def download_file(url: str, out_path: Path, retry_429: int = 0, retry_wait: float = 0.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                out_path.write_bytes(resp.read())
                return
        except HTTPError as exc:
            if exc.code != 429 or attempt >= retry_429:
                raise
            wait_s = retry_wait * (2**attempt)
            print(f"429 while downloading image, waiting {wait_s:.0f}s before retrying...", file=sys.stderr)
            time.sleep(wait_s)
            attempt += 1


def existing_file_for_species(output_dir: Path, species: str) -> Path | None:
    stem = scientific_name_to_stem(species)
    matches = sorted(output_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def main() -> int:
    args = parse_args()
    if args.thumb_width not in THUMB_WIDTH_CHOICES:
        raise SystemExit(
            f"--thumb-width must be one of {sorted(THUMB_WIDTH_CHOICES)} to comply with Wikimedia thumbnail steps."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    species_list = SPECIES[: args.limit] if args.limit is not None else SPECIES
    report: list[dict[str, str]] = []

    print(f"Species count: {len(species_list)}")
    print(f"Output directory: {args.output_dir}")

    for idx, species in enumerate(species_list, start=1):
        stem = scientific_name_to_stem(species)
        existing = existing_file_for_species(args.output_dir, species)
        if existing and not args.overwrite:
            print(f"[{idx:02d}/{len(species_list)}] SKIP {species} -> {existing.name}")
            report.append(
                {
                    "species": species,
                    "status": "skipped",
                    "file": existing.name,
                    "source_title": "",
                    "source_url": "",
                }
            )
            continue

        try:
            candidates = search_file_candidates(
                species,
                thumb_width=args.thumb_width,
                retry_429=args.retry_429,
                retry_wait=args.retry_wait,
            )
            chosen = pick_candidate(species, candidates)
            if chosen is None:
                print(f"[{idx:02d}/{len(species_list)}] MISS {species}")
                report.append(
                    {
                        "species": species,
                        "status": "not_found",
                        "file": "",
                        "source_title": "",
                        "source_url": "",
                    }
                )
                time.sleep(args.delay)
                continue

            suffix = RASTER_MIME_TYPES.get(chosen["mime"]) or mimetypes.guess_extension(chosen["mime"]) or ".jpg"
            out_path = args.output_dir / f"{stem}{suffix}"
            download_file(
                chosen["url"],
                out_path,
                retry_429=args.retry_429,
                retry_wait=args.retry_wait,
            )
            print(f"[{idx:02d}/{len(species_list)}] OK   {species} -> {out_path.name}")
            report.append(
                {
                    "species": species,
                    "status": "downloaded",
                    "file": out_path.name,
                    "source_title": chosen["title"],
                    "source_url": chosen.get("source_url", chosen["url"]),
                }
            )
        except Exception as exc:
            print(f"[{idx:02d}/{len(species_list)}] ERR  {species} -> {exc}", file=sys.stderr)
            report.append(
                {
                    "species": species,
                    "status": "error",
                    "file": "",
                    "source_title": "",
                    "source_url": str(exc),
                }
            )

        time.sleep(args.delay)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved report: {args.save_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
