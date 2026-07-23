#!/usr/bin/env python3
"""Plot IrisK collapse-wave XCTS handoff convergence and AthenaK slices."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def font(size: int) -> ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TEXT = (31, 41, 55)
MUTED = (92, 105, 122)
GRID = (210, 216, 224)
BACKGROUND = (250, 251, 253)
SERIES = ((24, 114, 156), (218, 95, 43), (68, 153, 112), (147, 94, 178))

VIRIDIS = (
    (0.00, (68, 1, 84)),
    (0.20, (59, 82, 139)),
    (0.40, (33, 145, 140)),
    (0.60, (94, 201, 98)),
    (0.80, (202, 225, 31)),
    (1.00, (253, 231, 37)),
)

PLASMA = (
    (0.00, (13, 8, 135)),
    (0.20, (106, 0, 168)),
    (0.40, (177, 42, 144)),
    (0.60, (225, 100, 98)),
    (0.80, (252, 166, 54)),
    (1.00, (240, 249, 33)),
)


def color_table(anchors: tuple[tuple[float, tuple[int, int, int]], ...]) -> np.ndarray:
    table = np.empty((256, 3), dtype=np.uint8)
    for index in range(256):
        value = index / 255.0
        for left, right in zip(anchors[:-1], anchors[1:]):
            if left[0] <= value <= right[0]:
                fraction = (value - left[0]) / (right[0] - left[0])
                table[index] = np.rint(
                    (1.0 - fraction) * np.asarray(left[1])
                    + fraction * np.asarray(right[1])
                )
                break
    return table


def read_plane(path: Path) -> np.ndarray:
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(value) for value in line.split()])
    if not rows:
        raise ValueError(f"{path} has no plane data")
    return np.asarray(rows)


def rasterize(
    rows: np.ndarray,
    plane: str,
    values: np.ndarray,
    meshblock_size: int,
    pixels: int = 700,
) -> np.ndarray:
    domain_min, domain_max = -7.0, 7.0
    raster = np.full((pixels, pixels), np.nan)
    ownership = np.full((pixels, pixels), -1, dtype=np.int16)
    horizontal = rows[:, 0]
    vertical = rows[:, 1] if plane == "xy" else rows[:, 2]
    levels = rows[:, 10].astype(int)
    for index in np.argsort(levels):
        level = levels[index]
        spacing = (domain_max - domain_min) / (meshblock_size * 2**level)
        x0 = int(
            np.floor(
                (horizontal[index] - 0.5 * spacing - domain_min)
                / (domain_max - domain_min)
                * pixels
            )
        )
        x1 = int(
            np.ceil(
                (horizontal[index] + 0.5 * spacing - domain_min)
                / (domain_max - domain_min)
                * pixels
            )
        )
        y0 = int(
            np.floor(
                (domain_max - (vertical[index] + 0.5 * spacing))
                / (domain_max - domain_min)
                * pixels
            )
        )
        y1 = int(
            np.ceil(
                (domain_max - (vertical[index] - 0.5 * spacing))
                / (domain_max - domain_min)
                * pixels
            )
        )
        x0, x1 = max(0, x0), min(pixels, x1)
        y0, y1 = max(0, y0), min(pixels, y1)
        if x0 >= x1 or y0 >= y1:
            continue
        region = ownership[y0:y1, x0:x1]
        replace = level >= region
        raster_region = raster[y0:y1, x0:x1]
        raster_region[replace] = values[index]
        region[replace] = level
    return raster


def normalize(data: np.ndarray, lower: float | None = None, upper: float | None = None):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("slice raster has no finite values")
    if lower is None:
        lower = float(np.percentile(finite, 1.0))
    if upper is None:
        upper = float(np.percentile(finite, 99.0))
    if not upper > lower:
        upper = lower + max(abs(lower), 1.0) * 1.0e-12
    scaled = np.clip((data - lower) / (upper - lower), 0.0, 1.0)
    return scaled, lower, upper


def heatmap_panel(
    raster: np.ndarray,
    title: str,
    unit: str,
    palette: np.ndarray,
    width: int = 440,
    height: int = 350,
) -> Image.Image:
    panel = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    title_font, tick_font = font(18), font(13)
    left, top, plot_size = 45, 34, 270
    normalized, lower, upper = normalize(raster)
    colors = palette[np.rint(np.nan_to_num(normalized) * 255).astype(np.uint8)]
    colors[~np.isfinite(raster)] = np.asarray(BACKGROUND)
    image = Image.fromarray(colors, mode="RGB").resize(
        (plot_size, plot_size), Image.Resampling.NEAREST
    )
    panel.paste(image, (left, top))
    draw.rectangle(
        (left, top, left + plot_size, top + plot_size), outline=GRID, width=1
    )
    draw.text((left, 7), title, fill=TEXT, font=title_font)
    for value in (-7, 0, 7):
        x = left + int((value + 7) / 14 * plot_size)
        y = top + plot_size - int((value + 7) / 14 * plot_size)
        draw.line((x, top + plot_size, x, top + plot_size + 5), fill=MUTED, width=1)
        draw.text((x - 8, top + plot_size + 8), str(value), fill=MUTED, font=tick_font)
        draw.line((left - 5, y, left, y), fill=MUTED, width=1)
        draw.text((8, y - 7), str(value), fill=MUTED, font=tick_font)
    bar_x, bar_y, bar_w, bar_h = left + plot_size + 18, top, 16, plot_size
    gradient = palette[np.arange(255, -1, -1)].reshape(256, 1, 3)
    bar = Image.fromarray(gradient, mode="RGB").resize((bar_w, bar_h))
    panel.paste(bar, (bar_x, bar_y))
    draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), outline=GRID, width=1)
    draw.text((bar_x + 22, bar_y - 4), f"{upper:.3e}", fill=MUTED, font=tick_font)
    draw.text((bar_x + 22, bar_y + bar_h - 13), f"{lower:.3e}", fill=MUTED, font=tick_font)
    if unit:
        draw.text((bar_x + 22, bar_y + bar_h // 2 - 7), unit, fill=MUTED, font=tick_font)
    return panel


def make_slice_plot(
    input_path: Path,
    output_path: Path,
    plane: str,
    meshblock_size: int,
) -> None:
    rows = read_plane(input_path)
    epsilon = 1.0e-18
    fields = (
        ("ψ", rows[:, 3], ""),
        ("α", rows[:, 4], ""),
        ("|β|", rows[:, 5], ""),
        ("SMR level", rows[:, 10], ""),
        ("log10 |H|", np.log10(np.maximum(np.abs(rows[:, 6]), epsilon)), ""),
        ("log10 |M|", np.log10(np.maximum(rows[:, 7], epsilon)), ""),
        ("log10 C", np.log10(np.maximum(rows[:, 8], epsilon)), ""),
        ("log10 Z", np.log10(np.maximum(rows[:, 9], epsilon)), ""),
    )
    viridis, plasma = color_table(VIRIDIS), color_table(PLASMA)
    panels = []
    for index, (title, values, unit) in enumerate(fields):
        raster = rasterize(rows, plane, values, meshblock_size)
        panels.append(
            heatmap_panel(raster, title, unit, viridis if index < 4 else plasma)
        )
    width, height = 4 * panels[0].width, 2 * panels[0].height
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    for index, panel_image in enumerate(panels):
        canvas.paste(
            panel_image,
            ((index % 4) * panel_image.width, (index // 4) * panel_image.height),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(180, 180))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def log_chart(
    title: str,
    series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    x_label: str,
    width: int = 520,
    height: int = 410,
) -> Image.Image:
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 76, width - 30, 48, height - 62
    title_font, label_font, tick_font = font(18), font(14), font(12)
    draw.text((left, 12), title, fill=TEXT, font=title_font)
    lx0, lx1 = math.log(x_limits[0]), math.log(x_limits[1])
    ly0, ly1 = math.log10(y_limits[0]), math.log10(y_limits[1])

    def xpixel(value: float) -> float:
        return left + (math.log(value) - lx0) / (lx1 - lx0) * (right - left)

    def ypixel(value: float) -> float:
        return bottom - (math.log10(value) - ly0) / (ly1 - ly0) * (bottom - top)

    start_decade = math.floor(ly0)
    end_decade = math.ceil(ly1)
    for exponent in range(start_decade, end_decade + 1):
        value = 10.0**exponent
        if y_limits[0] <= value <= y_limits[1]:
            y = ypixel(value)
            draw.line((left, y, right, y), fill=GRID, width=1)
            draw.text((9, y - 7), f"1e{exponent}", fill=MUTED, font=tick_font)
    for value in sorted({point[0] for _, points, _ in series for point in points}):
        if x_limits[0] <= value <= x_limits[1]:
            x = xpixel(value)
            draw.line((x, top, x, bottom), fill=GRID, width=1)
            draw.text((x - 9, bottom + 8), f"{value:g}", fill=MUTED, font=tick_font)
    draw.line((left, bottom, right, bottom), fill=MUTED, width=2)
    draw.line((left, top, left, bottom), fill=MUTED, width=2)
    draw.text(((left + right) // 2 - 45, height - 28), x_label, fill=MUTED, font=label_font)
    for name, points, color in series:
        pixels = [(xpixel(x), ypixel(y)) for x, y in points]
        draw.line(pixels, fill=color, width=3, joint="curve")
        for x, y in pixels:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=BACKGROUND)
        x, y = pixels[-1]
        draw.text((min(x + 7, right - 60), y - 9), name, fill=color, font=label_font)
    return image


def make_convergence_plot(
    dg_path: Path,
    fd_path: Path,
    full_path: Path,
    output_path: Path,
) -> None:
    dg = read_csv(dg_path)
    fd = read_csv(fd_path)
    full = read_csv(full_path)
    dg_series = [
        (
            "H",
            [(float(row["order"]), float(row["overcollocated_hamiltonian_l2"])) for row in dg],
            SERIES[0],
        ),
        (
            "M",
            [(float(row["order"]), float(row["overcollocated_momentum_l2"])) for row in dg],
            SERIES[1],
        ),
    ]
    difference_series = []
    for formal_order, color in ((2, SERIES[0]), (4, SERIES[1]), (6, SERIES[2])):
        rows = [row for row in fd if int(row["formal_order"]) == formal_order]
        difference_series.append(
            (
                f"{formal_order}th" if formal_order != 2 else "2nd",
                [(float(row["meshblock_n"]), float(row["hamiltonian"])) for row in rows],
                color,
            )
        )
    full_series = []
    for spatial_order, color in ((2, SERIES[0]), (4, SERIES[1]), (6, SERIES[2])):
        rows = [row for row in full if int(row["spatial_order"]) == spatial_order]
        full_series.append(
            (
                f"FD {spatial_order}",
                [(float(row["meshblock_n"]), float(row["h_rms"])) for row in rows],
                color,
            )
        )
    panels = (
        log_chart("IrisK DG p-convergence, A=1", dg_series, (3.8, 6.3), (2e-4, 1.4e-3), "DG order N"),
        log_chart("AthenaK pulse-patch truncation", difference_series, (10, 78), (2e-11, 7e-5), "MeshBlock cells"),
        log_chart("Full-domain constraint floor", full_series, (10, 78), (2e-4, 4e-3), "MeshBlock cells"),
    )
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), BACKGROUND)
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(180, 180))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xy", type=Path, required=True)
    parser.add_argument("--xz", type=Path, required=True)
    parser.add_argument("--dg-convergence", type=Path, required=True)
    parser.add_argument("--fd-convergence", type=Path, required=True)
    parser.add_argument("--full-constraints", type=Path, required=True)
    parser.add_argument("--meshblock-size", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_slice_plot(args.xy, args.output_dir / "athenak_xy_fields_constraints.png", "xy", args.meshblock_size)
    make_slice_plot(args.xz, args.output_dir / "athenak_xz_fields_constraints.png", "xz", args.meshblock_size)
    make_convergence_plot(
        args.dg_convergence,
        args.fd_convergence,
        args.full_constraints,
        args.output_dir / "xcts_athenak_convergence.png",
    )


if __name__ == "__main__":
    main()
