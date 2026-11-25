"""Plot the matched StellarAI embedded-torus histories and render a midplane movie."""

import argparse
import sys
import subprocess
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "vis/python"))
from bin_convert import read_binary
from athena_read import hst

m = 1 / 101
xs = -25 * 100 / 101
primary = 25 / 101
rhor = 1 + np.sqrt(1 - 0.7**2)
rhoxy = np.hypot(rhor, 0.7)
gfac = 1 / np.sqrt(1 - (xs * 0.008) ** 2)
teal = "#006e75"
red = "#b74720"
dark = "#1d3039"
muted = "#546976"
plt.rcParams.update(
    {
        "font.size": 10,
        "text.color": dark,
        "axes.labelcolor": dark,
        "xtick.color": muted,
        "ytick.color": muted,
        "axes.edgecolor": "#8fa1aa",
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def raster(s, xmid, width, field, n=480):
    x = xmid + ((np.arange(n) + 0.5) * 2 / n - 1) * width
    y = ((np.arange(n) + 0.5) * 2 / n - 1) * width
    img = np.full((n, n), np.nan)
    for geo, a in zip(s["mb_geometry"], s["mb_data"][field]):
        lo, hi, bot, top = geo[:4]
        i, j = np.searchsorted(x, [lo, hi])
        k, l = np.searchsorted(y, [bot, top])
        if i == j or k == l:
            continue
        dat = a[0]
        ix = np.floor((x[i:j] - lo) * dat.shape[1] / (hi - lo)).astype(int)
        iy = np.floor((y[k:l] - bot) * dat.shape[0] / (top - bot)).astype(int)
        img[k:l, i:j] = dat[np.ix_(iy, ix)]
    assert np.isfinite(img).all()
    return img


def movie(unequal, out, frames_dir):
    import imageio_ffmpeg

    paths = [
        sorted((unequal / case / "bin").glob("*midplane*.bin")) for case in ("fixed", "cyclic")
    ]
    if len(paths[0]) != len(paths[1]) or len(paths[0]) < 2:
        raise ValueError("Expected matched fixed and cyclic midplane sequences")
    starts = [read_binary(str(p[0])) for p in paths]
    for s in starts:
        ix = np.lexsort(s["mb_logical"].T)
        s["keys"] = s["mb_logical"][ix]
        s["fields"] = {k: np.asarray(v)[ix] for k, v in s["mb_data"].items()}
    np.testing.assert_array_equal(starts[0]["keys"], starts[1]["keys"])
    for k in starts[0]["fields"]:
        np.testing.assert_array_equal(starts[0]["fields"][k], starts[1]["fields"][k])

    nzones = int(
        next(
            line.split("=")[1]
            for line in starts[1]["header"]
            if line.split("=")[0].strip() == "nlevels"
        )
    )
    frames = []
    for fixed, cycle in zip(*paths):
        ss = [read_binary(str(f)) for f in (fixed, cycle)]
        t0, t1 = [s["time"] for s in ss]
        assert abs(t0 - t1) < 0.003 and all(
            np.isfinite(a).all() for s in ss for a in (np.asarray(v) for v in s["mb_data"].values())
        )
        if frames and abs(frames[-1] - t1) < 1e-7:
            continue
        n = len(frames)
        frames.append(t1)
        # The leaf at the puncture measures coarsening on the actual saved slice.
        levels = []
        for snapshot in ss:
            for geometry, logical in zip(snapshot["mb_geometry"], snapshot["mb_logical"]):
                xl, xu, yl, yu = geometry[:4]
                if xl <= xs < xu and yl <= 0 < yu:
                    levels.append(logical[3])
                    break
        if len(levels) != 2:
            raise ValueError("Missing midplane leaf at the secondary")
        zone = levels[0] - levels[1]
        fig = plt.figure(figsize=(10, 8.2), dpi=128)
        fig.text(0.10, 0.953, "The embedded secondary", fontsize=21, weight="bold")
        fig.text(
            0.10,
            0.912,
            f"100:1 binary   •   {nzones}-zone sweep   •   Corotating midplane   •   White ellipse marks the modeled horizon",
            color=muted,
        )
        for j, (s, time) in enumerate(zip(ss, (t0, t1))):
            left = 0.10 + 0.405 * j
            fig.text(
                left,
                0.85,
                ("Fixed finest" if j == 0 else f"Cyclic zoom · zone {zone}")
                + f"    t = {time:.3f} M",
                fontsize=12,
                weight="bold",
            )
            for row, (field, limits, cmap, label) in enumerate(
                [
                    ("dens", (-1.5, 1.5), "magma", r"$\log_{10}\,\rho$"),
                    ("bcc3", (-3.4, -0.4), "viridis", r"$\log_{10}\,|\tilde{B}^{z}|$"),
                ]
            ):
                ax = fig.add_axes([left, 0.493 - 0.399 * row, 0.345, 0.323])
                a = raster(s, xs, 20 * m, field)
                im = ax.imshow(
                    np.log10(np.maximum(abs(a), 1.0e-30)),
                    origin="lower",
                    extent=(-20, 20, -20, 20),
                    cmap=cmap,
                    vmin=limits[0],
                    vmax=limits[1],
                    interpolation="nearest",
                )
                ax.add_patch(
                    Ellipse((0, 0), 2 * rhoxy, 2 * rhoxy / gfac, fc="none", ec="white", lw=0.95)
                )
                if j == 0:
                    ax.set_ylabel(r"$y/m_2$")
                if row:
                    ax.set_xlabel(r"$(x-x_2)/m_2$")
                else:
                    ax.tick_params(labelbottom=False)
                if j == 1:
                    cax = fig.add_axes([0.885, 0.518 - 0.399 * row, 0.016, 0.267])
                    cb = fig.colorbar(im, cax=cax, extend="both")
                    cb.set_label(label)
        fig.text(
            0.10,
            0.025,
            "Lengths in secondary masses. Colors are fixed across time and both runs. Field is the stored densitized vertical component.",
            fontsize=8.4,
            color=muted,
        )
        fig.savefig(frames_dir / f"step-{n:03d}.png")
        plt.close(fig)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ff,
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "3/2",
            "-i",
            str(frames_dir / "step-%03d.png"),
            "-frames:v",
            str(16 * len(frames)),
            "-vf",
            "fps=24,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(out / "embedded-secondary.mp4"),
        ],
        check=True,
    )
    print(
        "Movie physical times",
        frames,
        "encoded frames/duration",
        imageio_ffmpeg.count_frames_and_secs(str(out / "embedded-secondary.mp4")),
    )


def summary(unequal, equal, shallow, out):
    f = hst(unequal / "fixed/embedded.user.hst")
    c = hst(unequal / "cyclic/embedded.user.hst")
    short = hst(shallow / "cyclic/embedded.user.hst")
    reports = [json.loads((p / "report.json").read_text()) for p in (unequal, shallow, equal)]
    fig = plt.figure(figsize=(12.8, 8.8), dpi=150)
    fig.text(0.08, 0.944, "Cyclic zoom around an embedded secondary", weight="bold", fontsize=22)
    fig.text(
        0.08,
        0.903,
        "Two B200 GPUs per run   •   Fixed circular orbit at 25M   •   One vertical-field torus with pressure maximum at 28M",
        color=muted,
        fontsize=11,
    )
    ax = fig.add_axes([0.075, 0.514, 0.32, 0.317])
    s = read_binary(str(next((unequal / "cyclic/bin").glob("*00000.bin"))))
    density = raster(s, 0, 65, "dens", 640)
    im = ax.imshow(
        np.log10(np.maximum(density, 1.0e-12)),
        cmap="magma",
        extent=(-65, 65, -65, 65),
        vmin=-5,
        vmax=0,
        origin="lower",
    )
    ax.scatter([primary, xs], [0, 0], marker="+", s=72, lw=3.1, c="#202a30")
    ax.scatter([primary, xs], [0, 0], marker="+", s=54, lw=1.2, c="white")
    for x, name, xy in [(primary, "Primary", (13, 23)), (xs, "Secondary", (-58, -36))]:
        ax.annotate(
            name,
            xy=(x, 0),
            xytext=xy,
            color="white",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="white", lw=0.65),
            bbox=dict(boxstyle="round,pad=.15", fc="#29232e", alpha=0.75, ec="none"),
        )
    ax.set(xlabel="x / M", ylabel="y / M")
    ax.set_title("100:1 initial midplane", loc="left", fontsize=12, weight="bold", pad=8)
    cb = fig.colorbar(im, cax=fig.add_axes([0.399, 0.543, 0.012, 0.255]))
    cb.set_label(r"$\log_{10}\,\rho$")
    ax = fig.add_axes([0.555, 0.524, 0.398, 0.30])
    ax.set_title(
        "Projected cost of one binary orbit", loc="left", fontsize=12, weight="bold", pad=10
    )
    styles = [
        ("Fixed finest", red, "-"),
        ("4-zone zoom", teal, "--"),
        ("2-zone zoom", "#7254a2", "-."),
    ]
    for pos, (left, right) in enumerate([(reports[0], reports[1]), (reports[2], None)]):
        y = 1 - pos
        costs = [
            (case["projected_B200_hours_per_binary_orbit"], style)
            for case, style in zip(
                [left["fixed"], left["cyclic"]] + ([right["cyclic"]] if right else []), styles
            )
        ]
        heights = [y, y, y] if right else [y + 0.055, y - 0.055]
        xy = sorted((v, h) for (v, _), h in zip(costs, heights))
        ax.plot(*zip(*xy), lw=1.4, color="#c0c9cc")
        for k, (v, (label, col, _)) in enumerate(costs):
            ax.scatter(
                v,
                heights[k],
                s=45,
                color=col,
                marker=["o", "s", "D"][k],
                label=label if pos == 0 else None,
                zorder=3,
            )
            offset = (0, 9) if k == 0 else (0, -17)
            ax.annotate(
                f"{v:.2f}" if v < 10 else f"{v:.1f}",
                (v, heights[k]),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=col,
            )
    ax.set_yticks([1, 0], ["100:1", "1:1"])
    ax.set_ylim(-0.6, 1.65)
    ax.set_xscale("log")
    ax.set_xlim(
        left=min(r["cyclic"]["projected_B200_hours_per_binary_orbit"] for r in reports) * 0.65,
        right=reports[0]["fixed"]["projected_B200_hours_per_binary_orbit"] * 1.65,
    )
    ax.set_xlabel("B200 GPU-hours per binary orbit · logarithmic scale")
    ax.legend(frameon=False, ncols=3, loc="upper left", fontsize=8)
    ax.grid(axis="x", lw=0.4, alpha=0.35, which="both")
    ax.set_axisbelow(True)
    for i, (key, surface, title) in enumerate(
        [
            ("phiB_H2", "H2", "Transfer sphere at 20 secondary masses"),
            ("phiB_BH2_Kerr", "BH2_Kerr", "Modeled secondary horizon"),
        ]
    ):
        ax = fig.add_axes([0.081 + 0.474 * i, 0.13, 0.396, 0.259])
        for h, (name, col, style) in zip((f, c, short), styles):
            sel = h["valid_" + surface] == 1
            if surface == "BH2_Kerr" and name != "Fixed finest":
                sel &= h["zoom_zone"] == 0
            ax.plot(
                h["time"] / m,
                1e4 * np.where(sel, h[key], np.nan),
                c=col,
                label=name,
                ls=style,
                lw=1.55 if name != "Fixed finest" else 1.2,
            )
        ax.set(
            title=title,
            xlabel=r"$t/m_2$",
            ylabel=r"$10^4\,\Phi_B$  [code units]",
            xlim=(0, c["time"][-1] / m),
        )
        ax.title.set_fontsize(12)
        ax.title.set_weight("bold")
        ax.grid(lw=0.4, alpha=0.35)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(
        0.081,
        0.049,
        "100:1 flux histories. Horizon curves show qualified finest visits. Each cost extrapolates the measured startup cadence.",
        fontsize=9,
        color=muted,
    )
    fig.savefig(out / "embedded-comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unequal", type=Path, help="100:1 pair, four-zone for the summary plot")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--movie", action="store_true", help="Render the four-zone and fixed matched slices"
    )
    parser.add_argument(
        "--summary",
        nargs=2,
        type=Path,
        metavar=("EQUAL", "SHALLOW"),
        help="Include the 1:1 and shallower 100:1 timings",
    )
    args = parser.parse_args()
    if not args.movie and not args.summary:
        parser.error("Select --movie or --summary")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.movie:
        with TemporaryDirectory() as frames_dir:
            movie(args.unequal, args.output, Path(frames_dir))
    if args.summary:
        summary(args.unequal, *args.summary, args.output)
