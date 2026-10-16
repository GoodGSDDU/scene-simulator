"""Generate the application icon.

    python packaging/make_icon.py

Writes ``packaging/icon.ico`` (multi-resolution, 16..256 px) plus PNG previews.
Drawn with PIL so the icon stays reproducible and no binary needs committing.

The design is deliberately sparse -- it has to stay legible at 16 px on a
taskbar.  Everything is drawn on one supersampled layer and downscaled once so
the anti-aliasing is uniform.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SIZE = 512
SS = 8                                   # supersample factor

# --- palette: matches src/radar_sim/style.py -------------------------------- #
PLATE_TOP = (26, 36, 50)
PLATE_BOTTOM = (10, 14, 21)
RING = (78, 104, 128)
SECTOR = (26, 52, 64)
HOT = (124, 255, 174)
COLD = (30, 150, 116)
BLIP = (255, 208, 92)
CAR = (236, 244, 252)
GLASS = (118, 138, 160)


def _lerp(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def build() -> Image.Image:
    s = SIZE * SS
    ox, oy = s * 0.500, s * 0.790            # radar phase centre
    r_max = s * 0.560

    # ------------------------------------------------------------- background
    radius = int(s * 0.205)
    plate_mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(plate_mask).rounded_rectangle([0, 0, s - 1, s - 1],
                                                 radius=radius, fill=255)

    gradient = Image.new("RGBA", (1, s))
    gd = ImageDraw.Draw(gradient)
    for y in range(s):
        gd.point((0, y), fill=_lerp(PLATE_TOP, PLATE_BOTTOM, y / (s - 1)) + (255,))
    plate = gradient.resize((s, s), Image.BILINEAR)

    layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # --------------------------------------------------------------- the fan
    box = [ox - r_max, oy - r_max, ox + r_max, oy + r_max]
    d.pieslice(box, start=180, end=360, fill=SECTOR + (255,))

    # sweep: 220 thin wedges so the cold->hot ramp has no visible banding
    centre, half = 244.0, 46.0
    steps = 220
    sweep = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sweep)
    for i in range(steps):
        t1 = (i + 1) / steps
        a0 = centre - half + 2 * half * (i / steps)
        a1 = centre - half + 2 * half * t1
        sd.pieslice(box, start=a0, end=a1,
                    fill=_lerp(COLD, HOT, t1 ** 1.9) + (int(16 + 234 * t1 ** 2.6),))
    layer = Image.alpha_composite(layer, sweep)
    d = ImageDraw.Draw(layer)

    # ------------------------------------------------------------- range rings
    lw = max(2, int(s * 0.0090))
    for frac in (0.40, 0.70, 1.00):
        r = r_max * frac
        d.arc([ox - r, oy - r, ox + r, oy + r], start=180, end=360,
              fill=RING + (190,), width=lw)

    # leading edge of the sweep
    lead = math.radians(centre + half)
    d.line([ox, oy, ox + r_max * math.cos(lead), oy + r_max * math.sin(lead)],
           fill=HOT + (245,), width=max(3, int(s * 0.0125)))

    # ------------------------------------------------------------------ blips
    for dx, dy, rr in ((-0.44, 0.36, 0.062), (0.06, 0.62, 0.046),
                       (0.38, 0.28, 0.056)):
        bx, by = ox + dx * r_max, oy - dy * r_max
        r = rr * r_max
        d.ellipse([bx - r * 2.6, by - r * 2.6, bx + r * 2.6, by + r * 2.6],
                  fill=BLIP + (46,))
        d.ellipse([bx - r, by - r, bx + r, by + r], fill=BLIP + (255,))

    # ---------------------------------------------------------------- vehicle
    cw, ch = s * 0.190, s * 0.098
    cx0, cy0 = ox - cw / 2, oy - ch * 0.42
    d.rounded_rectangle([cx0, cy0, cx0 + cw, cy0 + ch],
                        radius=int(ch * 0.30), fill=CAR + (255,))
    d.rounded_rectangle([cx0 + cw * 0.58, cy0 + ch * 0.16,
                         cx0 + cw * 0.86, cy0 + ch * 0.84],
                        radius=int(ch * 0.16), fill=GLASS + (255,))
    # nose, so the car does not read as a plain pill
    d.polygon([(cx0 + cw, cy0 + ch * 0.5),
               (cx0 + cw * 0.88, cy0 + ch * 0.20),
               (cx0 + cw * 0.88, cy0 + ch * 0.80)], fill=GLASS + (255,))

    out = Image.alpha_composite(plate, layer)
    clipped = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    clipped.paste(out, (0, 0), plate_mask)
    return clipped.resize((SIZE, SIZE), Image.LANCZOS)


def main() -> int:
    icon = build()
    target = HERE / "icon.ico"
    icon.save(target, format="ICO",
              sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                     (64, 64), (128, 128), (256, 256)])
    icon.save(HERE / "icon.png")
    icon.resize((64, 64), Image.LANCZOS).save(HERE / "icon_preview_64.png")
    icon.resize((32, 32), Image.LANCZOS).save(HERE / "icon_preview_32.png")
    print(f"wrote {target}  ({target.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
