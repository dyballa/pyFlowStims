"""
FlowStims (Python port of FlowStims 2.2, Processing).

Generates the frames of a visual-stimulation movie (flows of dot patterns and
square-wave gratings) and saves them as grayscale PNG files, together with a
trials log that records which stimulus is shown in which frames.

Requirements: numpy, scipy, pillow.
Usage: edit the parameters below, then run `python flowstims.py`.
"""

import os
import sys
import time
import math
import datetime
import multiprocessing as mp

import numpy as np
from scipy.spatial import cKDTree
from PIL import Image


# =============================================================================
# PARAMETERS
# =============================================================================

# --- Output ------------------------------------------------------------------
OUTPUT_DIR = "movieframes"   # frames and logs are written here
PNG_COMPRESSION = 1          # 0 (fastest, largest files) to 9 (slowest, smallest)
N_WORKERS = os.cpu_count()   # trials are rendered in parallel

# --- Frame and geometry ------------------------------------------------------
WIDTH_PX = 1824
HEIGHT_PX = 1140
FRAME_RATE = 60              # frames per second of the final movie
PX_PER_DEG = 10              # set to None to compute it from the two values below
SCR_WIDTH_CM = 40.           # used only if PX_PER_DEG is None
SCR_DIST_CM = 25.            # used only if PX_PER_DEG is None
RANDOM_SEED = 19             # None for a random seed (the value used is logged)

# --- All stimuli -------------------------------------------------------------
TRIAL_LEN_SEC = 1.25         # stimulus presentation
PRE_STIM_LEN_SEC = 1.0       # pre-stimulus interval (at least 1 s recommended for flows)
POST_STIM_LEN_SEC = 0.25     # post-stimulus interval
N_TRIAL_BLOCKS = 1           # repetitions of every stimulus/direction combination
N_DIRS = 8                   # directions, evenly spaced around 360 deg
DIR_DEG_SHIFT = 0            # added to every direction (deg)
TEMP_FREQS = [4.]            # temporal frequencies (cycles/s)
N_FADE_FRAMES = 0            # frames for fade-in/out transitions (0 for none)
PHOTO_SQ_SIDE_PX = 0         # side of the photodiode square (bottom-right corner); 0 to disable

# Circular aperture. Each (horizontal, vertical) position pair is combined with each diameter.
USE_MASK = False
MASK_DIAMS_DEG = [20.]
MASK_HORZ_POS = [.5]         # fraction of frame width (0 = left)
MASK_VERT_POS = [.5]         # fraction of frame height (0 = top)
MASK_BORDER_PCT = .15        # width of the smooth edge, as a fraction of the diameter

# --- Flows -------------------------------------------------------------------
USE_FLOWS = True
N_DOTS = [3]                 # dots per flow element
DOT_FG_VALS = [8, 248, 8, 248]       # gray level of the dots [0-255]
DOT_BG_VALS = [128, 128, -1, -1]     # background; -1 for mean luminance 128
DOT_INTER_VALS = [128, 128, 128, 128]  # interstimulus level; -1 for mean luminance of the stimulus
DOT_DIAMS_DEG = [8, 4.2, 2.2, 1.1, 0.6, 0.3]    # diameter of single dots (deg)
DOT_SPACINGS = [2.25, 2.25, 2.25, 2.25, 2.0, 2.0]  # element spacing, in multiples of the diameter
DOT_SPAT_FREQS = None        # optional, one per diameter (cycles/deg); if None, the period is the spacing
EQUAL_AREA = True            # scale dot diameter by 1/sqrt(nDots) to keep the element area constant
RIGID_TRANS = False          # True: rigid translation during the trial (no separation or flow field)
SEP_WEIGHT = 1.5             # weight of the separation force relative to the flow-field force
MAX_FORCE_RATE = 0.9         # maximum steering force, in units of speed per second
SEP_INTERVAL_SEC = 1 / 12.   # interval between separation updates for each element
TILE_SIZE = 2.               # flow-field tile size, in multiples of the element spacing
DIR_STD = 0.09               # s.d. of the flow-field directions (rad)
POS_STD = 0.1                # s.d. of the initial positions, as a fraction of the spacing
FIX_RAND_STATE = False       # True: identical flow field and initial positions in every trial block
SAME_FLOWS = True            # True: all flow variations share flow fields and initial positions

# --- Gratings ----------------------------------------------------------------
USE_GRATINGS = True
GRAT_WIDTHS_DEG = [10, 5, 2.5, 1.25, 0.625, 0.301]  # bar width (deg) = 1/(2*spatial freq.)
GRAT_FG_VALS = [68]
GRAT_BG_VALS = [188]
GRAT_INTER_VALS = [128]      # -1 for the mean of fg and bg
GRAT_PHASE_STEPS = False     # True: initial phase steps evenly across trial blocks (block/N_TRIAL_BLOCKS)


# =============================================================================
# HELPERS
# =============================================================================

W, H = WIDTH_PX, HEIGHT_PX
_PIX_X = None   # pixel-centre coordinate maps, created lazily
_PIX_Y = None


def pixel_grids():
    global _PIX_X, _PIX_Y
    if _PIX_X is None:
        _PIX_Y, _PIX_X = np.mgrid[0:H, 0:W].astype(np.float64)
        _PIX_X += .5
        _PIX_Y += .5
    return _PIX_X, _PIX_Y


def limit(v, m):
    """Limit the magnitude of each row vector in v to m."""
    n = np.hypot(v[:, 0], v[:, 1])
    scale = np.where(n > m, m / np.maximum(n, 1e-12), 1.)
    return v * scale[:, None]


def splat_dots(cx, cy, r):
    """Antialiased coverage (0-1) of discs of radius r centred at (cx, cy).

    The coverage of each pixel is a linear ramp of 1 px width at the disc edge,
    with the ramp radius chosen so that the total coverage equals pi*r^2 exactly.
    """
    if r >= 1 / math.sqrt(3):
        c = math.sqrt(r * r - 1 / 12.) + .5
    else:
        c = (3 * r * r) ** (1 / 3.)
    h = int(math.ceil(c)) + 1
    keep = (cx > -c) & (cx < W + c) & (cy > -c) & (cy < H + c)
    cx, cy = cx[keep], cy[keep]
    cov = np.zeros(W * H)
    if cx.size == 0:
        return cov.reshape(H, W)
    offs = np.arange(-h, h + 1)
    x0 = np.floor(cx).astype(np.int64)
    y0 = np.floor(cy).astype(np.int64)
    # process in chunks to bound memory use
    chunk = max(1, 4_000_000 // offs.size ** 2)
    for s in range(0, cx.size, chunk):
        px = x0[s:s + chunk, None, None] + offs[None, None, :]
        py = y0[s:s + chunk, None, None] + offs[None, :, None]
        d = np.hypot(px + .5 - cx[s:s + chunk, None, None], py + .5 - cy[s:s + chunk, None, None])
        w = np.clip(c - d, 0., 1.)
        ok = (w > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        cov += np.bincount((py * W + px)[ok], weights=w[ok], minlength=W * H)
    return np.minimum(cov, 1.).reshape(H, W)


def make_mask(diam_px, cx, cy):
    """Aperture transparency map (1 inside, 0 outside) with a sigmoid edge."""
    X, Y = pixel_grids()
    R = diam_px / 2.
    d = R - np.hypot(X - cx, Y - cy)          # distance inward from the edge
    w = max(1, round(diam_px * MASK_BORDER_PCT))
    k = math.log(255.) / (w / 2.)
    a = np.minimum(256. / (1 + np.exp(-k * (d - w / 2.))), 255.) / 255.
    a[d < 0] = 0.
    return a


def compose(cov, alpha, fg, bg, gray):
    """Gray levels of a frame: fg drawn with coverage cov over a background that fades between bg and gray."""
    base = bg * alpha + gray * (1 - alpha)
    out = base + cov * alpha * (fg - base)
    if np.ndim(out) == 0:   # nothing drawn (stimulus fully faded out): uniform frame
        out = np.full((H, W), out)
    return out


# =============================================================================
# STIMULI
# =============================================================================

class Flow:

    def __init__(self, base_seed, tempfreq, ndots, diam_deg, spacing, sep_px, diam_px,
                 period_px, fg, bg, inter, mask):
        self.kind = "FLOW"
        self.base_seed = base_seed
        self.tempfreq = tempfreq
        self.p = ndots
        self.diam_deg = diam_deg
        self.spacing = spacing
        self.sep = sep_px
        self.D = diam_px
        self.r = diam_px / 2.
        self.fg = fg
        self.mask = mask    # (diam_px, x_px, y_px) or None

        # Lattice spacings: across the element axis (along the heading) and along it.
        # They differ by the extra element length, (p-1)*D, so that the gaps between
        # elements are equal in both directions.
        e = (ndots - 1) * diam_px
        if EQUAL_AREA:
            # product equal to sep^2: same element density as single-dot flows
            self.sp_head = (-e + math.sqrt(e * e + 4 * sep_px ** 2)) / 2
        else:
            # same gaps as single-dot flows (lower element density)
            self.sp_head = sep_px
        self.sp_axis = self.sp_head + e

        # background level for mean luminance 128
        n_dots = self.p * W * H / (self.sp_head * self.sp_axis)
        flow_area = n_dots * math.pi * diam_px ** 2 / 4
        bg_area = W * H - flow_area
        if bg == -1:
            bg = round((128 * W * H - flow_area * fg) / bg_area)
            if not 0 <= bg <= 255:
                sys.exit(f"Flow (nDots={ndots}, diam={diam_deg}, fg={fg}): the background level for "
                         f"mean luminance 128 is {bg}, outside [0, 255].")
        if inter == -1:
            inter = round((flow_area * fg + bg_area * bg) / (W * H))
        self.bg, self.inter = bg, inter

        self.wiggle = (not RIGID_TRANS) and POS_STD > 0
        if tempfreq == 0:
            self.wiggle = False
            self.max_speed = 3. * sep_px / FRAME_RATE   # settling speed during pre-stimulus
        else:
            if period_px is None:
                period_px = self.sp_head
            self.max_speed = tempfreq * period_px / FRAME_RATE
        self.max_force = MAX_FORCE_RATE * self.max_speed / FRAME_RATE
        self.pre_forces = POS_STD > 0 or tempfreq == 0
        self.sep_freq = max(1, round(SEP_INTERVAL_SEC * FRAME_RATE))

    def init(self, dir_deg, dir_idx, block):
        p, r, b = self.p, self.r, self.sep
        th = -math.radians(dir_deg)            # screen coordinates (y down)
        self.theta0 = th
        sin, cos = math.sin(th), math.cos(th)
        self.dir_deg = dir_deg

        sh, sa = self.sp_head, self.sp_axis
        # lattice spacing projected on x and y (exact for the cardinal directions,
        # so that the lattice tiles the wraparound domain without a seam)
        sx = sh * cos ** 2 + sa * sin ** 2
        sy = sh * sin ** 2 + sa * cos ** 2

        # wraparound domain, larger than the frame by at least one element extent
        ext_x = max(1, p * abs(sin)) * self.D
        ext_y = max(1, p * abs(cos)) * self.D
        ncols = max(int(W // sx) + 1, math.ceil((W + ext_x) / sx))
        nrows = max(int(H // sy) + 1, math.ceil((H + ext_y) / sy))
        self.Lx, self.Ly = ncols * sx, nrows * sy
        if p > 1:
            self.bfx, self.bfy = max(1, p * abs(sin)), max(1, p * abs(cos))
        else:
            self.bfx = self.bfy = 1.

        # elliptical separation perimeter: radius sh+1 along the heading, sa+1 along the element axis
        a, bb = (sh + 1) ** 2, (sa + 1) ** 2
        self.s1 = cos ** 2 / a + sin ** 2 / bb
        self.s2 = 2 * cos * sin * (1 / a - 1 / bb)
        self.s3 = sin ** 2 / a + cos ** 2 / bb
        self.sep_rmax = sa + 1

        # random state: flow field and initial positions use independent streams
        key = [self.base_seed, dir_idx] if FIX_RAND_STATE else [self.base_seed, block, dir_idx]
        rng_field = np.random.default_rng(key + [0])
        rng_pos = np.random.default_rng(key + [1])

        if DIR_STD > 0:
            self.tile = b * TILE_SIZE
            cols, rows = math.ceil(W / self.tile), math.ceil(H / self.tile)
            self.field = th + DIR_STD * rng_field.standard_normal((rows, cols))
        else:
            self.field = None

        # rectangular lattice aligned with the heading, clipped to the domain
        ox, oy = self.bfx * r, self.bfy * r
        m = int(math.ceil(math.hypot(self.Lx, self.Ly) / min(sh, sa))) + 1
        gi, gj = np.mgrid[-m:m + 1, -m:m + 1]
        px = gi.ravel() * sh * cos + gj.ravel() * sa * sin
        py = gi.ravel() * sh * sin - gj.ravel() * sa * cos
        inside = (px >= -ox) & (px < self.Lx - ox) & (py >= -oy) & (py < self.Ly - oy)
        n = int(inside.sum())
        pos = np.column_stack((px[inside], py[inside]))
        pos += POS_STD * b * rng_pos.standard_normal((n, 2))
        self.pos = pos
        self.wrap()

        self.v0 = np.array([cos, sin]) * self.max_speed
        self.vel = np.tile(self.v0, (n, 1))
        self.sepf = np.zeros((n, 2))
        self.theta = np.full(n, th)
        self.ids = np.arange(n)
        self.t = 0
        self.alpha = 0.

    def wrap(self):
        ox, oy = self.bfx * self.r, self.bfy * self.r
        self.pos[:, 0] = np.mod(self.pos[:, 0] + ox, self.Lx) - ox
        self.pos[:, 1] = np.mod(self.pos[:, 1] + oy, self.Ly) - oy

    def separation(self, idx):
        """Separation steering force for the elements in idx."""
        ox, oy = self.bfx * self.r, self.bfy * self.r
        pts = np.column_stack((np.mod(self.pos[:, 0] + ox, self.Lx), np.mod(self.pos[:, 1] + oy, self.Ly)))
        pts[pts[:, 0] >= self.Lx, 0] = 0.   # guard against rounding to exactly L
        pts[pts[:, 1] >= self.Ly, 1] = 0.
        tree = cKDTree(pts, boxsize=(self.Lx, self.Ly))
        pairs = tree.query_pairs(self.sep_rmax, output_type="ndarray")
        sel = np.zeros(len(pts), bool)
        sel[idx] = True
        steer = np.zeros((len(pts), 2))
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            keep = sel[i] | sel[j]
            i, j = i[keep], j[keep]
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            dx = (dx + self.Lx / 2) % self.Lx - self.Lx / 2
            dy = (dy + self.Ly / 2) % self.Ly - self.Ly / 2
            d2 = self.s1 * dx * dx + self.s2 * dx * dy + self.s3 * dy * dy
            ok = (d2 > 0) & (d2 < 1)
            i, j, dx, dy, d2 = i[ok], j[ok], dx[ok], dy[ok], d2[ok]
            mag = 1. / (d2 * np.hypot(dx, dy))
            fx, fy = dx * mag, dy * mag
            N = len(pts)
            steer[:, 0] = np.bincount(i, fx, N) - np.bincount(j, fx, N)
            steer[:, 1] = np.bincount(i, fy, N) - np.bincount(j, fy, N)
        steer = steer[idx]
        nz = np.hypot(steer[:, 0], steer[:, 1]) > 0
        out = np.zeros_like(steer)
        s = steer[nz]
        s = s / np.hypot(s[:, 0], s[:, 1])[:, None] * self.max_speed - self.vel[idx][nz]
        out[nz] = limit(s, self.max_force)
        return out

    def step(self, forces):
        if forces:
            acc = np.zeros_like(self.vel)
            if SEP_WEIGHT > 0:
                idx = self.ids[(self.ids + self.t + 1) % self.sep_freq == 0]
                if idx.size:
                    self.sepf[idx] = SEP_WEIGHT * self.separation(idx)
                acc += self.sepf
            if self.field is not None:
                rows, cols = self.field.shape
                c = np.clip((self.pos[:, 0] // self.tile).astype(int), 0, cols - 1)
                r = np.clip((self.pos[:, 1] // self.tile).astype(int), 0, rows - 1)
                f = self.field[r, c]
                desired = np.column_stack((np.cos(f), np.sin(f))) * self.max_speed - self.vel
            else:
                desired = self.v0 - self.vel
            desired = limit(desired, self.max_force)
            acc += desired
            if self.p > 1:
                h = self.vel + desired
                self.theta = np.arctan2(h[:, 1], h[:, 0])
            self.vel = limit(self.vel + acc, self.max_speed)
        elif self.tempfreq == 0:
            self.vel[:] = 0.
        else:
            self.vel[:] = self.v0
            self.theta[:] = self.theta0
        self.pos += self.vel
        self.wrap()
        self.t += 1

    def frame(self, period):
        """Advance one frame and return the gray levels. period: 'PRE', 'TRIAL' or 'POST'."""
        if period == "PRE":
            self.forces = self.pre_forces
        elif period == "TRIAL":
            self.forces = self.wiggle
        self.step(self.forces)
        fade = 1. / max(1, N_FADE_FRAMES)
        self.alpha = min(1., self.alpha + fade) if period == "TRIAL" else max(0., self.alpha - fade)

        cov = 0.
        if self.alpha > 0:
            x, y = self.pos[:, 0], self.pos[:, 1]
            if self.p == 1:
                cx, cy = x, y
            else:
                k = np.arange(-(self.p - 1), self.p, 2) * self.r
                ux, uy = np.sin(self.theta), -np.cos(self.theta)  # element axis, orthogonal to heading
                cx = (x[:, None] + ux[:, None] * k).ravel()
                cy = (y[:, None] + uy[:, None] * k).ravel()
            cov = splat_dots(cx, cy, self.r)
        return compose(cov, self.alpha, self.fg, self.bg, self.inter)

    def info(self):
        return ("stim=FLOW nDots=%d dir=%d tfreq=%.1f diam=%.2f spac=%.2f dotLvl=%d bgLvl=%d interLvl=%d"
                % (self.p, self.dir_deg, self.tempfreq, self.diam_deg, self.spacing, self.fg, self.bg, self.inter)
                + mask_info(self.mask))


class Grating:

    def __init__(self, tempfreq, width_deg, width_px, fg, bg, inter, mask):
        self.kind = "GRAT"
        self.tempfreq = tempfreq
        self.width_deg = width_deg
        self.bw = width_px
        self.P = 2 * width_px
        self.speed = tempfreq * self.P / FRAME_RATE
        self.fg, self.bg = fg, bg
        self.inter = round((fg + bg) / 2) if inter == -1 else inter
        self.mask = mask

    def init(self, dir_deg, dir_idx, block):
        self.dir_deg = dir_deg
        th = math.radians(dir_deg)
        X, Y = pixel_grids()
        # coordinate along the direction of motion (y axis pointing up, origin at bottom-left)
        self.u = X * math.cos(th) + (H - Y) * math.sin(th)
        self.phase = block / N_TRIAL_BLOCKS if GRAT_PHASE_STEPS else 0.
        self.t = None   # frames since trial onset
        self.alpha = 0.

    def frame(self, period):
        if period == "TRIAL" and self.t is None:
            self.t = 0
        fade = 1. / max(1, N_FADE_FRAMES)
        self.alpha = min(1., self.alpha + fade) if period == "TRIAL" else max(0., self.alpha - fade)
        cov = 0.
        if self.alpha > 0:
            offset = self.phase * self.P + self.speed * self.t
            s = np.mod(self.u - offset, self.P)
            # fraction of the pixel's 1-px footprint (along u) covered by a bar in [0, bw) or [P, P+bw)
            lo, hi = s - .5, s + .5
            cov = (np.clip(np.minimum(hi, self.bw) - np.maximum(lo, 0.), 0., 1.)
                   + np.clip(np.minimum(hi, self.P + self.bw) - np.maximum(lo, self.P), 0., 1.))
        if self.t is not None:
            self.t += 1
        return compose(cov, self.alpha, self.fg, self.bg, self.inter)

    def info(self):
        return ("stim=GRAT dir=%d tfreq=%.1f width=%.2f fgLvl=%d bgLvl=%d interLvl=%d phase=%.2f"
                % (self.dir_deg, self.tempfreq, self.width_deg, self.fg, self.bg, self.inter, self.phase)
                + mask_info(self.mask))


def mask_info(mask):
    if mask is None:
        return " maskD=0 maskX=0 maskY=0"
    return " maskD=%d maskX=%d maskY=%d" % tuple(round(m) for m in mask)


# =============================================================================
# STIMULUS LIST AND TRIAL SEQUENCE
# =============================================================================

def build_stims(rng, px_per_deg):
    masks = [None]
    if USE_MASK:
        masks = [(d * px_per_deg, hp * W, vp * H)
                 for d in MASK_DIAMS_DEG for hp, vp in zip(MASK_HORZ_POS, MASK_VERT_POS)]
    stims = []
    for tf in TEMP_FREQS:
        if USE_FLOWS:
            shared_seed = int(rng.integers(2 ** 31))
            for nd in N_DOTS:
                for sz, diam_deg in enumerate(DOT_DIAMS_DEG):
                    d1 = diam_deg * px_per_deg
                    sep_px = DOT_SPACINGS[sz] * d1   # based on the single-dot diameter
                    diam_px = d1 / math.sqrt(nd) if (nd > 1 and EQUAL_AREA) else d1
                    # None: the period is the lattice spacing along the direction of motion
                    period = None if DOT_SPAT_FREQS is None else px_per_deg / DOT_SPAT_FREQS[sz]
                    for fg, bg, inter in zip(DOT_FG_VALS, DOT_BG_VALS, DOT_INTER_VALS):
                        for m in masks:
                            seed = shared_seed if SAME_FLOWS else int(rng.integers(2 ** 31))
                            stims.append(Flow(seed, tf, nd, diam_deg, DOT_SPACINGS[sz], sep_px, diam_px,
                                              period, fg, bg, inter, m))
        if USE_GRATINGS:
            for wd in GRAT_WIDTHS_DEG:
                for fg, bg, inter in zip(GRAT_FG_VALS, GRAT_BG_VALS, GRAT_INTER_VALS):
                    for m in masks:
                        stims.append(Grating(tf, wd, wd * px_per_deg, fg, bg, inter, m))
    return stims


def build_trials(rng, n_stims):
    """List of trials (stim index, direction index, block), in presentation order."""
    dir_order = [list(range(N_DIRS)) for _ in range(n_stims)]
    trials = []
    for block in range(N_TRIAL_BLOCKS):
        for d in dir_order:
            rng.shuffle(d)
        for rnd in range(N_DIRS):
            for s in rng.permutation(n_stims):
                trials.append((int(s), dir_order[s][rnd], block))
    return trials


# =============================================================================
# RENDERING
# =============================================================================

PRE_LEN = round(PRE_STIM_LEN_SEC * FRAME_RATE)
TRIAL_LEN = round(TRIAL_LEN_SEC * FRAME_RATE)
POST_LEN = round(POST_STIM_LEN_SEC * FRAME_RATE)
TRIAL_FRAMES = PRE_LEN + TRIAL_LEN + POST_LEN
DIRS = [round(i * 360. / N_DIRS) + DIR_DEG_SHIFT for i in range(N_DIRS)]

STIMS = []        # set in main(); inherited by worker processes
MASKS = {}


def render_trial(args):
    trial_no, (s, di, block) = args
    stim = STIMS[s]
    stim.init(DIRS[di], di, block)
    alpha_mask = None
    if stim.mask is not None:
        alpha_mask = MASKS.setdefault(stim.mask, make_mask(*stim.mask))
    first = trial_no * TRIAL_FRAMES + 1
    periods = ["PRE"] * PRE_LEN + ["TRIAL"] * TRIAL_LEN + ["POST"] * POST_LEN
    for f, period in enumerate(periods):
        img = stim.frame(period)
        if alpha_mask is not None:
            img = stim.inter + alpha_mask * (img - stim.inter)
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        if PHOTO_SQ_SIDE_PX > 0:
            img[-PHOTO_SQ_SIDE_PX:, -PHOTO_SQ_SIDE_PX:] = 255 if period == "TRIAL" else 0
        Image.fromarray(img, "L").save(os.path.join(OUTPUT_DIR, "%06d.png" % (first + f)),
                                       compress_level=PNG_COMPRESSION)
    return trial_no


def main():
    global STIMS
    seed = RANDOM_SEED if RANDOM_SEED is not None else int(np.random.SeedSequence().entropy % 2 ** 31)
    rng = np.random.default_rng(seed)
    if PX_PER_DEG is None:
        px_per_deg = W / (2 * math.degrees(math.atan(.5 * SCR_WIDTH_CM / SCR_DIST_CM)))
    else:
        px_per_deg = PX_PER_DEG

    STIMS = build_stims(rng, px_per_deg)
    if not STIMS:
        sys.exit("No stimuli selected.")
    trials = build_trials(rng, len(STIMS))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    # parameters log
    with open(os.path.join(OUTPUT_DIR, stamp + "_params.log"), "w") as f:
        f.write("FlowStims-python (port of FlowStims 2.2)\n")
        for k, v in globals().items():
            if k.isupper() and not k.startswith("_") and k not in ("W", "H", "STIMS", "MASKS"):
                f.write("%s %s\n" % (k, v))
        f.write("seedUsed %d\npxPerDegUsed %.4f\n" % (seed, px_per_deg))

    # trials log (frame numbers match the PNG file names; time in ms from the first frame)
    with open(os.path.join(OUTPUT_DIR, stamp + "_trials.log"), "w") as f:
        f.write("Frame Time Period TrialNo Stimulus\n")
        ms = lambda fr: round((fr - 1) * 1000 / FRAME_RATE)
        for t, (s, di, block) in enumerate(trials):
            stim = STIMS[s]
            stim.dir_deg = DIRS[di]
            if stim.kind == "GRAT":
                stim.phase = block / N_TRIAL_BLOCKS if GRAT_PHASE_STEPS else 0.
            start = t * TRIAL_FRAMES + 1
            if PRE_LEN:
                f.write("%d %d PRESTIM\n" % (start, ms(start)))
            fr = start + PRE_LEN
            f.write("%d %d TRIAL %d %s\n" % (fr, ms(fr), t + 1, stim.info()))
            if POST_LEN:
                fr += TRIAL_LEN
                f.write("%d %d POSTSTIM\n" % (fr, ms(fr)))

    n = len(trials)
    print("%d stimuli, %d trials, %d frames -> %s" % (len(STIMS), n, n * TRIAL_FRAMES, OUTPUT_DIR))
    t0 = time.time()
    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        for k, _ in enumerate(pool.imap_unordered(render_trial, enumerate(trials)), 1):
            el = time.time() - t0
            print("\r%d/%d trials  (%.0f s elapsed, %.0f s left)" % (k, n, el, el / k * (n - k)),
                  end="", flush=True)
    print()


if __name__ == "__main__":
    main()