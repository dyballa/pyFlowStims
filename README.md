# FlowStims (Python)

A revised Python version of [FlowStims](https://github.com/zuckerlab/FlowStims), aimed at generating stimulus movies offline. It renders flow stimuli (fields of moving dot patterns) and square-wave gratings, and saves every frame as a PNG file together with a trials log that records which stimulus appears in which frames. The frames can then be assembled into a movie and played back with any software that logs frame times.

Unlike the original, this version does not display stimuli on screen in real time and has no network (UDP) communication. All parameters are set as variables at the top of a single script, `flowstims.py`.

The original FlowStims was developed at the Yale Computational Vision Lab by Luciano Dyballa under the guidance of Professor Steven W. Zucker. More information on the flow stimuli can be found in:

Dyballa L., Hoseini M.S., Dadarlat M.C., Zucker S.W., Stryker M.P., **Flow stimuli reveal ecologically appropriate responses in mouse visual cortex**, PNAS, 2018. [10.1073/pnas.1811265115](https://doi.org/10.1073/pnas.1811265115)

## Installation

The script requires Python 3.10 or later with numpy, scipy and pillow. It runs on Linux and macOS; Windows is not currently supported.

With [pixi](https://pixi.sh), from the repository folder:

```
pixi install
```

Alternatively, with pip:

```
pip install numpy scipy pillow
```

## Usage

1. Edit the parameters in the `PARAMETERS` section at the top of `flowstims.py`.
2. Run the script:

   ```
   pixi run render
   ```

   or, without pixi, `python flowstims.py`.

The script prints the number of stimuli, trials and frames, and shows its progress. Trials are rendered in parallel (`N_WORKERS`, by default the number of CPU cores).

Note that the total size of the frames grows quickly. At 1824×1140 pixels, each frame takes roughly 0.1–1 MB, and a movie of 240 trials of 2.5 s at 60 fps has 36,000 frames.

## Parameters

Parameters are grouped in the script as follows; each one is described by a comment next to it.

- **Output:** output folder, PNG compression level, number of parallel workers.
- **Frame and geometry:** frame size in pixels, frame rate, pixels per degree of visual angle (set directly, or computed from the screen width and viewing distance), random seed.
- **All stimuli:** durations of the pre-stimulus, trial and post-stimulus periods; number of trial blocks; number of directions; temporal frequencies; fade-in/out frames; photodiode square; circular aperture.
- **Flows:** number of dots per element, gray levels, dot diameters and spacings, equal-area option, rigid translation, flow-field and separation-force settings, randomization options.
- **Gratings:** bar widths, gray levels, phase stepping across trial blocks.

Parameters that take lists (e.g. `TEMP_FREQS`, `DOT_DIAMS_DEG`) produce one stimulus variation per value, and all variations are combined. Lists that belong together must have the same length: `DOT_FG_VALS`, `DOT_BG_VALS` and `DOT_INTER_VALS`; `DOT_DIAMS_DEG`, `DOT_SPACINGS` and `DOT_SPAT_FREQS`; `GRAT_FG_VALS`, `GRAT_BG_VALS` and `GRAT_INTER_VALS`; `MASK_HORZ_POS` and `MASK_VERT_POS`.

The total number of trials is `N_TRIAL_BLOCKS × N_DIRS × (number of stimulus variations)`. Within each block, every stimulus variation is shown once before any is repeated, and the order of variations and directions is randomized.

## Output

All files are written to `OUTPUT_DIR`:

- `000001.png`, `000002.png`, ...: 8-bit grayscale frames.
- `DATE_TIME_params.log`: the values of all parameters, the random seed used, and the pixels per degree used.
- `DATE_TIME_trials.log`: one line per period, in space-separated columns:
  - `Frame`: frame number at which the period starts (matches the PNG file names).
  - `Time`: time from the first frame, in ms, computed from the frame number and frame rate.
  - `Period`: `PRESTIM`, `TRIAL` or `POSTSTIM`.
  - `TrialNo`: trial counter (trials only).
  - `Stimulus`: stimulus properties as `property=value` pairs (trials only). All stimuli include `stim`, `dir` and `tfreq`; flows add `nDots`, `diam`, `spac`, `dotLvl`, `bgLvl` and `interLvl`; gratings add `width`, `fgLvl`, `bgLvl`, `interLvl` and `phase`; all include the aperture settings `maskD`, `maskX` and `maskY`.

## Improvements on the original FlowStims

Beyond the change to offline rendering, several behaviours differ from the Processing version (2.2):

- Sizes (dot diameters, spacings, bar widths) are not rounded to whole pixels, and dots and gratings are antialiased with the correct drawn area at all sizes.
- Multi-dot flow elements are placed on a lattice aligned with their direction of motion, with equal gaps between elements along and across the element axis. With `EQUAL_AREA = True` the element density is the same as for single-dot flows; with `EQUAL_AREA = False` the gaps are the same as for single-dot flows.
- The flow speed is computed from the spacing between elements along the direction of motion, so that the temporal frequency is exact for multi-dot flows.
- Force and timing parameters are defined in seconds (`MAX_FORCE_RATE`, `SEP_INTERVAL_SEC`) and scale with dot size, so flow dynamics are the same across dot sizes and frame rates.
- All elements are updated simultaneously in each frame.
- The automatic background level (`DOT_BG_VALS = -1`) is rounded rather than truncated, and the script stops with an error if it falls outside [0, 255].

## License

This program is derived from [FlowStims](https://github.com/zuckerlab/FlowStims) and, like the original, is distributed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the full text.
