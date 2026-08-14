# GUI.md — VocalPy USV Workbench (design contract)

This document is the visual + workflow contract for the PySide6 app in `vpgui/`.
Keep it in sync with the code whenever the GUI structure or major workflow changes.

## Purpose

A dark "scientific workstation" desktop app that **detects, classifies, and
segments animal ultrasonic vocalizations** and **exports a per-file CSV**, driving
the vendored [`gumadeiras/vocalpy`](https://github.com/gumadeiras/vocalpy) engine
(under `vocalpy_engine/`, inspired by VocalMat). It preview-renders band-limited
spectrograms, runs the detect → classify → (segment) pipeline per file, overlays
detected calls as class-colored boxes, and lists them in a results table.

## Visual style

Cyan / teal / blue accents on a near-black `#0b1017` background, with subtle
gradients (header bar, primary/preset buttons, progress, slider) and metric-tile /
chip classes. All colors live in `vpgui/theme.py`; `plotting.py` reuses them so the
matplotlib figures blend in. Fonts prefer what is installed: UI `Ubuntu → Noto Sans
→ DejaVu Sans`; mono `Ubuntu Mono → DejaVu Sans Mono`.

| token | value | use |
|---|---|---|
| `BG` / `BG_PANEL` / `BG_ELEV` | `#0b1017` / `#0e1722` / `#13202d` | window / panels / inputs |
| `BG_PLOT` | `#0c141d` | matplotlib axes face |
| `ACCENT` / `ACCENT_TEAL` / `ACCENT_BLUE` | `#19c8d6` / `#2dd4bf` / `#38bdf8` | titles, focus, gradients |
| `OK / WARN / ERROR` | green / amber / red | log + status levels |

Detected-call boxes are colored per syllable class via `engine.CLASS_COLORS`.

## Layout

```
┌ HeaderBar: [logo] "VocalPy USV Workbench" + subtitle ...... [FILE][SAMPLE RATE][DURATION] chips ┐
├───────────────┬────────────────────────────────────────────────────────────────────────────────┤
│ LEFT (scroll) │ RIGHT (vertical splitter)                                                        │
│  ┌Audio files┐│  ┌ Tabs: [ Spectrogram ] [ Gallery ] ──────────────────────────────────────┐    │
│  ┌Detection  ┐│  │  Tiles: [VOCALIZATIONS][CALLS/MIN][MEAN DUR][TOP CLASS][ CURSOR readout ]  │    │
│  │ pipeline: ││  │  Spectrogram: waveform · band-limited spectrogram · class-colored boxes    │    │
│  │ species,  ││  │  Gallery: Build gallery (N) · thumbnails auto-seeked to call-rich windows  │    │
│  │ bin, band,││  └────────────────────────────────────────────────────────────────────────┘    │
│  │ threads,  ││  ┌ Detection ───────────────────────────────────────────────────────────────┐   │
│  │ segmenter ││  │  Scope [Current window|Whole file] [Detect] [Detect all→CSV] [Cancel] [Open]│   │
│  └───────────┘│  │  progress · phase                                                          │   │
│  ┌Preview win┐│  │  ┌ Detections table ─────────────┐  ┌ log (color-coded) ────────────────┐ │   │
│  ┌Display────┐│  │  │ # start end dur freq bw class  │  │ pipeline messages · CSV path       │ │   │
└───────────────┴──┴──┴────────────────────────────────┴──┴────────────────────────────────────┴───┘
Status bar: engine version · transient messages
```

The header carries a drawn "spectrogram-bars" logo and three live chips (FILE /
SAMPLE RATE / DURATION). Above the spectrogram, a metric-tiles bar reports the
detection summary + a **Cursor** tile that tracks the mouse.

## Controls

**Audio files** — `Add files…`, `Add folder…` (recursive toggle), `Remove selected`,
`Clear all`. Rows show paths relative to the common ancestor.

**Detection pipeline** — `Species` (mouse / rat / guineapig; sets the default band),
`Bin size` (parallel-processing chunk, s), `Auto band` + explicit `Band` (kHz)
override, `Threads` (auto = cores/2), `SqueakOut neural segmentation`, `Save
validation overlays`. These map onto the engine `args`.

**Preview window** — `Preview whole file`, `Start` / `Length` (s, capped at 60 s),
scrub slider, `◀ prev` / `next ▶`. Previews read only the visible window and compute
the *same* band-limited spectrogram the detector uses (`spectro.compute_display_spectrogram`).

**Display** (re-render only) — `Colormap`, `Dynamic range (dB)`, `Denoise (subtract
background)`, `Show waveform panel`, `Save current figure…`.

**Detection** — `Scope` ∈ {Current window, Whole file}; `Detect vocalizations`
(current file), `Detect all → CSV` (batch every file, whole-file), `Cancel`, `Open
output`. A results **table** (`#`, start, end, dur ms, freq kHz, bandwidth, class,
2nd) lists detections; clicking a row jumps the preview to that call. A color-coded
log streams the pipeline's own messages and the written CSV path.

**Gallery** — `Build gallery` renders a thumbnail per file, each auto-seeked to its
most vocal window (`spectro.find_active_window`). Click a card to open + jump there.

## Workflow contract

1. Add files or a folder; pick the species (sets the analysis band).
2. Select a file → windowed, band-limited spectrogram preview (coalesced + generation
   token so rapid scrubbing never piles up compute).
3. `Detect vocalizations` — *Current window* writes a temp CSV and overlays results
   fast; *Whole file* runs the full pipeline and writes `{name}_outputs/{name}_stats.csv`.
   Detected calls are boxed by class, listed in the table, summarized in the tiles.
4. Click a table row to inspect that call; hover for exact time/frequency/power.
5. `Detect all → CSV` batches every file (whole-file) with progress + per-file logging.
6. Or run headless: `vocalpy-cli PATH… [-a rat] [--segmenter]`.

## Architecture (one responsibility per module)

| module | responsibility | Qt? |
|---|---|---|
| `theme.py` | palette + global QSS (gradients, tiles, chips) | no |
| `engine.py` | adapter over vendored `vocalpy`: `DetectParams`, `run_detection`(+window), `Detection`, `summarize`, class colors | no |
| `spectro.py` | audio discovery, windowed read, engine-consistent display spectrogram, denoise/limits, `find_active_window` | no |
| `plotting.py` | Figure rendering + class-colored detection overlays (thread-safe) | no |
| `workers.py` | `PreviewWorker`, `DetectWorker`, `BatchDetectWorker`, `GalleryWorker` | yes |
| `widgets.py` | `MetricTile`/`MetricsBar`, `SpectrogramView` (hover crosshair) | yes |
| `results.py` | `DetectionsTable` (class-colored, clickable rows) | yes |
| `gallery.py` | `FlowLayout`, `ThumbnailCard`, `GalleryView` | yes |
| `cli.py` | headless detect files/folders → per-file CSV | no |
| `app.py` | `MainWindow` (header, pipeline controls, detection) + `main()` | yes |

## Engine + models

The engine is vendored under `vocalpy_engine/` and installed editable as `vocalpy`.
Detection needs numpy/scipy/opencv/scikit-image; classification (MobileNetV2 noise +
11-class syllable models) and SqueakOut segmentation need torch/torchvision and the
Git-LFS checkpoints in `vocalpy_engine/vocalpy/nn/pretrained/` (`noise_model.pth.tar`,
`class_model.pth.tar`, `segment_model.ckpt`). The `UVS` conda env (`environment.yaml`)
provides the full stack (torch CPU build).
