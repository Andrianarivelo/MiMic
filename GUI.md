# GUI.md — VocalPy Spectrogram Workbench (design contract)

This document is the visual + workflow contract for the PySide6 app in `vpgui/`.
Keep it in sync with the code whenever the GUI structure or major workflow changes.

## Purpose

A dark "scientific workstation" desktop app for **batch spectrogram processing**
and **interactive spectrogram visualization**, built on the
[VocalPy](https://github.com/vocalpy/vocalpy) library (v0.11.0).

## Visual style

Matches the rest of the lab's tools: cyan / teal / blue accents on a near-black
`#0b1017` background. All colors live in `vpgui/theme.py`; the matplotlib figures
reuse the same constants so they blend into the surrounding widgets.

| token | value | use |
|---|---|---|
| `BG` | `#0b1017` | window background |
| `BG_PANEL` | `#0e1722` | group boxes |
| `BG_ELEV` | `#13202d` | inputs, list rows |
| `BG_PLOT` | `#0c141d` | matplotlib axes face |
| `ACCENT` | `#19c8d6` | primary cyan (titles, focus, progress) |
| `ACCENT_TEAL` | `#2dd4bf` | hover |
| `OK / WARN / ERROR` | green / amber / red | log + status levels |

## Layout

```
┌ Header: "VocalPy Spectrogram Workbench" + subtitle ........ info label (file·sr·ch·window·shape) ┐
├───────────────┬────────────────────────────────────────────────────────────────────────────────┤
│ LEFT (scroll) │ RIGHT (vertical splitter)                                                        │
│  ┌Audio files┐│  ┌ Tabs: [ Spectrogram ] [ Gallery ] ────────────────────────────────────────┐  │
│  │ list (rel ││  │  Spectrogram: nav toolbar · waveform · spectrogram (imshow) · dB colorbar  │  │
│  │  paths)   ││  │  Gallery: Build gallery (N) · reflowing grid of clickable thumbnail cards  │  │
│  │ add/folder││  └───────────────────────────────────────────────────────────────────────────┘  │
│  │ example   ││  ┌ Batch processing ─────────────────────────────────────────────────────────┐  │
│  │ remove/clr││  │  output folder + Browse                                                     │  │
│  └───────────┘│  │  [x].npz  [x].png  dpi   Max s/file                                         │  │
│  ┌Preview win┐│  │  [Process all (N)] [Cancel]  ▓▓▓ progress                                   │  │
│  │ meta · sr ││  │  log (monospace, color-coded)                                               │  │
│  │ whole-file││  └───────────────────────────────────────────────────────────────────────────┘  │
│  │ start/len ││                                                                                   │
│  │ scrub ◀▶  ││                                                                                   │
│  └───────────┘│                                                                                   │
│  ┌Method─────┐│                                                                                   │
│  ┌Display────┐│                                                                                   │
└───────────────┴────────────────────────────────────────────────────────────────────────────────┘
Status bar: VocalPy version · transient messages
```

## Controls

**Audio files** — `Add files…`, `Add folder…` (recursively scans sub-folders when
*Scan sub-folders recursively* is on), `Load built-in example ▾`, `Remove selected`,
`Clear all`. Rows show each file **relative to the common ancestor** so nested files
from a recursive scan are disambiguated (e.g. `31096/6/clip.wav`). Duplicates ignored.

**Preview window** — for long / high-sample-rate recordings, only a window is read &
shown: file metadata (`duration · kHz · ch · Nyquist`), `Preview whole file` toggle,
`Start` / `Length` (s), a scrub slider, and `◀ prev` / `next ▶`. A memory guard refuses
windows whose spectrogram would exceed ~1 GB and explains how to shrink it.

**Spectrogram method** (compute params — changing them recomputes the preview):
`method` ∈ {`librosa-db`, `sat-multitaper`, `soundsig-spectro`}, `n_fft`, `hop_length`,
soundsig-only rows shown only for that method, `to_mono`, `Update preview`.

**Display** (re-render only — never recompute): `colormap`, `dynamic range (dB)`,
`show waveform`, `limit frequency axis` + range, `Save current figure…`.

**Gallery tab** — `Build gallery (N)` renders a small spectrogram thumbnail (first
*thumb window* seconds) for every file in a `GalleryWorker`, progressively filling a
reflowing grid of cards (thumbnail + relative path + `duration · kHz · ch`). Clicking a
card selects that file and opens the interactive **Spectrogram** tab.

**Batch processing**: output folder, `.npz` and/or `.png` outputs, PNG dpi,
`Max s/file` (0 = whole file), `Process all files (N)`, `Cancel`, progress, color log.

## Workflow contract

1. Add files, **add a folder (recursively scanning sub-folders for audio)**, or load an example.
2. Selecting a file reads its metadata cheaply, configures the preview window, and computes
   a **windowed** spectrogram on a `PreviewWorker`. Preview requests are **coalesced** (only
   the latest runs) and a **generation token** discards stale results.
3. Scrub / `prev` / `next` to move the window through a long recording (axes show absolute
   file time). Tune compute params → recompute; tune display params → instant re-render.
4. `Build gallery` for a visual overview of every file; click any card to inspect it.
5. Set an output folder, pick formats and an optional per-file cap, `Process all files` →
   `BatchWorker` writes `<stem>.npz` and/or `<stem>.png` (disambiguating colliding stems),
   reporting progress and honoring `Cancel`. Controls lock during the run; per-file errors
   are isolated and logged.

## Architecture (one responsibility per module)

| module | responsibility | Qt? | pyplot? |
|---|---|---|---|
| `theme.py` | palette + global QSS | no | no |
| `spectro.py` | vocalpy logic: discovery, info, windowed load, dispatch | no | no |
| `plotting.py` | Figure-based rendering + thumbnails (thread-safe) | no | no |
| `workers.py` | `PreviewWorker`, `BatchWorker`, `GalleryWorker` (QThread) | yes | no |
| `widgets.py` | `MplCanvas`, toolbar, `SpectrogramView` | yes | no |
| `gallery.py` | `FlowLayout`, `ThumbnailCard`, `GalleryView` | yes | no |
| `app.py` | `MainWindow` + `main()` | yes | no |

## Handling long / high-rate recordings

The app is designed for ultrasonic vocalization (USV) data: e.g. 384 kHz mono, 15 min,
~700 MB files. It never loads a whole such file for preview — `spectro.sound_info` reads
metadata only, `read_sound_segment` reads just the visible window via soundfile's
`start`/`frames`, the display decimates to ≤4000 columns (min/max envelope for the
waveform), and a memory guard blocks any preview whose spectrogram would exceed ~1 GB.
Batch can cap each file with *Max s/file*. Empty / truncated files are reported, not fatal.

`pyplot` is never used so PNG rendering is safe on the batch thread (`Agg` Figure)
while the live preview renders onto the embedded canvas Figure — both through
`plotting.draw_spectrogram`.

## VocalPy backend notes

`soundsig-spectro` is dispatched directly (pre-scaled to int16, `scale=False`) to
work around two real bugs in vocalpy 0.11.0: the `voc.spectrogram` convenience
wrapper mis-forwards `n_fft`/`hop_length`, and `soundsig_spectro(scale=True)`
references a non-existent `Sound.path`. See the docstring in `spectro.py`.
