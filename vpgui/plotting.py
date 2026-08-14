"""Matplotlib rendering for the USV workbench (thread-safe, no pyplot).

Draws a dark-themed spectrogram from raw numpy arrays (as produced by
:func:`vpgui.spectro.compute_display_spectrogram`), an optional waveform panel,
a dB colorbar, and detection-box overlays colored by predicted syllable class.
Everything works on an explicitly supplied Figure so batch/gallery rendering is
safe off the UI thread.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from . import spectro, theme
from .engine import class_color
from .spectro import DisplaySettings

for _f in ("Ubuntu", "Noto Sans", "DejaVu Sans"):
    if _f in {f.name for f in mpl.font_manager.fontManager.ttflist}:
        mpl.rcParams["font.family"] = _f
        break

MAX_DISPLAY_COLS = 4000


def _style_axes(ax) -> None:
    ax.set_facecolor(theme.BG_PLOT)
    for spine in ax.spines.values():
        spine.set_color(theme.BORDER_LIGHT)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=theme.TEXT_DIM, labelsize=8.5, length=3, width=0.8)
    ax.xaxis.label.set_color(theme.TEXT_DIM)
    ax.yaxis.label.set_color(theme.TEXT_DIM)
    ax.title.set_color(theme.ACCENT)


def _decimate(data, t, max_cols):
    if data.shape[1] > max_cols:
        stride = int(np.ceil(data.shape[1] / max_cols))
        return data[:, ::stride], t[::stride]
    return data, t


def draw_spectrogram(fig, f, t_abs, S_db, samples, sr, settings: DisplaySettings,
                     detections=None, title=None):
    """Draw the spectrogram (+ waveform + detection overlays). Returns the image."""
    fig.clear()
    fig.patch.set_facecolor(theme.BG)

    data = np.asarray(S_db, dtype=float)
    t = np.asarray(t_abs, dtype=float)
    f = np.asarray(f, dtype=float)
    data, t = _decimate(data, t, MAX_DISPLAY_COLS)
    data = spectro.to_display_db(data, settings.denoise)

    show_wave = settings.show_waveform and samples is not None and len(samples)
    if show_wave:
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 3.8], width_ratios=[1, 0.022],
                              hspace=0.07, wspace=0.014,
                              left=0.075, right=0.918, top=0.95, bottom=0.095)
        ax_wave = fig.add_subplot(gs[0, 0])
        ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_wave)
        cax = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.022], wspace=0.014,
                              left=0.075, right=0.918, top=0.95, bottom=0.11)
        ax_wave = None
        ax_spec = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

    t0 = float(t.min()) if t.size else 0.0
    t1 = float(t.max()) if t.size else 1.0

    # ---- waveform ----
    if show_wave:
        wav = np.asarray(samples, dtype=float)
        peak = float(np.max(np.abs(wav))) if wav.size else 1.0
        peak = peak if peak > 0 else 1.0
        dur = wav.shape[-1] / sr
        ax_wave.axhline(0.0, color=theme.BORDER, linewidth=0.6, zorder=0)
        if wav.shape[-1] > 2 * MAX_DISPLAY_COLS:
            nbins = MAX_DISPLAY_COLS
            usable = (wav.shape[-1] // nbins) * nbins
            binned = wav[:usable].reshape(nbins, -1)
            tb = t0 + np.linspace(0, dur, nbins)
            ax_wave.fill_between(tb, binned.min(1), binned.max(1),
                                 color=theme.ACCENT, alpha=0.55, linewidth=0, zorder=2)
        else:
            tw = t0 + np.arange(wav.shape[-1]) / sr
            ax_wave.plot(tw, wav, color=theme.ACCENT, linewidth=0.6, zorder=2)
        ax_wave.set_ylabel("amp.", fontsize=8.5)
        ax_wave.margins(x=0)
        ax_wave.set_xlim(t0, t0 + dur)
        ax_wave.set_ylim(-peak * 1.12, peak * 1.12)
        ax_wave.tick_params(labelbottom=False)
        _style_axes(ax_wave)
        if title:
            ax_wave.set_title(title, fontsize=11, pad=7, fontweight="bold", loc="left")
    elif title:
        ax_spec.set_title(title, fontsize=11, pad=7, fontweight="bold", loc="left")

    # ---- spectrogram ----
    vmin, vmax = spectro.display_limits(data, settings.dynamic_range_db, settings.denoise)
    f0 = float(f.min()) if f.size else 0.0
    f1 = float(f.max()) if f.size else 1.0
    im = ax_spec.imshow(data, origin="lower", aspect="auto",
                        extent=[t0, t1, f0, f1], cmap=settings.colormap,
                        vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (kHz)")
    ax_spec.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1000:g}"))
    ax_spec.grid(axis="y", color=theme.BORDER_LIGHT, alpha=0.15, linewidth=0.6)
    _style_axes(ax_spec)

    # ---- detection overlays ----
    # USV calls are often only a few ms / kHz wide, i.e. sub-pixel at file scale.
    # Enforce a small minimum *drawn* size (display only) and a translucent fill
    # so every detection is visible; the table/CSV carry the exact bounds.
    if detections:
        y0, y1 = ax_spec.get_ylim()
        min_w = (t1 - t0) * 0.004
        min_h = (y1 - y0) * 0.012
        for d in detections:
            col = class_color(d.top1)
            w = max(d.end - d.start, min_w)
            h = max(d.max_freq - d.min_freq, min_h)
            x = d.start - max(0.0, (w - (d.end - d.start)) / 2.0)
            y = d.min_freq - max(0.0, (h - (d.max_freq - d.min_freq)) / 2.0)
            ax_spec.add_patch(Rectangle((x, y), w, h, facecolor=col, alpha=0.16,
                                        edgecolor="none", zorder=4))
            ax_spec.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=col,
                                        linewidth=1.5, joinstyle="round", zorder=5))
            if (d.end - d.start) > (t1 - t0) * 0.02:
                ax_spec.text(x, y + h, f" {d.top1 or ''}", color=col, fontsize=7.5,
                             fontweight="bold", va="bottom", ha="left", zorder=6)

    cbar = fig.colorbar(im, cax=cax)
    label = "Power over background (dB)" if settings.denoise else "Power (dB)"
    cbar.set_label(label, color=theme.TEXT_DIM, fontsize=8.5)
    cbar.ax.yaxis.set_tick_params(color=theme.TEXT_DIM, labelsize=7.5)
    cbar.outline.set_edgecolor(theme.BORDER_LIGHT)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_color(theme.TEXT_DIM)
    return im


def draw_thumbnail(fig, f, t, S_db, settings: DisplaySettings, detections=None):
    fig.clear()
    fig.patch.set_facecolor(theme.BG_PLOT)
    data = np.asarray(S_db, dtype=float)
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    data, t = _decimate(data, t, 1400)
    data = spectro.to_display_db(data, settings.denoise)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(theme.BG_PLOT)
    vmin, vmax = spectro.display_limits(data, settings.dynamic_range_db, settings.denoise)
    t0, t1 = float(t.min()), float(t.max())
    f0, f1 = float(f.min()), float(f.max())
    ax.imshow(data, origin="lower", aspect="auto", extent=[t0, t1, f0, f1],
              cmap=settings.colormap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if detections:
        for d in detections:
            ax.add_patch(Rectangle((d.start, d.min_freq), max(d.end - d.start, 1e-4),
                                   max(d.max_freq - d.min_freq, 1.0), fill=False,
                                   edgecolor=class_color(d.top1), linewidth=0.9))
    ax.set_xticks([])
    ax.set_yticks([])


def render_thumbnail_png(f, t, S_db, settings: DisplaySettings, detections=None,
                         size=(3.3, 1.9), dpi=104) -> bytes:
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    try:
        draw_thumbnail(fig, f, t, S_db, settings, detections)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        fig.clear()


def draw_placeholder(fig, message: str) -> None:
    fig.clear()
    fig.patch.set_facecolor(theme.BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.BG)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=theme.TEXT_FAINT,
            fontsize=12.5, transform=ax.transAxes, wrap=True)
