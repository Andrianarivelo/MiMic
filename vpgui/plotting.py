"""Matplotlib rendering for spectrograms, styled to match the dark UI.

Everything here works on an explicitly supplied :class:`matplotlib.figure.Figure`
and never touches :mod:`matplotlib.pyplot`.  That keeps it thread-safe: the batch
worker renders PNGs on a background ``Agg`` figure while the GUI thread renders the
live preview onto the embedded canvas figure, both through :func:`draw_spectrogram`.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

import vocalpy as voc

from . import theme
from .spectro import SpectSettings, first_channel

# Cap the number of time columns actually drawn. A preview window can easily be
# tens of thousands of columns wide; the screen has ~2k pixels, so striding the
# data keeps rendering instant with no visible loss. Saved .npz files are full
# resolution -- this only affects what is drawn on screen.
MAX_DISPLAY_COLS = 4000


def _display_data(data: np.ndarray, method: str) -> np.ndarray:
    """Map a spectrogram array to a dB scale for display.

    ``librosa-db`` and ``soundsig-spectro`` are already dB-scaled, but
    ``sat-multitaper`` returns *linear power*; shown on the same dB color scale it
    would collapse to a single flat color. We convert it to dB for display only
    (the saved ``.npz`` keeps vocalpy's native linear output for analysis).
    """
    if method == "sat-multitaper":
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(np.maximum(data, 1e-20))
    return data


def _style_axes(ax) -> None:
    """Apply the dark workstation styling to a single Axes."""
    ax.set_facecolor(theme.BG_PLOT)
    for spine in ax.spines.values():
        spine.set_color(theme.BORDER)
    ax.tick_params(colors=theme.TEXT_DIM, labelsize=8, length=3)
    ax.xaxis.label.set_color(theme.TEXT_DIM)
    ax.yaxis.label.set_color(theme.TEXT_DIM)
    ax.title.set_color(theme.ACCENT)


def draw_spectrogram(
    fig: Figure,
    sound: voc.Sound | None,
    spect: voc.Spectrogram,
    settings: SpectSettings,
    title: str | None = None,
    time_offset: float = 0.0,
) -> None:
    """Clear ``fig`` and draw ``spect`` (and optionally the waveform) onto it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure. It is fully cleared first, so the same figure can be
        re-rendered repeatedly (live preview) without leaking stale colorbars.
    sound : vocalpy.Sound or None
        Used only to draw the waveform panel. If None, no waveform is shown.
    spect : vocalpy.Spectrogram
        Spectrogram to display.
    settings : SpectSettings
        Display options (colormap, dynamic range, frequency limits, waveform).
    title : str, optional
        Axes title (typically the file name).
    """
    fig.clear()
    fig.patch.set_facecolor(theme.BG)

    spect = first_channel(spect)
    data = np.squeeze(spect.data, axis=0)  # (freq, time)
    t, f = spect.times, spect.frequencies

    # Decimate columns for display only (keeps big windows snappy).
    if data.shape[1] > MAX_DISPLAY_COLS:
        stride = int(np.ceil(data.shape[1] / MAX_DISPLAY_COLS))
        data = data[:, ::stride]
        t = t[::stride]

    # Show absolute time within the source file (windowed previews start at 0).
    t = t + float(time_offset)
    data = _display_data(data, settings.method)

    # A dedicated colorbar column (not `ax=`) keeps the waveform and spectrogram
    # x-axes the same width, so their shared time axis stays aligned.
    show_wave = settings.show_waveform and sound is not None
    if show_wave:
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1, 3.6], width_ratios=[1, 0.02],
            hspace=0.08, wspace=0.015,
            left=0.085, right=0.92, top=0.93, bottom=0.10,
        )
        ax_wave = fig.add_subplot(gs[0, 0])
        ax_spec = fig.add_subplot(gs[1, 0], sharex=ax_wave)
        cax = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(
            1, 2, width_ratios=[1, 0.02], wspace=0.015,
            left=0.085, right=0.92, top=0.93, bottom=0.12,
        )
        ax_wave = None
        ax_spec = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

    # ---- waveform ----
    if show_wave:
        wav = sound.data
        if settings.to_mono and wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(axis=0)   # match the mono mixdown used for the spectrogram
        else:
            wav = wav[0] if wav.ndim > 1 else wav
        peak = float(np.max(np.abs(wav))) if wav.size else 1.0
        peak = peak if peak > 0 else 1.0
        t0 = float(t.min())
        dur = wav.shape[-1] / sound.samplerate
        if wav.shape[-1] > 2 * MAX_DISPLAY_COLS:
            # Min/max envelope so amplitude peaks survive decimation.
            nbins = MAX_DISPLAY_COLS
            usable = (wav.shape[-1] // nbins) * nbins
            binned = wav[:usable].reshape(nbins, -1)
            wmin, wmax = binned.min(axis=1), binned.max(axis=1)
            tb = t0 + np.linspace(0, dur, nbins)
            ax_wave.fill_between(tb, wmin, wmax, color=theme.ACCENT, linewidth=0)
        else:
            tw = t0 + np.arange(wav.shape[-1]) / sound.samplerate
            ax_wave.plot(tw, wav, color=theme.ACCENT, linewidth=0.6)
        ax_wave.set_ylabel("amp.", fontsize=8)
        ax_wave.margins(x=0)
        ax_wave.set_xlim(t0, t0 + dur)
        ax_wave.tick_params(labelbottom=False)
        ax_wave.set_ylim(-peak * 1.1, peak * 1.1)
        _style_axes(ax_wave)
        if title:
            ax_wave.set_title(title, fontsize=10, pad=6)
    elif title:
        ax_spec.set_title(title, fontsize=10, pad=6)

    # ---- spectrogram ----
    # soundsig can emit -inf in silent bins; compute limits over finite values.
    finite = data[np.isfinite(data)]
    vmax = float(np.nanpercentile(finite, 99.8)) if finite.size else 0.0
    vmin = vmax - float(settings.dynamic_range_db)
    extent = [float(t.min()), float(t.max()), float(f.min()), float(f.max())]
    im = ax_spec.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=settings.colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (kHz)")
    ax_spec.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / 1000:g}"))

    if settings.limit_freq:
        lo = min(settings.flim_min_khz, settings.flim_max_khz) * 1000.0
        hi = max(settings.flim_min_khz, settings.flim_max_khz) * 1000.0
        if hi <= lo:
            hi = lo + 1.0  # guard against an inverted / zero-height axis
        ax_spec.set_ylim(lo, hi)
    _style_axes(ax_spec)

    # ---- colorbar (own axes -> panels stay aligned) ----
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Power (dB)", color=theme.TEXT_DIM, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=theme.TEXT_DIM, labelsize=7)
    cbar.outline.set_edgecolor(theme.BORDER)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_color(theme.TEXT_DIM)


def draw_thumbnail(
    fig: Figure,
    spect: voc.Spectrogram,
    settings: SpectSettings,
) -> None:
    """Draw a compact, full-bleed spectrogram (no axes/colorbar) for the gallery."""
    fig.clear()
    fig.patch.set_facecolor(theme.BG_PLOT)
    spect = first_channel(spect)
    data = np.squeeze(spect.data, axis=0)
    t, f = spect.times, spect.frequencies
    if data.shape[1] > 1400:
        stride = int(np.ceil(data.shape[1] / 1400))
        data = data[:, ::stride]
        t = t[::stride]
    data = _display_data(data, settings.method)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(theme.BG_PLOT)
    finite = data[np.isfinite(data)]
    vmax = float(np.nanpercentile(finite, 99.8)) if finite.size else 0.0
    vmin = vmax - float(settings.dynamic_range_db)
    ax.imshow(
        data, origin="lower", aspect="auto",
        extent=[float(t.min()), float(t.max()), float(f.min()), float(f.max())],
        cmap=settings.colormap, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    if settings.limit_freq:
        lo = min(settings.flim_min_khz, settings.flim_max_khz) * 1000.0
        hi = max(settings.flim_min_khz, settings.flim_max_khz) * 1000.0
        ax.set_ylim(lo, hi if hi > lo else lo + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def render_thumbnail_png(
    spect: voc.Spectrogram,
    settings: SpectSettings,
    size: tuple[float, float] = (3.1, 1.75),
    dpi: int = 82,
) -> bytes:
    """Render a gallery thumbnail to PNG bytes (thread-safe Agg, no pyplot)."""
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    try:
        draw_thumbnail(fig, spect, settings)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        return buf.getvalue()
    finally:
        fig.clear()


def draw_placeholder(fig: Figure, message: str) -> None:
    """Draw a centered hint message on an empty dark figure."""
    fig.clear()
    fig.patch.set_facecolor(theme.BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme.BG)
    ax.axis("off")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", color=theme.TEXT_FAINT,
        fontsize=12, transform=ax.transAxes, wrap=True,
    )
