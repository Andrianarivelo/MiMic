"""Independent white-background graphics and reports for injected-USV recovery.

Read completed numerical CSVs without importing the experiment or detector.
Run this file directly for rendering, or call ``make_outputs(output_folder)``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

# Rendering and descriptive-bootstrap parameters. No detector settings live here.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'lgdel_usv_analysis' / 'detector_recovery'
COLORS = {'WT': '#327A9E', 'HET': '#E68A32'}
GROUPS = ['WT', 'HET']
MODES = ['amplitude', 'snr']
XLABELS = {'amplitude': 'Injected active RMS (dBFS)', 'snr': 'Injected signal / background (dB)'}
DPI = 320
FORMATS = ('png', 'pdf', 'svg')
FONT_SIZE = 9
SEED = 20261008
N_BOOT = 4000
SAMPLE_RATE = 384000
SPECTROGRAM_WINDOW = 512
SPECTROGRAM_HOP = 128
AUDIT_BAND = (45000., 125000.)


def configure() -> None:
    """Use editable vector text, restrained colors, and a white canvas."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': FONT_SIZE,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .75,
        'axes.edgecolor': '#667780', 'axes.facecolor': 'white', 'figure.facecolor': 'white',
        'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, out: Path, name: str) -> None:
    """Save the same reviewed layout in all three scientific artifact formats."""
    for fmt in FORMATS:
        fig.savefig(out / f'{name}.{fmt}', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def number(value, digits=3) -> str:
    """Keep missing values explicit and avoid rounding small p-values to zero."""
    return f'{value:.{digits}g}' if np.isfinite(value) else 'NA'


def boolean(series: pd.Series) -> np.ndarray:
    """Decode CSV booleans strictly rather than treating the string False as True."""
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    decoded = series.astype(str).str.lower().map(mapping)
    if decoded.isna().any():
        raise ValueError(f'Invalid boolean values in {series.name}')
    return decoded.to_numpy(bool)


def validate(out: Path, tables: dict) -> None:
    """Check injection geometry, support, signal targets, and stage accounting.

    Amplitude/SNR target checks validate the normalized active-support design.
    They do not claim a microphone SPL calibration or measure biological source
    loudness. Reduced eligible counts remain visible, without imputing recovery.
    """
    manifest, bg, trials, sessions = (tables[k] for k in ['manifest', 'background', 'trials', 'sessions'])
    rows = []
    def check(name, passed, value):
        rows.append({'check': name, 'passed': bool(passed), 'value': str(value)})
    check('unique_recordings', len(bg) == 24 and bg.animal_id.nunique() == 24, len(bg))
    balance = bg.groupby('genotype').size().to_dict()
    check('twelve_per_genotype', balance == {'HET': 12, 'WT': 12}, balance)
    context_counts = bg.groupby(['genotype', 'proposed_start_s']).size()
    check('planned_three_contexts_four_per_genotype', len(context_counts) == 6 and (context_counts == 4).all(), context_counts.to_dict())
    check('context_length', np.allclose(bg.end_s - bg.start_s, 60.15), '60 s plus 0.15 s halo')
    check('injection_ids_unique', not manifest.duplicated(['animal_id', 'mode', 'variant', 'slot']).any(), len(manifest))
    check('positive_truth_support', (manifest.truth_end > manifest.truth_start).all(), len(manifest))
    check('truth_within_context', ((manifest.truth_start >= manifest.context_start_s) &
          (manifest.truth_end <= manifest.context_start_s + 60.15)).all(), len(manifest))
    check('no_mixture_clipping', manifest.mixture_peak.between(0, 1, inclusive='left').all(), manifest.mixture_peak.max())
    check('nonzero_active_injection_background', (manifest.noise_rms > 1e-7).all(), manifest.noise_rms.min())
    audit = tables['audio_audit']
    bins = tables['audio_bins']
    check('audio_audit_has_all_recordings', len(audit) == 24 and audit.animal_id.nunique() == 24, len(audit))
    check('audio_read_missing_accounting', (audit.read_frames+audit.missing_frames == audit.expected_frames).all(), len(audit))
    zero_seconds = bins.assign(zero=boolean(bins.all_zero_complete_second)).groupby('animal_id').zero.sum()
    check('audio_zero_second_accounting', np.array_equal(zero_seconds.to_numpy(),
          audit.set_index('animal_id').all_zero_seconds.reindex(zero_seconds.index).to_numpy()), int(zero_seconds.sum()))
    check('snr_active_support_identity', np.allclose(manifest.received_snr_db,
          manifest.received_amplitude_dbfs - 20*np.log10(manifest.noise_rms), atol=1e-5), 'same-slot band RMS')
    for mode, column in [('snr', 'received_snr_db'), ('amplitude', 'received_amplitude_dbfs')]:
        sub = manifest[manifest['mode'] == mode]
        check(f'{mode}_targets', np.allclose(sub[column], sub.target_level, atol=1e-5), len(sub))
    pooled = manifest[manifest.variant == 'pooled']
    counts = pooled.groupby(['animal_id', 'mode', 'target_level']).size()
    check('four_templates_per_pooled_level', len(counts) == 24*2*7 and (counts == 4).all(), counts.to_dict())
    iso_counts = manifest[manifest.variant == 'isolated'].groupby(['animal_id', 'mode']).size()
    check('four_templates_per_isolated_transition', len(iso_counts) == 48 and (iso_counts == 4).all(), len(iso_counts))
    check('trial_rows_conserved', len(trials) == len(manifest), f'{len(trials)}/{len(manifest)}')
    check('session_denominators', ((sessions.n_eligible >= 0) & (sessions.n_eligible <= sessions.n_injected)).all(), len(sessions))
    keys = ['animal_id', 'mode', 'variant', 'target_level']
    actual = trials.assign(eligible=~boolean(trials.sham_retained)).groupby(keys).eligible.sum()
    recorded = sessions.set_index(keys).n_eligible.reindex(actual.index)
    check('eligible_counts_match_sham_flags', np.array_equal(actual.to_numpy(), recorded.to_numpy()), len(actual))
    # The final retained intervals are a subset of identifier candidates. The
    # identifier coverage must use detected_candidate on the common denominator.
    candidate, retained = boolean(trials.detected_candidate), boolean(trials.detected_retained)
    check('retained_detection_requires_candidate', (~retained | candidate).all(), len(trials))
    defined = sessions[['p_identifier', 'probability']].dropna()
    check('retained_probability_not_above_identifier', (defined.probability <= defined.p_identifier + 1e-12).all(), len(defined))
    for name in ['curves', 'sessions']:
        table = tables[name]
        check(f'{name}_probability_bounds', table.probability.dropna().between(0, 1).all(), len(table))
    curves = tables['curves']
    check('curve_ci_bounds', curves.ci_low.between(0, 1).all() and curves.ci_high.between(0, 1).all()
          and (curves.ci_low <= curves.ci_high).all(), len(curves))
    check('primary_family_two_modes', len(tables['auc']) == 2 and set(tables['auc']['mode']) == set(MODES), len(tables['auc']))
    check('secondary_family_fourteen_levels', len(tables['levels']) == 14, len(tables['levels']))
    archive = np.load(out / 'template_waveforms.npz')
    for r in tables['templates'].itertuples():
        raw, clean = archive[r.template_id+'_raw'], archive[r.template_id+'_clean']
        check('template_finite_'+r.template_id, np.isfinite(raw).all() and np.isfinite(clean).all()
              and len(raw) == len(clean) and np.any(clean != 0), len(clean))
        # Independently reproduce the declared RMS support from archived clean
        # samples. This validates normalization geometry without running a model.
        active_lo = int(.003*SAMPLE_RATE)
        active_hi = min(len(clean), int((.003+r.source_end_s-r.source_start_s)*SAMPLE_RATE))
        filtered = signal.sosfiltfilt(signal.butter(4, AUDIT_BAND, btype='bandpass',
                                                   fs=SAMPLE_RATE, output='sos'), clean)
        measured = 20*np.log10(np.sqrt(np.mean(filtered[active_lo:active_hi]**2)))
        check('active_rms_support_'+r.template_id,
              active_hi > active_lo and np.isclose(measured, r.source_band_rms_dbfs, atol=1e-5), measured)
    checks = pd.DataFrame(rows)
    checks.to_csv(out / 'recovery_validation.csv', index=False)
    if not checks.passed.all():
        raise AssertionError(checks.loc[~checks.passed].to_string(index=False))


def draw_curves(out: Path, curves: pd.DataFrame, auc: pd.DataFrame) -> None:
    """Plot the primary retained-call recovery curves with pointwise intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), layout='constrained', sharey=True)
    for ax, mode in zip(axes, MODES):
        for group in GROUPS:
            t = curves[(curves['mode'] == mode) & (curves.genotype == group)].sort_values('target_level')
            n = ','.join(map(str, sorted(t.n_recordings.unique())))
            ax.plot(t.target_level, 100*t.probability, color=COLORS[group], lw=1.45,
                    marker='o', ms=3.4, label=f'{group if group == "WT" else "Het"} (n={n})')
            ax.fill_between(t.target_level.to_numpy(), 100*t.ci_low.to_numpy(),
                            100*t.ci_high.to_numpy(), color=COLORS[group], alpha=.13, lw=0)
        r = auc[auc['mode'] == mode].iloc[0]
        ax.text(.04, .94, 'p = '+number(r.p_holm), transform=ax.transAxes,
                va='top', color='#667780', fontsize=8)
        ax.set_xlabel(XLABELS[mode]); ax.set_ylim(-2, 103)
        ax.set_title('Matched received amplitude' if mode == 'amplitude' else 'Matched received SNR', fontsize=10, pad=11)
        ax.yaxis.grid(True, color='#EDF0F2', lw=.65); ax.set_axisbelow(True)
    axes[0].set_ylabel('Injected calls retained (%)'); axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Recovery of the same whistles in WT and Het recording backgrounds', fontsize=12)
    sensitivity_run = ('exploratory_context_sensitivity' in auc and
                       boolean(auc.exploratory_context_sensitivity).all())
    caption = 'Four fixed whistles; pooled challenge; 95% recording-bootstrap intervals; Holm-adjusted AUC tests'
    if sensitivity_run:
        caption += '\nIntact-context sensitivity: Het 29539 uses 540 s; original results preserved separately'
    fig.supxlabel(caption, fontsize=8, color='#667780')
    save(fig, out, 'detector_recovery_curves')


def identifier_curves(out: Path, sessions: pd.DataFrame) -> None:
    """Export identifier coverage and the same recording-level uncertainty unit."""
    rng = np.random.default_rng(SEED)
    rows = []
    for key, table in sessions[sessions.variant == 'pooled'].groupby(['mode', 'genotype', 'target_level']):
        mode, group, level = key
        x = table.p_identifier.dropna().to_numpy(float)
        if not len(x): continue
        boots = rng.choice(x, (N_BOOT, len(x))).mean(axis=1)
        lo, hi = np.quantile(boots, [.025, .975])
        rows.append(dict(mode=mode, genotype=group, target_level=level,
                         probability=x.mean(), ci_low=lo, ci_high=hi, n_recordings=len(x)))
    curves = pd.DataFrame(rows); curves.to_csv(out/'identifier_curve_statistics.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), layout='constrained', sharey=True)
    for ax, mode in zip(axes, MODES):
        for group in GROUPS:
            t = curves[(curves['mode'] == mode) & (curves.genotype == group)].sort_values('target_level')
            ax.plot(t.target_level, 100*t.probability, lw=1.3, ms=3, marker='o', color=COLORS[group], label=group)
            ax.fill_between(t.target_level.to_numpy(), 100*t.ci_low.to_numpy(), 100*t.ci_high.to_numpy(), color=COLORS[group], alpha=.13, lw=0)
        ax.set_xlabel(XLABELS[mode]); ax.set_ylim(-2, 103)
        ax.yaxis.grid(True, color='#EDF0F2', lw=.65)
    axes[0].set_ylabel('Injected intervals identified (%)'); axes[1].legend(fontsize=8)
    fig.suptitle('Identifier coverage before the noise classifier', fontsize=12)
    save(fig, out, 'detector_identifier_curves')


def diagnostic_figures(out: Path, tables: dict) -> None:
    """Show paired challenge sensitivity and measured recording background levels."""
    sensitivity = tables['isolated']
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.8), layout='constrained', sharex=True, sharey=True)
    for ax, mode in zip(axes, MODES):
        t = sensitivity[sensitivity['mode'] == mode]
        for group in GROUPS:
            sub = t[t.genotype == group]
            ax.scatter(100*sub.pooled_probability, 100*sub.isolated_probability, color=COLORS[group],
                       s=27, edgecolors='white', lw=.6, label=group)
        ax.plot([0, 100], [0, 100], color='#AEB9BF', ls=':', lw=.8)
        level = t.level.iloc[0]
        ax.set_title(f'{mode.capitalize()}: transition level {level:g}', fontsize=10)
        ax.set(xlim=(-3, 103), ylim=(-3, 103), xlabel='Pooled challenge recovery (%)')
    axes[0].set_ylabel('Isolated transition recovery (%)'); axes[1].legend(fontsize=8)
    fig.suptitle('Paired sensitivity to other injected levels', fontsize=12)
    save(fig, out, 'detector_pooled_isolated_sensitivity')
    manifest = tables['manifest'].drop_duplicates(['animal_id', 'slot']).copy()
    manifest['slot_background_dbfs'] = 20*np.log10(manifest.noise_rms)
    quiet = manifest.groupby(['animal_id', 'genotype']).slot_background_dbfs.median().reset_index()
    background = tables['background']
    audit = tables['audio_audit'].copy()
    # Isolated quantized zero samples are ordinary PCM values. Count only
    # complete all-zero seconds as unavailable in this coarse audit; subtract
    # missing file samples separately. Partial silent intervals can remain.
    window_s = audit.window_end_s-audit.window_start_s
    audit['nonzero_recorded_coverage_pct'] = 100*(audit.read_frames/audit.samplerate-audit.all_zero_seconds)/window_s
    audit.to_csv(out/'audio_coverage_statistics.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7), layout='constrained')
    rng = np.random.default_rng(SEED)
    for ax, table, column, title in [(axes[0], quiet, 'slot_background_dbfs', 'Selected injection slots'),
                                    (axes[1], background, 'band_rms_dbfs', 'Full sampled context'),
                                    (axes[2], audit, 'nonzero_recorded_coverage_pct', 'Audio availability: 300-900 s')]:
        for i, group in enumerate(GROUPS):
            values = table[table.genotype == group][column].to_numpy()
            ax.boxplot([values], positions=[i], widths=.32, showfliers=False, patch_artist=True,
                       boxprops={'facecolor': COLORS[group], 'alpha': .7}, medianprops={'color': 'white'})
            ax.scatter(i+rng.uniform(-.06, .06, len(values)), values, s=26, facecolors='white', edgecolors=COLORS[group], lw=.9, zorder=3)
        ax.set_xticks([0, 1], ['WT', 'Het']); ax.set_title(title, fontsize=10)
        ax.yaxis.grid(True, color='#EDF0F2', lw=.65)
    axes[0].set_ylabel('45-125 kHz background RMS (dBFS)')
    axes[2].set_ylabel('Recorded time outside all-zero 1 s bins (%)')
    axes[2].set_ylim(0, 103)
    fig.suptitle('Background level and recorded audio availability', fontsize=11)
    fig.supxlabel('Full context includes natural sounds. Availability excludes missing samples and complete all-zero seconds, a coarse audit.', fontsize=8)
    save(fig, out, 'detector_background_levels')
    quiet.to_csv(out/'selected_slot_background_statistics.csv', index=False)


def audit_templates(out: Path, templates: pd.DataFrame) -> None:
    """Display original and cleaned spectral tracks using an independent FFT."""
    archive = np.load(out/'template_waveforms.npz')
    fig, axes = plt.subplots(len(templates), 2, figsize=(8.8, 2.0*len(templates)), layout='constrained', squeeze=False)
    frequency = np.fft.rfftfreq(SPECTROGRAM_WINDOW, 1/SAMPLE_RATE)/1000
    keep = (frequency >= 40) & (frequency <= 135)
    for i, r in enumerate(templates.itertuples()):
        waves = [archive[r.template_id+'_raw'], archive[r.template_id+'_clean']]
        specs = []
        for wave in waves:
            padded = np.pad(wave, (0, max(0, SPECTROGRAM_WINDOW-len(wave))))
            frames = np.lib.stride_tricks.sliding_window_view(padded, SPECTROGRAM_WINDOW)[::SPECTROGRAM_HOP]
            specs.append(np.abs(np.fft.rfft(frames*np.hanning(SPECTROGRAM_WINDOW), axis=1)).T)
        reference = max(float(s.max()) for s in specs)
        for j, spec in enumerate(specs):
            db = 20*np.log10(np.maximum(spec, reference*1e-6)/reference)
            end_ms = ((spec.shape[1]-1)*SPECTROGRAM_HOP+SPECTROGRAM_WINDOW)/SAMPLE_RATE*1000
            axes[i, j].imshow(db[keep], origin='lower', aspect='auto', cmap='magma', vmin=-55, vmax=0,
                              extent=[0, end_ms, frequency[keep][0], frequency[keep][-1]])
            axes[i, j].set_ylabel('Frequency (kHz)' if j == 0 else '')
            axes[i, j].set_xlabel('Time in template (ms)')
            axes[i, j].set_title(f'{r.template_id}: {r.duration_ms:.1f} ms | '+('original' if j == 0 else 'cleaned'), fontsize=9)
    fig.suptitle('Fixed real-whistle library: spectral audit', fontsize=12)
    fig.supxlabel('Original and cleaned views share their template-specific intensity reference; display range 55 dB.', fontsize=8)
    save(fig, out, 'detector_template_audit')


def detection_stage_summary(out: Path, sessions: pd.DataFrame) -> pd.DataFrame:
    """Export descriptive equal-recording stage means and complete denominators.

    Levels and templates are repeated within recording. Totals document support;
    they are not the denominator for the displayed equal-recording probability.
    Candidate-minus-retained is a per-recording coverage loss before averaging.
    No new hypothesis test or selected-peak significance claim is introduced.
    """
    rows = []
    for keys, table in sessions.groupby(['mode', 'variant', 'genotype', 'target_level'], sort=True):
        mode, variant, group, level = keys
        for stage, column in [('identifier', 'p_identifier'), ('retained', 'probability')]:
            rows.append(dict(mode=mode, variant=variant, genotype=group, target_level=level,
                stage=stage, mean_session_probability=float(table[column].mean()),
                n_recordings_defined=int(table[column].notna().sum()),
                n_recordings_manifest=len(table), n_injected_total=int(table.n_injected.sum()),
                n_eligible_total=int(table.n_eligible.sum()),
                n_sham_retained_total=int(table.n_sham_retained.sum()),
                n_eligible_min=int(table.n_eligible.min()), n_eligible_max=int(table.n_eligible.max()),
                mean_identifier_minus_retained=float((table.p_identifier-table.probability).mean()),
                weighting='equal recording means; fixed template library',
                interpretation='descriptive stage coverage, no additional test'))
    result = pd.DataFrame(rows)
    result.to_csv(out/'detection_stage_summary.csv', index=False)
    return result


def sensitivity_analysis_provenance(out: Path, auc: pd.DataFrame) -> None:
    """Record sensitivity inference settings without altering its wrapper cache.

    Read the driver's literal settings and function source through Python's AST;
    do not import its detector or execute the numerical experiment again.
    """
    if 'exploratory_context_sensitivity' not in auc or not boolean(auc.exploratory_context_sensitivity).all():
        return
    script = ROOT/'scripts'/'detector_recovery.py'
    source = script.read_text(encoding='utf-8')
    tree = ast.parse(source)
    names = {'SEED': 'seed', 'N_BOOT': 'n_bootstrap', 'N_PERM': 'n_permutations'}
    settings = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    settings[names[target.id]] = ast.literal_eval(node.value)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'statistics')
    inference = ast.get_source_segment(source, function)+'\n'
    metadata = dict(**settings, inference_sha256=hashlib.sha256(inference.encode()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        recovery_trials_sha256=hashlib.sha256((out/'recovery_trials.csv').read_bytes()).hexdigest(),
        python=sys.version, executable=sys.executable, numpy=np.__version__, pandas=pd.__version__,
        experiment='separate exploratory intact-context sensitivity; original primary results preserved',
        correction_family='two exploratory AUC comparisons; separate Holm family',
        replaced_recording='29539', replacement_context_start_s=540.)
    (out/'analysis_provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def reports(out: Path, tables: dict) -> None:
    """Write bilingual, source-grounded conclusions with explicit experiment limits."""
    auc, trials, sessions = tables['auc'], tables['trials'], tables['sessions']
    sham_n = int(boolean(trials.sham_retained).sum())
    sham_candidate_n = int(boolean(trials.sham_candidate).sum())
    excluded = int((sessions.n_injected-sessions.n_eligible).sum())
    background = tables['background']
    actual_counts = background.groupby(['genotype', 'start_s']).size()
    allocation = '; '.join(group+': '+', '.join(f'{start:g} s={actual_counts.get((group, start), 0)}'
                           for start in [360., 540., 720.]) for group in GROUPS)
    replacement = background[background.animal_id == '29539'].iloc[0]
    sensitivity_run = ('exploratory_context_sensitivity' in auc and
                       boolean(auc.exploratory_context_sensitivity).all())
    context_notice = (f'29539: proposed {replacement.proposed_start_s:g} s; actual {replacement.start_s:g} s; '
                      f'exactly zero samples in the selected context: {100*replacement.fraction_exact_zero_samples:.2f}%.')
    lines = ['# Detector recovery / Récupération des appels injectés', '',
        '## Résultats / Results', '',
        '| Mode | WT normalized AUC | Het normalized AUC | Het - WT [95% CI] | Raw p | Holm p | n WT / Het |',
        '|---|---:|---:|---|---:|---:|---:|']
    for r in auc.itertuples():
        lines.append(f'| {r.mode} | {number(r.auc_wt)} | {number(r.auc_het)} | {number(r.effect_het_minus_wt)} '
                     f'[{number(r.ci_low)}; {number(r.ci_high)}] | {number(r.p)} | {number(r.p_holm)} | {r.n_wt}/{r.n_het} |')
    if sensitivity_run:
        lines += ['', 'Cette sortie est une sensibilité exploratoire distincte: seul le contexte de 29539 est remplacé '
                  'par 540-600,15 s. Les résultats principaux originaux sont conservés. Les deux AUC partagent une nouvelle '
                  'famille Holm exploratoire; ce contrôle choisi après découverte de l’audio nul n’est pas une confirmation indépendante.',
                  'This is a separate exploratory sensitivity: only recording 29539 uses the 540-600.15 s context. '
                  'Original primary results remain unchanged. Its two AUCs share a separate exploratory Holm family. '
                  'This check was selected after discovering zero audio and is not independent confirmation.']
    lines += ['', 'Les différences sont estimées sur les moyennes par enregistrement. Un p non significatif ne prouve pas une sensibilité identique. '
              'Les résultats concernent la bibliothèque et les contextes testés; ils ne démontrent pas que des appels Het manqués expliquent les différences biologiques. '
              'Ils n’excluent pas non plus une émission Het plus faible, une distance différente au microphone ou une morphologie vocale non représentée.',
              'Differences concern recording-level means. A nonsignificant p does not establish equal sensitivity. '
              'These results do not prove that missed Het calls explain biological vocalization differences. '
              'They also do not exclude quieter Het emission, different microphone distance, or untested vocal morphology.', '',
        '## Diagnostic des étapes / Detection-stage diagnostic', '',
        '| Mode | Group | Descriptive peak level | Peak retained (%) | Highest tested level | Identifier there (%) | Retained there (%) |',
        '|---|---|---:|---:|---:|---:|---:|']
    summary = tables['stage_summary']
    pooled = summary[summary.variant == 'pooled']
    for mode in MODES:
        for group in GROUPS:
            sub = pooled[(pooled['mode'] == mode) & (pooled.genotype == group)]
            retained = sub[sub.stage == 'retained'].sort_values('target_level')
            peak = retained.loc[retained.mean_session_probability.idxmax()]
            last = retained.iloc[-1]
            identifier = sub[(sub.stage == 'identifier') & (sub.target_level == last.target_level)].iloc[0]
            lines.append(f'| {mode} | {group} | {peak.target_level:g} | {100*peak.mean_session_probability:.2f} | '
                         f'{last.target_level:g} | {100*identifier.mean_session_probability:.2f} | {100*last.mean_session_probability:.2f} |')
    lines += ['', 'Les courbes finales ne sont pas monotones: la conservation diminue aux niveaux élevés alors que '
              'l’identification reste élevée. La perte observée se situe entre les candidats identifiés et la sélection finale '
              'du classificateur bruit. Les pics du tableau sont choisis descriptivement sur ces données; ils ne définissent pas '
              'un seuil biologique optimal. Des modifications des masques et des images d’entrée peuvent également contribuer au rejet.',
              'Final recovery is nonmonotonic: retained calls decline at high levels while identifier coverage remains high. '
              'The observed loss occurs between identified candidates and final noise-classifier selection. '
              'The table selects empirical peaks descriptively, not an optimal biological threshold. '
              'Changes to segmentation masks and classifier input images may also contribute to rejection.', '',
              '| Mode | Group | Paired recordings | Isolated minus pooled recovery (percentage points) |',
              '|---|---|---:|---:|']
    for (mode, group), sub in tables['isolated'].groupby(['mode', 'genotype']):
        paired = sub.dropna(subset=['pooled_probability', 'isolated_probability'])
        delta = float((paired.isolated_probability-paired.pooled_probability).mean())
        lines.append(f'| {mode} | {group} | {len(paired)} | {100*delta:.2f} |')
    lines += ['', 'Les améliorations au niveau isolé montrent que les autres injections du défi commun modifient '
              'la récupération au niveau de transition. Ce contrôle local ne mesure pas une courbe complète de détection '
              'd’appels isolés et ne justifie pas une extrapolation à toutes les amplitudes.',
              'Improvement at the isolated transition level shows that other injections in the pooled challenge affect recovery. '
              'This local check does not measure a complete isolated-call detection curve or justify extrapolation across all amplitudes.', '',
        '## Méthode / Methods', '',
        '- 24 enregistrements, 12 WT et 12 Het. Le plan initial répartit quatre enregistrements par génotype entre 360, 540 et 720 s. '
        'Les appels naturels et les éventuelles portions nulles sont conservés dans les fenêtres de 60,15 s.',
        '- 24 recordings, 12 WT and 12 Het. Planned starts at 360, 540 and 720 s were balanced four per genotype. '
        'Actual allocation: '+allocation+'. '+context_notice+' '
        'The original 360 s replacement contains approximately five initial all-zero seconds. '
        'Only active injection slots require background RMS >1e-7. Intact-context sensitivity uses 540 s and is labeled separately.',
        '- Quatre modèles de vrais sifflements provenant de dyades à souris expérimentale WT, audités visuellement, environ 20-65 ms. '
        'Leur origine ne donne pas l’identité du locuteur. Filtrage Wiener spectral avant normalisation RMS sur le support actif.',
        '- Four visually audited real-whistle templates from WT-resident dyads, approximately 20-65 ms. '
        'The caller is unknown. Spectral Wiener cleaning precedes active-support band-RMS normalization. '
        'The experiment does not calibrate very short ambiguous spikes or every USV morphology.',
        '- Défi commun: 28 injections (quatre modèles × sept niveaux) dans chaque fichier de chaque mode. '
        'Les niveaux partagent donc la normalisation globale et le contexte de segmentation. '
        'Le contrôle isolé répète uniquement un niveau de transition aux mêmes positions.',
        '- The shared challenge contains 28 injections (four templates × seven levels) per mode. '
        'Its levels interact through global normalization and segmentation context. '
        'The isolated transition check uses identical positions but cannot establish equivalence at every level.',
        '- Amplitude reçue numérique en dBFS RMS dans 45-125 kHz. Le SNR rapporte ce signal au RMS du fond dans la même bande et le même support actif. '
        'Ce ne sont ni des dB SPL calibrés ni le volume émis par une souris.',
        '- Digital received amplitude is active-support band RMS dBFS, not calibrated SPL or biological source loudness. '
        'SNR uses the same USV band and active time support in the injection slot.',
        '- Appariement temporel un-à-un, IoU ≥ 0,30. Identification et conservation après classificateur bruit sont séparées. '
        'Les fenêtres déjà détectées comme appel dans le sham sont exclues des deux dénominateurs.',
        '- One-to-one temporal IoU matching (≥0.30). Identifier coverage and noise-classifier-retained recovery are separate stages. '
        'Sham-retained slots are excluded from both denominators. Identifier coverage uses detected candidate intervals, rather than candidate novelty.',
        f'- Sham matches across injection-manifest rows: {sham_candidate_n} candidate, {sham_n} retained. '
        f'{excluded} trial rows excluded by retained-sham matching. These counts include the repeated pooled/isolated variants; '
        'they are not independent false-positive observations.',
        '- Une session est l’unité statistique. IC ponctuels par bootstrap des enregistrements au sein du génotype. '
        'Les mêmes courbes entières sont permutées entre génotypes à l’intérieur du virus, en conservant les effectifs.',
        '- The recording is the inferential unit. Pointwise bootstrap intervals resample recordings within genotype. '
        'Genotype permutations preserve group counts within virus and keep each complete recovery curve together. '
        'Recordings are assumed independent and exchangeable within virus; reused partners may leave residual dependence.',
        '- Des bandes bootstrap peuvent se réduire à [0,0] ou [1,1] lorsque toutes les sessions ont le même résultat. '
        'Elles ne prouvent ni une impossibilité ni une certitude dans la population.',
        '- Bootstrap bands may collapse to [0,0] or [1,1] when all recordings have the same recovery. '
        'Those empirical limits do not establish population impossibility or certainty.',
        ('- Deux AUC normalisées définissent une famille Holm exploratoire de sensibilité.' if sensitivity_run else
         '- Deux AUC normalisées définissent la famille principale Holm.')+' Les 14 comparaisons de niveaux définissent une famille secondaire Holm. '
        'Les intervalles de confiance ne sont pas simultanés. La bibliothèque de quatre modèles est fixe; '
        'le bootstrap ne généralise pas à toutes les morphologies possibles.',
        ('- Two normalized-AUC tests share an exploratory sensitivity Holm correction.' if sensitivity_run else
         '- Two normalized-AUC tests share the primary Holm correction.')+' Fourteen level comparisons share a separate secondary Holm correction. '
        'Intervals are pointwise. The four-template library is fixed, so bootstrap uncertainty does not cover arbitrary call morphology.',
        '- Le RMS du contexte complet inclut les appels naturels et autres sons. Le RMS des slots sélectionnés mesure leur fond local. '
        'Ces diagnostics ne suffisent pas à conclure à un bruit caché ou à un mécanisme causal.',
        '- Full-context RMS includes natural calls and other sounds. Selected-slot RMS describes local injection backgrounds. '
        'Neither measure alone establishes a hidden-noise explanation or a causal mechanism.', '',
        '## Audio nul et analyses précédentes / Zero audio and preceding analyses', '',
        'L’audit 300-900 s trouve 254 secondes entièrement nulles pour WT 29999, 129 pour WT 30000, '
        '244 pour Het 29539 et sept pour Het 31101. Le fichier 31101 contient en outre environ 2,43 s manquantes. '
        'WT 31337 et 31315 présentent chacun une seconde entièrement nulle. Les petits nombres d’échantillons zéro isolés '
        'peuvent simplement résulter de la quantification; ils ne sont pas assimilés à des secondes de perte audio.',
        'Dans le contexte original de 29539 à 360 s, des fenêtres spectrales entièrement nulles peuvent produire log10(0)=-inf '
        'puis une normalisation non finie et aucune détection, y compris pour des injections fortes. '
        'Ce mécanisme numérique diffère d’un simple fond acoustique élevé. Le contrôle à 540 s est une sensibilité exploratoire distincte.',
        'The 300-900 s audit identifies 254 complete all-zero seconds for WT 29999, 129 for WT 30000, '
        '244 for Het 29539, and seven for Het 31101. Recording 31101 additionally lacks approximately 2.43 s of audio. '
        'WT 31337 and 31315 each have one all-zero second. Isolated zero samples can be ordinary quantized values '
        'and are not counted as complete silent-second losses.',
        'In the original 360 s context for 29539, entirely zero spectral windows can produce log10(0)=-inf, '
        'nonfinite normalization and no detections, even for strong injections. '
        'This numerical mechanism differs from an elevated acoustic background. The 540 s control is a separately labeled exploratory sensitivity.',
        'La couverture représentée est un audit grossier: temps lu moins secondes entièrement nulles. '
        'Elle ne certifie pas que chaque échantillon restant contient un enregistrement exploitable. '
        'Les analyses comportement-vocalisation précédentes utilisaient l’exposition vidéo, sans intersection avec une exposition audio valide. '
        'La présente calibration ne corrige pas leurs taux, probabilités, corrélations ou ANOVA. Une réanalyse distincte devrait '
        'croiser disponibilité audio et vidéo avant de considérer l’absence d’appel comme observable.',
        'Displayed coverage is a coarse audit: recorded duration minus complete all-zero seconds. '
        'It does not certify every remaining sample as usable audio. Preceding behavior-vocalization analyses used video exposure '
        'without intersecting it with valid-audio exposure. This calibration does not correct their rates, probabilities, correlations '
        'or ANOVA. A separate reanalysis should intersect audio and video availability before treating call absence as observed.', '',
        '## Fichiers / Files', '',
        '- [Primary AUC statistics](primary_auc_statistics.csv), [all secondary level statistics](level_statistics.csv).',
        '- [Recording recovery](session_recovery.csv), [individual injection trials](recovery_trials.csv).',
        '- [Descriptive detection-stage support and probabilities](detection_stage_summary.csv).',
        '- [Template provenance](template_manifest.csv), [background contexts](background_manifest.csv), [injection design](injection_manifest.csv).',
        '- [Isolated challenge sensitivity](isolated_level_sensitivity.csv), [validation checks](recovery_validation.csv).',
        '- [Audio audit](audio_dropout_audit.csv), [one-second audio inventory](audio_dropout_bins.csv), [coarse audio coverage](audio_coverage_statistics.csv).',
        '- Main curves, identifier curves, isolated sensitivity, background levels and template audit are exported as PNG, PDF and SVG.', '']
    (out/'RAPPORT_RECUPERATION_DETECTEUR.md').write_text('\n'.join(lines), encoding='utf-8')


def make_outputs(out: Path) -> None:
    """Validate completed numeric artifacts before rendering and writing reports."""
    out = Path(out)
    names = {'manifest': 'injection_manifest', 'background': 'background_manifest',
             'templates': 'template_manifest', 'trials': 'recovery_trials', 'sessions': 'session_recovery',
             'curves': 'curve_statistics', 'auc': 'primary_auc_statistics', 'levels': 'level_statistics',
             'isolated': 'isolated_level_sensitivity', 'audio_audit': 'audio_dropout_audit',
             'audio_bins': 'audio_dropout_bins'}
    tables = {key: pd.read_csv(out/f'{name}.csv', dtype={'animal_id': str}) for key, name in names.items()}
    validate(out, tables); configure()
    draw_curves(out, tables['curves'], tables['auc'])
    identifier_curves(out, tables['sessions'])
    diagnostic_figures(out, tables)
    audit_templates(out, tables['templates'])
    tables['stage_summary'] = detection_stage_summary(out, tables['sessions'])
    reports(out, tables)
    sensitivity_analysis_provenance(out, tables['auc'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    make_outputs(parser.parse_args().output_dir)
