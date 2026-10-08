"""Independent intact-context sensitivity for recording 29539.

Prepare five genuine 540-600.15 s challenge files without changing the primary
experiment. Detection is an explicit separate stage. After both experiments
finish, combine all 24 recordings while replacing only this recording, and
label its two AUC contrasts as an exploratory sensitivity family.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import detector_recovery as driver

# Fixed sensitivity choices, isolated from the original experiment settings.
PRIMARY = driver.ROOT / 'lgdel_usv_analysis' / 'detector_recovery'
SENSITIVITY = PRIMARY / 'intact_context_sensitivity'
ANIMAL = '29539'
CONTEXT_START = 540.
REASON = ('Original 360 s replacement context contains literal zero audio; '
          'log10(0) and subsequent normalization can disable the unchanged detector.')
FAMILY = 'two exploratory intact-context AUC tests; separate Holm sensitivity family'


def fingerprint(primary: Path) -> dict:
    """Tie the sensitivity to validated primary inputs and the unchanged driver."""
    original = json.loads((primary/'provenance.json').read_text(encoding='utf-8'))['fingerprint']
    current = json.loads(json.dumps(driver.stamp()))
    if current != original:
        raise RuntimeError('Primary experiment fingerprint differs from the current detection driver.')
    return dict(primary=original, animal=ANIMAL, context_start_s=CONTEXT_START,
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), reason=REASON)


@contextmanager
def single_recording(output: Path, prepare_templates: bool = False):
    """Restore every temporary override, even if preparation or detection fails.

    The template library still needs the full source-genotype map. Only the
    recipient-recording loop is restricted. The prepare loop still shuffles
    twelve context choices for each genotype, preserving its original random
    number consumption and therefore its template/level schedule.
    """
    old_output, old_starts = driver.OUT, driver.CONTEXT_STARTS
    old_map = driver.metadata.GENOTYPE_MAP
    old_library = driver.template_library
    def full_map_library():
        limited = driver.metadata.GENOTYPE_MAP
        driver.metadata.GENOTYPE_MAP = old_map
        try:
            return old_library()
        finally:
            driver.metadata.GENOTYPE_MAP = limited
    driver.OUT = output
    driver.CONTEXT_STARTS = [CONTEXT_START]*3
    driver.metadata.GENOTYPE_MAP = {ANIMAL: old_map[ANIMAL]}
    if prepare_templates:
        driver.template_library = full_map_library
    try:
        yield
    finally:
        driver.OUT, driver.CONTEXT_STARTS = old_output, old_starts
        driver.metadata.GENOTYPE_MAP = old_map
        driver.template_library = old_library


def prepare(primary: Path, output: Path) -> None:
    """Write only the new context experiment; never launch expensive detection."""
    fp = fingerprint(primary)
    output.mkdir(parents=True, exist_ok=True)
    provenance = output/'context_sensitivity_provenance.json'
    if provenance.exists():
        if json.loads(provenance.read_text())['fingerprint'] != fp:
            raise RuntimeError('Sensitivity fingerprint changed; choose a fresh output directory.')
        if (output/'injection_manifest.csv').exists():
            print('Validated intact-context preparation reused.', flush=True)
            return
    with single_recording(output, prepare_templates=True):
        driver.prepare()
    # Confirm the wrapper preserves the exact four-template/seven-level order.
    primary_design = pd.read_csv(primary/'injection_manifest.csv', dtype={'animal_id': str})
    experiment = pd.read_csv(output/'injection_manifest.csv', dtype={'animal_id': str})
    keys = ['mode', 'variant', 'slot', 'template_id', 'target_level']
    old = primary_design[primary_design.animal_id == ANIMAL].sort_values(keys)[keys].reset_index(drop=True)
    new = experiment.sort_values(keys)[keys].reset_index(drop=True)
    if not old.equals(new):
        raise AssertionError('The intact-context template/level schedule differs from the primary challenge.')
    bg = pd.read_csv(output/'background_manifest.csv', dtype={'animal_id': str})
    if len(bg) != 1 or not np.isclose(bg.start_s.iloc[0], CONTEXT_START):
        raise AssertionError('Unexpected intact-context recording or start time.')
    if bg.fraction_exact_zero_samples.iloc[0] >= .05:
        raise AssertionError('The replacement context still contains substantial literal zero audio.')
    provenance.write_text(json.dumps(dict(fingerprint=fp, experiment='exploratory intact-context sensitivity',
        primary_preserved=str(primary), recording=ANIMAL, context_start_s=CONTEXT_START,
        reason=REASON, detection_jobs=5, family=FAMILY), indent=2), encoding='utf-8')
    print('Prepared five intact-context files only. Detection must be started explicitly.', flush=True)


def detect(primary: Path, output: Path) -> None:
    """Run the unchanged detector on this recording after workers become free."""
    fp = fingerprint(primary)
    provenance = json.loads((output/'context_sensitivity_provenance.json').read_text())
    if provenance['fingerprint'] != fp:
        raise RuntimeError('Stale intact-context preparation.')
    with single_recording(output):
        driver.run()


def combine(primary: Path, experiment: Path) -> None:
    """Export a separately labeled full-cohort sensitivity without replacing originals."""
    fp = fingerprint(primary)
    provenance = json.loads((experiment/'context_sensitivity_provenance.json').read_text())
    if provenance['fingerprint'] != fp:
        raise RuntimeError('Stale intact-context experiment.')
    original = pd.read_csv(primary/'recovery_trials.csv', dtype={'animal_id': str})
    replacement = pd.read_csv(experiment/'recovery_trials.csv', dtype={'animal_id': str})
    if original.animal_id.nunique() != 24 or len(original) != 24*64:
        raise RuntimeError('The primary 24-recording detection experiment is not complete.')
    if set(replacement.animal_id) != {ANIMAL} or len(replacement) != 64:
        raise RuntimeError('The intact-context five-file detection experiment is not complete.')
    output = experiment/'combined_24_recordings'
    output.mkdir(parents=True, exist_ok=True)
    mixed = pd.concat([original[original.animal_id != ANIMAL], replacement], ignore_index=True)
    mixed.to_csv(output/'recovery_trials.csv', index=False)
    old_manifest = pd.read_csv(primary/'injection_manifest.csv', dtype={'animal_id': str})
    new_manifest = pd.read_csv(experiment/'injection_manifest.csv', dtype={'animal_id': str})
    pd.concat([old_manifest[old_manifest.animal_id != ANIMAL], new_manifest], ignore_index=True).to_csv(output/'injection_manifest.csv', index=False)
    background = pd.read_csv(primary/'background_manifest.csv', dtype={'animal_id': str})
    replacement_bg = pd.read_csv(experiment/'background_manifest.csv', dtype={'animal_id': str})
    # Retain the originally balanced planned start, while reporting the actual
    # new 540 s context and its reason. Nothing is rewritten in the primary cache.
    replacement_bg['proposed_start_s'] = background.loc[background.animal_id == ANIMAL, 'proposed_start_s'].iloc[0]
    replacement_bg['substituted_zero_background'] = True
    background['context_sensitivity'] = False
    background['context_sensitivity_reason'] = ''
    replacement_bg['context_sensitivity'] = True
    replacement_bg['context_sensitivity_reason'] = REASON
    pd.concat([background[background.animal_id != ANIMAL], replacement_bg], ignore_index=True).to_csv(output/'background_manifest.csv', index=False)
    for name in ['template_manifest.csv', 'template_waveforms.npz', 'audio_dropout_audit.csv', 'audio_dropout_bins.csv']:
        shutil.copy2(primary/name, output/name)
    old_output = driver.OUT
    try:
        driver.OUT = output
        driver.statistics()
    finally:
        driver.OUT = old_output
    auc = pd.read_csv(output/'primary_auc_statistics.csv')
    auc['family'] = FAMILY
    auc['exploratory_context_sensitivity'] = True
    auc.to_csv(output/'primary_auc_statistics.csv', index=False)
    (output/'provenance.json').write_text(json.dumps(dict(fingerprint=fp,
        experiment='exploratory intact-context sensitivity, not the primary experiment',
        primary_results_preserved=str(primary), replaced_recording=ANIMAL,
        replacement_context_start_s=CONTEXT_START, reason=REASON, family=FAMILY), indent=2), encoding='utf-8')
    from detector_recovery_figures import make_outputs
    make_outputs(output)
    print(f'Separate exploratory full-cohort sensitivity: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'detect', 'combine'], default='prepare')
    parser.add_argument('--primary-dir', type=Path, default=PRIMARY)
    parser.add_argument('--output-dir', type=Path, default=SENSITIVITY)
    args = parser.parse_args()
    primary, output = args.primary_dir.resolve(), args.output_dir.resolve()
    if output == primary or primary in output.parents and output.name == 'combined_24_recordings':
        raise ValueError('Choose a separate sensitivity experiment directory.')
    {'prepare': prepare, 'detect': detect, 'combine': combine}[args.stage](primary, output)
