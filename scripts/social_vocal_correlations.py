"""Session-level social-behavior / dyad-vocalization correlation workflow.

Read raw detector tables without modifying them. Build frame-resolution social
episodes and independent acoustic summaries on identical observed intervals.
Then test conditional rank associations by permutations within genotype/virus
strata, with one complete multiple-testing family. Run in an IDE or via .cmd.
"""
from __future__ import annotations

# Limit BLAS overhead for many small permutation products before importing NumPy.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

import attr_common as metadata

# Numerical configuration. Fix these before inspecting associations.
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent
OUTPUT = ROOT / "lgdel_usv_analysis" / "social_vocal_correlations"
TRIAL_PLAN = DATA_ROOT / "pykaboo_trial_plan_completed.xlsx"
KINEMATICS = ROOT / "lgdel_usv_analysis" / "attribution" / "beh_bins.csv.gz"
WINDOW = (300.0, 900.0)
FPS = 30.000063
MIN_COVERAGE = .95
GAP_SECONDS = 0.0
GAP_SENSITIVITY = .2
BOUT_GAP_SECONDS = .25
BOUT_GAP_SENSITIVITY = .5
N_PERM = 49999
N_BOOT = 2000
SEED = 20261007
ALPHA = .05
MIN_N = 10
MIN_UNIQUE = 3
CACHE_VERSION = 1
SOCIAL = ["nose2anogenital", "nose2body", "nose2nose", "oriented_toward",
          "following", "chasing", "approach", "sidebyside", "sidereside",
          "withdrawal_from_partner", "withdrawal_after_contact", "escape", "fighting"]
MUTUAL = {"nose2nose", "sidebyside", "sidereside", "fighting"}
BEHAVIOR_METRICS = ["cumulative_duration_s", "mean_episode_duration_s", "episode_rate_per_min"]
VOCAL_METRICS = ["call_rate_per_min", "vocal_output_s_per_min", "median_call_duration_ms",
                 "median_frequency_khz", "median_bandwidth_khz", "bout_rate_per_min",
                 "mean_calls_per_bout", "class_entropy_bits"]
REUSED_STIM_RESIDENTS = {a for a, stim in metadata.STIM_MAP.items()
                       if stim is not None and list(metadata.STIM_MAP.values()).count(stim) > 1}


def source_paths(animal: str) -> tuple[Path, Path]:
    """Resolve raw behavior and USV detector sources from the current data root."""
    session = DATA_ROOT / animal / "baseline" / "1"
    return (session / f"{animal}_1_baseline_live_detections.csv",
            session / f"{animal}_1_baseline_outputs" / f"{animal}_1_baseline_stats.csv")


def source_fingerprint() -> dict:
    """Stamp raw inputs and hash numerical code separately from figure styling."""
    paths = [TRIAL_PLAN, KINEMATICS]
    for animal in metadata.GENOTYPE_MAP:
        paths.extend(source_paths(animal))
    stamps = {}
    for path in paths:
        stat = path.stat()
        stamps[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    functions = (episodes, observed_call_duration, vocal_summary, extract_tables, ranks_residual, coefficient,
                 permutation_indices, test_pair, analyze, sensitivity)
    numeric_code = "\n".join(inspect.getsource(fn) for fn in functions)
    return {"version": CACHE_VERSION, "sources": stamps,
            "numeric_code_sha256": hashlib.sha256(numeric_code.encode()).hexdigest(),
            "metadata_sha256": hashlib.sha256(Path(metadata.__file__).read_bytes()).hexdigest(),
            "window": WINDOW, "fps": FPS, "coverage": MIN_COVERAGE,
            "gap": GAP_SECONDS, "gap_sensitivity": GAP_SENSITIVITY,
            "bout_gap": BOUT_GAP_SECONDS, "bout_gap_sensitivity": BOUT_GAP_SENSITIVITY,
            "n_perm": N_PERM, "n_boot": N_BOOT, "seed": SEED,
            "behavior_metrics": BEHAVIOR_METRICS, "vocal_metrics": VOCAL_METRICS,
            "social": SOCIAL, "mutual": sorted(MUTUAL), "alpha": ALPHA,
            "min_n": MIN_N, "min_unique": MIN_UNIQUE}


def episodes(active: np.ndarray, observed: np.ndarray, dt: np.ndarray, gap: float) -> dict:
    """Segment episodes, never merging across missing observations.

    Episode active duration sums positive frames only. Bridged inactive gaps are
    excluded from cumulative duration; sensitivity therefore cannot invent time.
    Episodes touching missing intervals or window edges are marked censored.
    """
    indices = np.flatnonzero(active & observed)
    if not len(indices):
        return dict(cumulative_duration_s=0., mean_episode_duration_s=np.nan,
                    median_episode_duration_s=np.nan, episode_count=0,
                    episode_rate_per_min=0., censored_episode_count=0)
    breaks = np.diff(indices) > 1
    if gap > 0:
        for i in np.flatnonzero(breaks):
            between = slice(indices[i]+1, indices[i+1])
            if observed[between].all() and dt[between].sum() <= gap+1e-10:
                breaks[i] = False
    groups = np.split(indices, np.flatnonzero(breaks)+1)
    durations = np.array([dt[group].sum() for group in groups])
    censored = sum(group[0] == 0 or group[-1] == len(active)-1
                   or not observed[max(group[0]-1, 0)]
                   or not observed[min(group[-1]+1, len(active)-1)] for group in groups)
    exposure = dt[observed].sum()
    return dict(cumulative_duration_s=float(durations.sum()),
                mean_episode_duration_s=float(durations.mean()),
                median_episode_duration_s=float(np.median(durations)),
                episode_count=len(groups), episode_rate_per_min=len(groups)/(exposure/60),
                censored_episode_count=int(censored))


def observed_call_duration(start: float, end: float, first: int, valid: np.ndarray) -> float:
    """Intersect an acoustic interval with actual shared video-frame exposure."""
    start,end=max(start,WINDOW[0]),min(end,WINDOW[1])
    if end<=start:return 0.
    frames=np.arange(int(np.floor(start*FPS)),int(np.ceil(end*FPS)))
    indices=frames-first
    inside=(indices>=0)&(indices<len(valid))
    frames,indices=frames[inside],indices[inside]
    duration=np.maximum(0,np.minimum((frames+1)/FPS,end)-np.maximum(frames/FPS,start))
    return float(duration[valid[indices]].sum())


def vocal_summary(calls: pd.DataFrame, exposure: float, gap: float) -> dict:
    """Summarize all dyad calls without geometric attribution to either animal."""
    n = len(calls)
    if n:
        starts = calls['start(s)'].to_numpy(float)
        ends = calls['end(s)'].to_numpy(float)
        # Missing shared coverage forces a new bout even if retained onsets are close.
        segments=calls.observation_segment.to_numpy(int)
        n_bouts,latest_end=1,ends[0]
        for i in range(1,n):
            if segments[i]!=segments[i-1] or starts[i]-latest_end>gap:
                n_bouts+=1
                latest_end=ends[i]
            else:
                latest_end=max(latest_end,ends[i])
        counts = calls.class_top1.dropna().value_counts().to_numpy(float)
        prob = counts/counts.sum() if len(counts) else np.array([])
        entropy = float(-(prob*np.log2(prob)).sum()) if len(prob) else np.nan
    else:
        n_bouts, entropy = 0, np.nan
    return dict(call_count=n, call_rate_per_min=n/(exposure/60),
                vocal_output_s_per_min=float(calls.observed_duration_s.sum())/(exposure/60),
                median_call_duration_ms=float(calls['duration(ms)'].median()) if n else np.nan,
                median_frequency_khz=float(calls.avg_freq.median())/1000 if n else np.nan,
                median_bandwidth_khz=float(calls.bandwidth.median())/1000 if n else np.nan,
                bout_count=n_bouts, bout_rate_per_min=n_bouts/(exposure/60),
                mean_calls_per_bout=n/n_bouts if n_bouts else np.nan,
                class_entropy_bits=entropy, entropy_low_call_count=n<20)


def extract_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join exact frame coverage, behaviors, corrected identities, virus and acoustics."""
    plan = pd.read_excel(TRIAL_PLAN)
    plan['animal_id'] = plan['Animal ID'].astype(str)
    plan = plan.set_index('animal_id')
    if plan.index.duplicated().any():
        raise ValueError("Trial plan contains duplicated animal identities.")
    first = int(np.floor(WINDOW[0]*FPS))
    last = int(np.ceil(WINDOW[1]*FPS))
    frames = np.arange(first, last)
    dt = np.maximum(0, np.minimum((frames+1)/FPS, WINDOW[1]) - np.maximum(frames/FPS, WINDOW[0]))
    behavior_rows, vocal_rows, qa_rows = [], [], []
    kin = pd.read_csv(KINEMATICS, usecols=['animal_id','t','speed_m1','speed_m2','dist'],
                      dtype={'animal_id': str})
    kin = kin[(kin.t >= WINDOW[0]) & (kin.t < WINDOW[1])]
    for i, (animal, genotype) in enumerate(metadata.GENOTYPE_MAP.items(), 1):
        behavior_path, call_path = source_paths(animal)
        flags = [f'behavior_{b}' for b in SOCIAL]
        raw = pd.read_csv(behavior_path, usecols=['frame_id','mouse_id','behavior_backend']+flags)
        raw = raw[raw.frame_id.between(first, last-1)]
        if not set(raw.behavior_backend.dropna().unique()) <= {'rules'}:
            raise ValueError(f"Unrecognized behavior backend: {animal}")
        if not raw[flags].isin([0,1]).all().all():
            raise ValueError(f"Nonbinary or missing behavior flags: {animal}")
        duplicate_rows = int(raw.duplicated(['frame_id','mouse_id']).sum())
        duplicate_flags = raw[raw.duplicated(['frame_id','mouse_id'],keep=False)]
        contradictory_duplicates = int(duplicate_flags.groupby(['frame_id','mouse_id'])[flags].nunique().gt(1).any(axis=1).sum())
        # Aggregate repeated inference rows once, preserving any positive flag.
        grids, observed = {}, {}
        for actor in [1,2]:
            grouped = raw[raw.mouse_id == actor].groupby('frame_id')[flags].max()
            observed[actor] = np.isin(frames, grouped.index)
            grids[actor] = grouped.reindex(frames).fillna(0).to_numpy(bool)
        valid = observed[1] & observed[2]
        exposure = float(dt[valid].sum())
        coverage = exposure/(WINDOW[1]-WINDOW[0])
        virus = str(plan.loc[animal,'virus'])
        if str(plan.loc[animal,'genotype_pyrat']).strip().upper() != genotype:
            raise ValueError(f"Genotype source disagreement: {animal}")
        common = dict(animal_id=animal, genotype=genotype, virus=virus,
                      stratum=genotype+'_'+virus, exposure_s=exposure, coverage=coverage,
                      eligible=coverage >= MIN_COVERAGE, stim_id=metadata.STIM_MAP.get(animal))
        calls = pd.read_csv(call_path).sort_values('start(s)')
        calls = calls[(calls['start(s)'] >= WINDOW[0]) & (calls['start(s)'] < WINDOW[1])]
        call_frames = np.floor(calls['start(s)'].to_numpy(float)*FPS).astype(int)-first
        calls_valid = calls.iloc[np.flatnonzero(valid[call_frames])].copy()
        calls_valid['observation_segment']=np.cumsum(~valid)[call_frames[valid[call_frames]]]
        calls_valid['observed_duration_s']=[observed_call_duration(s,e,first,valid)
                                          for s,e in zip(calls_valid['start(s)'],calls_valid['end(s)'])]
        lost_duration=max(0,float((calls_valid['end(s)']-calls_valid['start(s)']).sum())-
                          float(calls_valid.observed_duration_s.sum()))
        if (calls_valid['duration(ms)'] <= 0).any():
            raise ValueError(f"Nonpositive acoustic duration: {animal}")
        vocal = vocal_summary(calls_valid, exposure, BOUT_GAP_SECONDS)
        alternate = vocal_summary(calls_valid, exposure, BOUT_GAP_SENSITIVITY)
        k = kin[kin.animal_id == animal]
        vocal_rows.append({**common, **vocal,
                           'bout_rate_gap_sensitivity': alternate['bout_rate_per_min'],
                           'calls_per_bout_gap_sensitivity': alternate['mean_calls_per_bout'],
                           'mean_resident_speed_px_s': k.speed_m1.mean(),
                           'mean_partner_speed_px_s': k.speed_m2.mean(),
                           'median_distance_px': k.dist.median(),
                           'total_window_calls': len(calls), 'excluded_unobserved_calls': len(calls)-len(calls_valid)})
        mismatch, overlap_total = 0, 0
        for j,b in enumerate(SOCIAL):
            a1,a2 = grids[1][:,j],grids[2][:,j]
            if b in MUTUAL:
                mismatch += int(((a1 != a2)&valid).sum())
            overlap_total += int((a1&a2&valid).sum())
            roles = [('dyad',a1|a2)]
            if b not in MUTUAL:
                roles += [('resident',a1),('partner',a2)]
            for role,active in roles:
                values = episodes(active, valid, dt, GAP_SECONDS)
                alternate = episodes(active, valid, dt, GAP_SENSITIVITY)
                assert values['cumulative_duration_s'] <= exposure+1e-8
                assert np.isclose(values['cumulative_duration_s'], alternate['cumulative_duration_s'])
                behavior_rows.append({**common, 'behavior': b, 'role': role, **values,
                                      'mean_duration_gap_sensitivity': alternate['mean_episode_duration_s'],
                                      'episode_rate_gap_sensitivity': alternate['episode_rate_per_min']})
        qa_rows.append({**common,'duplicate_inference_rows':duplicate_rows,
                        'contradictory_duplicate_frame_actor_pairs':contradictory_duplicates,
                        'selected_call_duration_clipped_s':lost_duration,
                        'observed_resident_frames': int(observed[1].sum()),
                        'observed_partner_frames': int(observed[2].sum()),
                        'mutual_flag_disagreement_frames': mismatch,
                        'direction_overlap_frames_summed_over_behaviors': overlap_total,
                        'total_window_calls':len(calls),'analyzed_calls':len(calls_valid)})
        print(f"[{i}/24] {animal}: coverage={coverage:.3%}, calls={len(calls_valid)}/{len(calls)}", flush=True)
    return pd.DataFrame(behavior_rows),pd.DataFrame(vocal_rows),pd.DataFrame(qa_rows)


def ranks_residual(x: np.ndarray, strata: np.ndarray | None) -> np.ndarray:
    """Global average ranks, centered overall or within the conditioning strata."""
    r = rankdata(x, method='average', axis=-1)
    if strata is None:
        return r-r.mean(axis=-1,keepdims=True)
    for group in np.unique(strata):
        mask = strata == group
        r[...,mask] -= r[...,mask].mean(axis=-1,keepdims=True)
    return r


def coefficient(x: np.ndarray, y: np.ndarray, strata=None) -> float:
    """Pearson correlation of rank residuals, undefined for a constant residual."""
    rx,ry = ranks_residual(x,strata),ranks_residual(y,strata)
    denominator = np.linalg.norm(rx)*np.linalg.norm(ry)
    return float(np.dot(rx,ry)/denominator) if denominator > 1e-12 else np.nan


_PERM_CACHE: dict = {}


def permutation_indices(strata: np.ndarray) -> np.ndarray:
    """Reuse deterministic independent shuffles restricted to each valid stratum."""
    key = tuple(strata)
    if key not in _PERM_CACHE:
        seed = SEED+int(hashlib.sha256('|'.join(key).encode()).hexdigest()[:8],16)
        rng = np.random.default_rng(seed)
        indices = np.tile(np.arange(len(strata)),(N_PERM,1))
        for group in np.unique(strata):
            pos = np.flatnonzero(strata==group)
            indices[:,pos] = pos[np.argsort(rng.random((N_PERM,len(pos))),axis=1)]
        _PERM_CACHE[key] = indices
    return _PERM_CACHE[key]


def test_pair(x: np.ndarray, y: np.ndarray, strata: np.ndarray | None,
              seed: int, bootstrap: bool) -> dict:
    """Conditional-independence permutation test and stratified paired bootstrap.

    The permutation null is independence/exchangeability within each stratum,
    stronger than merely zero correlation. No normality assumption is required.
    Bootstrap intervals are descriptive pointwise intervals, not simultaneous.
    """
    n = len(x)
    ux,uy = len(np.unique(x)),len(np.unique(y))
    counts = np.unique(strata,return_counts=True)[1] if strata is not None else np.array([n])
    out = dict(n=n, n_strata=len(counts), min_stratum_n=int(counts.min()) if len(counts) else 0,
               exchangeable_n=int(counts[counts>1].sum()),
               unique_behavior=ux, unique_vocal=uy, rho=np.nan,p=np.nan,p_monte_carlo_se=np.nan,
               ci_low=np.nan,ci_high=np.nan,bootstrap_valid=0,
               loo_min=np.nan,loo_max=np.nan,loo_sign_stable=False,status='ok')
    if n < MIN_N or min(ux,uy)<MIN_UNIQUE:
        out['status']='insufficient_n_or_unique_values'
        return out
    rx,ry = ranks_residual(x,strata),ranks_residual(y,strata)
    den = np.linalg.norm(rx)*np.linalg.norm(ry)
    if den < 1e-12:
        out['status']='constant_after_adjustment'
        return out
    rho = float(rx@ry/den)
    groups = strata if strata is not None else np.array(['all']*n)
    indices = permutation_indices(groups)
    null = (ry[indices]@rx)/den
    out.update(rho=rho,p=float((1+(np.abs(null)>=abs(rho)-1e-12).sum())/(N_PERM+1)))
    out['p_monte_carlo_se']=float(np.sqrt(out['p']*(1-out['p'])/(N_PERM+1)))
    loo = []
    for i in range(n):
        mask = np.arange(n)!=i
        loo.append(coefficient(x[mask],y[mask],strata[mask] if strata is not None else None))
    finite = np.array(loo)[np.isfinite(loo)]
    if len(finite):
        out.update(loo_min=float(finite.min()),loo_max=float(finite.max()),
                   loo_sign_stable=bool(np.all(np.sign(finite)==np.sign(rho))))
    if bootstrap:
        rng = np.random.default_rng(seed)
        boot_idx = np.tile(np.arange(n),(N_BOOT,1))
        for group in np.unique(groups):
            pos = np.flatnonzero(groups==group)
            boot_idx[:,pos] = rng.choice(pos,size=(N_BOOT,len(pos)),replace=True)
        bx,by = ranks_residual(x[boot_idx],strata),ranks_residual(y[boot_idx],strata)
        denom = np.linalg.norm(bx,axis=1)*np.linalg.norm(by,axis=1)
        vals = np.divide((bx*by).sum(axis=1),denom,out=np.full(N_BOOT,np.nan),where=denom>1e-12)
        vals = vals[np.isfinite(vals)]
        if len(vals) >= .8*N_BOOT:
            lo,hi = np.quantile(vals,[.025,.975])
            out.update(ci_low=float(lo),ci_high=float(hi),bootstrap_valid=len(vals))
    return out


def analyze(behavior: pd.DataFrame, vocal: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every fixed pair, including absent and nonsignificant behaviors."""
    joined = behavior.merge(vocal,on=['animal_id','genotype','virus','stratum'],suffixes=('','_vocal'),validate='many_to_one')
    rows=[]
    for (role,b),table in joined.groupby(['role','behavior'],sort=False):
        scopes = ['adjusted_genotype_virus','pooled','WT','HET','adjusted_genotype'] if role=='dyad' else ['adjusted_genotype_virus']
        for bm in BEHAVIOR_METRICS:
            for vm in VOCAL_METRICS:
                for scope in scopes:
                    subset = table[table.eligible & table[bm].notna() & table[vm].notna()].copy()
                    if scope in ['WT','HET']:
                        subset=subset[subset.genotype==scope]
                    if scope=='adjusted_genotype_virus':
                        strata=subset.stratum.to_numpy(str)
                    elif scope=='adjusted_genotype':
                        strata=subset.genotype.to_numpy(str)
                    else:
                        strata=None
                    key=f'{role}|{b}|{bm}|{vm}|{scope}'
                    seed=SEED+int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
                    result=test_pair(subset[bm].to_numpy(float),subset[vm].to_numpy(float),strata,seed,
                                     bootstrap=scope=='adjusted_genotype_virus')
                    rows.append(dict(role=role,behavior=b,behavior_metric=bm,vocal_metric=vm,scope=scope,
                                     n_WT=int((subset.genotype=='WT').sum()),n_HET=int((subset.genotype=='HET').sum()),
                                     n_excluded=len(table)-len(subset),**result))
        print(f"Correlations: {role} {b}",flush=True)
    results=pd.DataFrame(rows)
    results['q_bh']=np.nan
    results['q_by']=np.nan
    results['p_holm']=np.nan
    # Main family contains all 312 planned dyad pairs, including untestable ones
    # assigned p=1 for adjustment. Sensitivity families are kept explicit.
    results['family'] = np.where(results.role=='dyad','dyad_','actor_secondary_')+results.scope
    for family,indices in results.groupby('family').groups.items():
        p=results.loc[indices,'p'].fillna(1).to_numpy(float)
        for method,column in [('fdr_bh','q_bh'),('fdr_by','q_by'),('holm','p_holm')]:
            results.loc[indices,column]=multipletests(p,method=method)[1]
    results['primary']=(results.role=='dyad')&(results.scope=='adjusted_genotype_virus')
    results['significant_bh']=results.q_bh<ALPHA
    return results


def sensitivity(behavior: pd.DataFrame,vocal: pd.DataFrame,results: pd.DataFrame) -> pd.DataFrame:
    """Quantify alternate segmentation, motion adjustment and reused-partner influence.

    These are descriptive coefficient checks with no selection of favorable settings.
    They do not introduce extra uncorrected significance claims.
    """
    joined=behavior.merge(vocal,on=['animal_id','genotype','virus','stratum'],suffixes=('','_vocal'))
    rows=[]
    for test in results[results.primary].itertuples():
        all_table=joined[(joined.role==test.role)&(joined.behavior==test.behavior)]
        table=all_table[all_table.eligible]
        bm,vm=test.behavior_metric,test.vocal_metric
        alt_b={'mean_episode_duration_s':'mean_duration_gap_sensitivity','episode_rate_per_min':'episode_rate_gap_sensitivity'}.get(bm,bm)
        alt_v={'bout_rate_per_min':'bout_rate_gap_sensitivity','mean_calls_per_bout':'calls_per_bout_gap_sensitivity'}.get(vm,vm)
        for name,xcol,ycol in [('episode_gap_0.2s',alt_b,vm),('vocal_bout_gap_0.5s',bm,alt_v),
                              ('exclude_reused_stim_partners',bm,vm),('adjust_motion_and_distance',bm,vm),
                              ('include_low_coverage_session',bm,vm)]:
            source=all_table if name=='include_low_coverage_session' else table
            sub=source[source[xcol].notna()&source[ycol].notna()]
            if name=='exclude_reused_stim_partners':
                sub=sub[~sub.animal_id.isin(REUSED_STIM_RESIDENTS)]
            rho=np.nan
            if name=='adjust_motion_and_distance':
                sub=sub.dropna(subset=['mean_resident_speed_px_s','median_distance_px'])
                if len(sub)>=MIN_N:
                    rx=ranks_residual(sub[xcol].to_numpy(float),sub.stratum.to_numpy(str))
                    ry=ranks_residual(sub[ycol].to_numpy(float),sub.stratum.to_numpy(str))
                    cov=pd.get_dummies(sub.stratum,dtype=float).to_numpy()
                    cov=np.column_stack([cov,rankdata(sub.mean_resident_speed_px_s),rankdata(sub.median_distance_px)])
                    rx-=cov@np.linalg.lstsq(cov,rx,rcond=None)[0]
                    ry-=cov@np.linalg.lstsq(cov,ry,rcond=None)[0]
                    den=np.linalg.norm(rx)*np.linalg.norm(ry)
                    rho=float(rx@ry/den) if den>1e-12 else np.nan
            elif len(sub)>=MIN_N:
                rho=coefficient(sub[xcol].to_numpy(float),sub[ycol].to_numpy(float),sub.stratum.to_numpy(str))
            rows.append(dict(behavior=test.behavior,behavior_metric=bm,vocal_metric=vm,sensitivity=name,
                             n=len(sub),rho=rho,primary_rho=test.rho,
                             delta_rho=rho-test.rho,primary_q=test.q_bh))
    return pd.DataFrame(rows)


def self_check() -> None:
    """Check segmentation, censoring, undefined values and rank implementation."""
    active=np.array([1,1,0,1,0,0,1],bool);obs=np.ones(7,bool);dt=np.full(7,.1)
    strict=episodes(active,obs,dt,0);merged=episodes(active,obs,dt,.1)
    assert strict['episode_count']==3 and merged['episode_count']==2
    assert np.isclose(strict['cumulative_duration_s'],.4)
    obs[2]=False
    assert episodes(active,obs,dt,.1)['episode_count']==3
    absent=episodes(np.zeros(7,bool),obs,dt,0)
    assert np.isnan(absent['mean_episode_duration_s']) and absent['episode_count']==0
    x=np.array([0,1,1,4,5,6.])
    from scipy.stats import spearmanr
    assert np.isclose(coefficient(x,x[::-1]),spearmanr(x,x[::-1]).statistic)
    g=np.array(['a']*3+['b']*3)
    assert np.allclose([ranks_residual(x,g)[g==k].mean() for k in ['a','b']],0)
    # A strong between-group offset disappears when there is no within-group variation.
    assert np.isnan(coefficient(np.array([0,0,0,1,1,1.]),x,g))
    first=int(np.floor(WINDOW[0]*FPS))
    valid=np.array([True,False,True])
    clipped=observed_call_duration(first/FPS+.001,(first+3)/FPS-.001,first,valid)
    assert np.isclose(clipped,2/FPS-.002)
    calls=pd.DataFrame({'start(s)':[300.01,300.05],'end(s)':[300.015,300.055],
                        'observation_segment':[0,1],'observed_duration_s':[.005,.005],
                        'duration(ms)':[5.,5.],'avg_freq':[75000.,75000.],
                        'bandwidth':[10000.,10000.],'class_top1':['short','short']})
    assert vocal_summary(calls,1.,.25)['bout_count']==2


def main() -> None:
    """Run analysis once, then allow style-only regeneration from a verified cache."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--recompute',action='store_true')
    modes.add_argument('--figures-only',action='store_true')
    args=parser.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    self_check()
    fingerprint=source_fingerprint()
    manifest=out/'provenance.json'
    names=['behavior_parameters','vocal_parameters','source_qa','correlation_statistics','sensitivity_statistics']
    valid=manifest.exists() and json.loads(manifest.read_text())['fingerprint']==json.loads(json.dumps(fingerprint))
    valid=valid and all((out/f'{name}.csv').exists() for name in names)
    if args.figures_only and not valid:
        raise RuntimeError('No valid numerical cache; run without --figures-only first.')
    if valid and not args.recompute:
        data=[pd.read_csv(out/f'{name}.csv',dtype={'animal_id':str}) for name in names]
        print('Validated numerical cache reused.',flush=True)
    else:
        with threadpool_limits(limits=1):
            behavior,vocal,qa=extract_tables()
            results=analyze(behavior,vocal)
            robustness=sensitivity(behavior,vocal,results)
        data=[behavior,vocal,qa,results,robustness]
        for name,table in zip(names,data):
            table.to_csv(out/f'{name}.csv',index=False)
        manifest.write_text(json.dumps({'fingerprint':fingerprint,
            'unit':'one independent experimental recording / resident-partner dyad',
            'window_s':WINDOW,'primary_family_size':len(results[results.primary]),
            'permutation_null':'conditional independence within genotype x virus strata',
            'mean_episode_definition':'mean positive-frame duration per episode; missing frames break episodes',
            'vocal_assignment':'all dyad calls; no geometrically inferred speaker assignment',
            'limitations':['24 recordings, 6 per genotype x virus stratum',
                'Two known stimulus partners are reused; exclusion sensitivity supplied',
                'Behavior labels are geometric rules, not independent manually scored observations',
                'Acoustic features/entropy are conditional on detected calls and unstable at low counts',
                'Pointwise bootstrap intervals are exploratory, not simultaneous'],
            'references':['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html',
                'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html',
                'https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html']},indent=2),encoding='utf-8')
    from social_vocal_correlation_figures import make_outputs
    make_outputs(out,*data)
    primary=data[3][data[3].primary]
    print(primary.sort_values('q_bh')[['behavior','behavior_metric','vocal_metric','n','rho','p','q_bh']].head(12).to_string(index=False),flush=True)
    print(f'Outputs: {out}',flush=True)


if __name__=='__main__':
    main()
