"""Estimate dyad-call probabilities during and around social behavior episodes.

The observational unit for inference is a recording, never an individual call
or episode. Null traces are uniformly shifted in contiguous observed segments
inside local time blocks. Raw data and existing analyses remain immutable.
Run in an IDE, from the adjacent .cmd file, or with --figures-only after caching.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft,irfft
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

import social_vocal_correlations as previous

# Numerical configuration, fixed before inspecting probabilities.
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'lgdel_usv_analysis'/'social_call_probability'
WINDOW=(300.,900.)
FPS=previous.FPS
BIN_S=.1
MIN_COVERAGE=.95
BEHAVIOR_BIN_FRACTION=.5
EPISODE_GAP_S=.2
MIN_EPISODE_ACTIVE_S=.2
NEAR_HALF_S=1.
PERI_HALF_S=5.
BLOCK_S=60.
BLOCK_SENSITIVITY=(30.,120.)
N_SHIFT=19999
N_BOOT=4000
SEED=20261008
MIN_STATE_TIME_S=5.
MIN_SESSIONS=5
MIN_EVENTS=20
ALPHA=.05
CACHE_VERSION=1
SOCIAL=previous.SOCIAL
MUTUAL=previous.MUTUAL
DEFINITIONS=([(b,'dyad') for b in SOCIAL]+
             [(b,role) for role in ['resident','partner'] for b in SOCIAL if b not in MUTUAL])
METRICS=['during_100ms','around_1s','before_1s','after_1s']


def fingerprint() -> dict:
    """Keep source stamps and numerical settings separate from plotting code."""
    paths=[previous.TRIAL_PLAN,previous.KINEMATICS]
    for animal in previous.metadata.GENOTYPE_MAP:paths.extend(previous.source_paths(animal))
    functions=[frame_episodes,build_sessions,segments,window_hit,xcorr,
               shift_statistics,aggregate,bootstrap_mean,analyse,sensitivities,self_check]
    return dict(version=CACHE_VERSION,
        sources={str(p):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths},
        numerical_code=hashlib.sha256('\n'.join(inspect.getsource(f) for f in functions).encode()).hexdigest(),
        previous_code=hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest(),
        metadata_code=hashlib.sha256(Path(previous.metadata.__file__).read_bytes()).hexdigest(),
        genotype_map=previous.metadata.GENOTYPE_MAP,stim_map=previous.metadata.STIM_MAP,
        window=WINDOW,fps=FPS,bin_s=BIN_S,coverage=MIN_COVERAGE,
        behavior_fraction=BEHAVIOR_BIN_FRACTION,gap_s=EPISODE_GAP_S,min_active_s=MIN_EPISODE_ACTIVE_S,
        near_s=NEAR_HALF_S,peri_s=PERI_HALF_S,block_s=BLOCK_S,block_sensitivity=BLOCK_SENSITIVITY,
        n_shift=N_SHIFT,n_boot=N_BOOT,seed=SEED,min_state_s=MIN_STATE_TIME_S,
        min_sessions=MIN_SESSIONS,min_events=MIN_EVENTS,alpha=ALPHA,definitions=DEFINITIONS)


def frame_episodes(active:np.ndarray,valid:np.ndarray,dt:np.ndarray,gap_s:float):
    """Return observed, noncensored onsets after merging short observed gaps.

    Require the previous frame to be observed and inactive. Minimum episode
    duration sums positive frames; inactive bridged gaps never add active time.
    """
    indices=np.flatnonzero(active&valid)
    if not len(indices):return np.array([],int)
    breaks=np.diff(indices)>1
    for i in np.flatnonzero(breaks):
        between=slice(indices[i]+1,indices[i+1])
        if valid[between].all() and dt[between].sum()<=gap_s+1e-9:breaks[i]=False
    groups=np.split(indices,np.flatnonzero(breaks)+1)
    onsets=[]
    for group in groups:
        first=group[0]
        if first>0 and valid[first-1] and not active[first-1] and dt[group].sum()>=MIN_EPISODE_ACTIVE_S-1e-9:
            onsets.append(first)
    return np.asarray(onsets,int)


def build_sessions(out:Path):
    """Build exact occupied-time rates and common-coverage 100 ms trace caches."""
    plan=pd.read_excel(previous.TRIAL_PLAN)
    plan['animal_id']=plan['Animal ID'].astype(str)
    plan=plan.set_index('animal_id')
    kin=pd.read_csv(previous.KINEMATICS,usecols=['animal_id','t','speed_m1','dist'],dtype={'animal_id':str})
    kin=kin[(kin.t>=WINDOW[0])&(kin.t<WINDOW[1])]
    n_bins=int(round((WINDOW[1]-WINDOW[0])/BIN_S))
    first=int(np.floor(WINDOW[0]*FPS));last=int(np.ceil(WINDOW[1]*FPS))
    frames=np.arange(first,last)
    dt=np.maximum(0,np.minimum((frames+1)/FPS,WINDOW[1])-np.maximum(frames/FPS,WINDOW[0]))
    # Integrate frame intervals at exact bin boundaries, including partial frames.
    bin_start=WINDOW[0]+np.arange(n_bins)*BIN_S
    bin_end=bin_start+BIN_S
    bin_frames=np.floor(bin_start*FPS).astype(int)[:,None]+np.arange(int(np.ceil(BIN_S*FPS))+1)
    frame_index=np.clip(bin_frames-first,0,len(frames)-1)
    frame_weights=np.maximum(0,np.minimum((bin_frames+1)/FPS,bin_end[:,None])-
                             np.maximum(bin_frames/FPS,bin_start[:,None]))
    sessions=[];raw_rows=[];qa_rows=[];cache={}
    for i,(animal,genotype) in enumerate(previous.metadata.GENOTYPE_MAP.items(),1):
        bp,cp=previous.source_paths(animal)
        cols=[f'behavior_{b}' for b in SOCIAL]
        raw=pd.read_csv(bp,usecols=['frame_id','mouse_id']+cols)
        raw=raw[raw.frame_id.between(first,last-1)]
        if not raw[cols].isin([0,1]).all().all():raise ValueError(f'Invalid binary flags: {animal}')
        observed={};grid={}
        for actor in [1,2]:
            g=raw[raw.mouse_id==actor].groupby('frame_id')[cols].max()
            observed[actor]=np.isin(frames,g.index)
            grid[actor]=g.reindex(frames).fillna(0).to_numpy(bool)
        valid_frame=observed[1]&observed[2]
        coverage=float(dt[valid_frame].sum()/(WINDOW[1]-WINDOW[0]))
        exposure=(frame_weights*valid_frame[frame_index]).sum(axis=1)
        valid_bin=~((frame_weights>1e-9)&~valid_frame[frame_index]).any(axis=1)
        flags=np.zeros((len(DEFINITIONS),n_bins),bool)
        onsets=np.zeros_like(flags)
        onsets_strict=np.zeros_like(flags)
        calls=pd.read_csv(cp)
        calls=calls[(calls['start(s)']>=WINDOW[0])&(calls['start(s)']<WINDOW[1])]
        cbins=np.floor((calls['start(s)'].to_numpy(float)-WINDOW[0])/BIN_S).astype(int)
        cframes=np.floor(calls['start(s)'].to_numpy(float)*FPS).astype(int)-first
        counts=np.bincount(cbins[valid_bin[cbins]],minlength=n_bins)
        for j,(behavior,role) in enumerate(DEFINITIONS):
            col=SOCIAL.index(behavior)
            active=(grid[1][:,col]|grid[2][:,col]) if role=='dyad' else grid[1 if role=='resident' else 2][:,col]
            occupied=(frame_weights*(active&valid_frame)[frame_index]).sum(axis=1)
            flags[j]=valid_bin&(occupied/np.maximum(exposure,1e-12)>=BEHAVIOR_BIN_FRACTION)
            for gap,target in [(EPISODE_GAP_S,onsets),(0.,onsets_strict)]:
                start_frames=frame_episodes(active,valid_frame,dt,gap)
                start_bins=np.floor((frames[start_frames]/FPS-WINDOW[0])/BIN_S).astype(int)
                start_bins=start_bins[(start_bins>=0)&(start_bins<n_bins)]
                target[j,np.unique(start_bins)]=True
            time=float(dt[active&valid_frame].sum())
            inside=int((active[cframes]&valid_frame[cframes]).sum())
            raw_rows.append(dict(animal_id=animal,genotype=genotype,virus=str(plan.loc[animal,'virus']),
                behavior=behavior,role=role,occupied_time_s=time,calls_inside_exact=inside,
                exact_call_rate_per_min=inside/(time/60) if time>0 else np.nan,
                valid_behavior_bins=int(flags[j].sum()),candidate_onsets=int(onsets[j].sum()),
                exposure_s=float(dt[valid_frame].sum()),coverage=coverage,eligible=coverage>=MIN_COVERAGE))
        k=kin[kin.animal_id==animal]
        speed=np.interp(WINDOW[0]+(np.arange(n_bins)+.5)*BIN_S,k.t,k.speed_m1)
        distance=np.interp(WINDOW[0]+(np.arange(n_bins)+.5)*BIN_S,k.t,k.dist)
        if str(plan.loc[animal,'genotype_pyrat']).upper()!=genotype:raise ValueError('Genotype disagreement')
        session=dict(animal_id=animal,genotype=genotype,virus=str(plan.loc[animal,'virus']),
                     counts=counts,valid=valid_bin,flags=flags,onsets=onsets,onsets_strict=onsets_strict,
                     speed=speed,distance=distance,eligible=coverage>=MIN_COVERAGE)
        sessions.append(session)
        qa_rows.append(dict(animal_id=animal,genotype=genotype,virus=session['virus'],coverage=coverage,
            eligible=session['eligible'],total_calls=len(calls),calls_in_complete_bins=int(counts.sum()),
            valid_bins=int(valid_bin.sum()),valid_bin_time_s=float(BIN_S*valid_bin.sum()),
            duplicate_frame_actor_rows=int(raw.duplicated(['frame_id','mouse_id']).sum())))
        for name in ['counts','valid','flags','onsets','onsets_strict','speed','distance']:cache[f'{animal}_{name}']=session[name]
        print(f'[{i}/24] {animal}: complete bins {valid_bin.mean():.2%}, calls {counts.sum()}/{len(calls)}',flush=True)
    np.savez_compressed(out/'traces.npz',**cache)
    return sessions,pd.DataFrame(raw_rows),pd.DataFrame(qa_rows)


def segments(valid:np.ndarray,block_s:float):
    """Intersect local time blocks with uninterrupted fully observed bins."""
    block_n=int(round(block_s/BIN_S))
    out=[]
    for start in range(0,len(valid),block_n):
        stop=min(start+block_n,len(valid))
        edges=np.diff(np.r_[False,valid[start:stop],False].astype(int))
        for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            out.append((start+a,start+b))
    return out


def window_hit(binary:np.ndarray,first_lag:int,last_lag:int) -> np.ndarray:
    """Circular any-call windows for all candidate alignments in one segment."""
    hit=np.zeros(len(binary),bool)
    for lag in range(first_lag,last_lag):hit|=np.roll(binary,-lag)
    return hit


def xcorr(a:np.ndarray,b:np.ndarray) -> np.ndarray:
    """Exact circular counts: out[j,s] = sum_i a[j,i]*b[(i+s) mod n]."""
    n=a.shape[-1]
    return np.maximum(0,np.rint(irfft(np.conj(rfft(a,axis=-1))*rfft(b),n=n,axis=-1)))


def shift_statistics(session:dict,block_s:float,n_shift:int,onset_key='onsets',peri=False):
    """Compute conditional probabilities and sampled local-shift nulls efficiently.

    Eligible episode windows lie wholly inside an observed segment and block.
    Cross-correlations enumerate all legal offsets in each segment exactly.
    Random offsets include zero and are shared across all behavior tests.
    """
    jn=len(DEFINITIONS);h=int(round(NEAR_HALF_S/BIN_S));ph=int(round(PERI_HALF_S/BIN_S))
    numerator=np.zeros((jn,4));denominator=np.zeros_like(numerator)
    baseline=np.zeros_like(numerator)
    null=np.zeros((n_shift,jn,4))
    rng=np.random.default_rng(SEED+int(session['animal_id'])+int(block_s*100))
    curve=np.zeros((jn,2*ph));curve_base=np.zeros_like(curve);curve_events=np.zeros(jn,int)
    for a,b in segments(session['valid'],block_s):
        binary=(session['counts'][a:b]>0).astype(float)
        length=b-a
        offsets=rng.integers(0,length,size=n_shift) if n_shift else np.array([],int)
        state=session['flags'][:,a:b].astype(float)
        onset=session[onset_key][:,a:b].copy()
        onset[:,:min(h,length)]=False;onset[:,max(length-h+1,0):]=False
        tables=[xcorr(state,binary)]
        for left,right in [(-h,h),(-h,0),(0,h)]:
            hit=window_hit(binary.astype(bool),left,right)
            tables.append(xcorr(onset.astype(float),hit.astype(float)))
        denominators=[state.sum(axis=1)]+[onset.sum(axis=1)]*3
        for m,(tab,den) in enumerate(zip(tables,denominators)):
            numerator[:,m]+=tab[:,0];denominator[:,m]+=den
            baseline[:,m]+=tab.mean(axis=1)
            if n_shift:null[:,:,m]+=tab[:,offsets].T
        if peri:
            full=session[onset_key][:,a:b].copy()
            full[:,:min(ph,length)]=False;full[:,max(length-ph+1,0):]=False
            corr=xcorr(full.astype(float),binary)
            curve+=corr[:,np.arange(-ph,ph)%length]
            n_events=full.sum(axis=1)
            curve_base+=n_events[:,None]*binary.mean()
            curve_events+=n_events.astype(int)
    observed=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)
    expected=np.divide(baseline,denominator,out=np.full_like(baseline,np.nan),where=denominator>0)
    null=np.divide(null,denominator[None,:,:],out=np.full_like(null,np.nan),where=denominator[None,:,:]>0)
    return dict(observed=observed,expected=expected,null=null,denominator=denominator,numerator=numerator,
                curve=np.divide(curve,curve_events[:,None],out=np.full_like(curve,np.nan),where=curve_events[:,None]>0),
                curve_expected=np.divide(curve_base,curve_events[:,None],out=np.full_like(curve,np.nan),where=curve_events[:,None]>0),
                curve_events=curve_events)


def aggregate(values:np.ndarray,genotypes:np.ndarray,scope:str) -> np.ndarray:
    """Equal-session means; combined estimate standardizes WT and HET to 50/50."""
    if scope in ['WT','HET']:return np.mean(values[...,genotypes==scope],axis=-1)
    return .5*np.mean(values[...,genotypes=='WT'],axis=-1)+.5*np.mean(values[...,genotypes=='HET'],axis=-1)


def bootstrap_mean(observed:np.ndarray,expected:np.ndarray,genotypes:np.ndarray,scope:str,seed:int):
    """Resample recordings together, stratified by genotype, keeping events clustered."""
    rng=np.random.default_rng(seed)
    n=len(observed)
    indices=np.tile(np.arange(n),(N_BOOT,1))
    for genotype in np.unique(genotypes):
        pos=np.flatnonzero(genotypes==genotype)
        indices[:,pos]=rng.choice(pos,size=(N_BOOT,len(pos)),replace=True)
    boot=aggregate(observed[indices],genotypes,scope)
    delta=aggregate((observed-expected)[indices],genotypes,scope)
    return (*np.quantile(boot,[.025,.975]),*np.quantile(delta,[.025,.975]))


def analyse(sessions:list,raw:pd.DataFrame,out:Path):
    """Test all behaviors and keep pooled/genotype/actor families explicit."""
    supported=[s for s in sessions if s['eligible']]
    summaries=[];nulls=[];per_session=[];curves=[]
    for session in supported:
        stat=shift_statistics(session,BLOCK_S,N_SHIFT,peri=True)
        summaries.append(stat);nulls.append(stat['null'])
        for j,(behavior,role) in enumerate(DEFINITIONS):
            for m,metric in enumerate(METRICS):
                denominator=stat['denominator'][j,m]
                per_session.append(dict(animal_id=session['animal_id'],genotype=session['genotype'],
                    virus=session['virus'],behavior=behavior,role=role,metric=metric,
                    observed_probability=stat['observed'][j,m],shift_probability=stat['expected'][j,m],
                    denominator=int(denominator),numerator=int(stat['numerator'][j,m]),
                    eligible_metric=(denominator*BIN_S>=MIN_STATE_TIME_S) if m==0 else denominator>0))
            for k in range(int(round(2*PERI_HALF_S/BIN_S))):
                curves.append(dict(animal_id=session['animal_id'],genotype=session['genotype'],behavior=behavior,role=role,
                    lag_s=-PERI_HALF_S+(k+.5)*BIN_S,probability=stat['curve'][j,k],
                    shift_probability=stat['curve_expected'][j,k],events=int(stat['curve_events'][j])))
        print(f'Local shift distributions: {session["animal_id"]}',flush=True)
    ps=pd.DataFrame(per_session)
    rows=[]
    genotype_all=np.array([s['genotype'] for s in supported])
    nulls=np.stack(nulls,axis=-1) # shifts x behavior x metric x session
    for j,(behavior,role) in enumerate(DEFINITIONS):
        for m,metric in enumerate(METRICS):
            subset=ps[(ps.behavior==behavior)&(ps.role==role)&(ps.metric==metric)]
            valid=subset.eligible_metric.to_numpy(bool)
            for scope in ['combined','WT','HET']:
                use=valid.copy()
                if scope!='combined':use&=genotype_all==scope
                selected=subset.iloc[np.flatnonzero(use)]
                g=genotype_all[use]
                n=len(selected);nw=int((g=='WT').sum());nh=int((g=='HET').sum())
                support=n>=MIN_SESSIONS and (m==0 or selected.denominator.sum()>=MIN_EVENTS)
                if scope=='combined':support&=min(nw,nh)>=2
                obs=expected=lo=hi=dlo=dhi=p=np.nan
                status='insufficient_support'
                if n and (scope!='combined' or min(nw,nh)>0):
                    obs=float(aggregate(selected.observed_probability.to_numpy(),g,scope))
                    expected=float(aggregate(selected.shift_probability.to_numpy(),g,scope))
                    seed=SEED+j*101+m*13+['combined','WT','HET'].index(scope)
                    lo,hi,dlo,dhi=bootstrap_mean(selected.observed_probability.to_numpy(),selected.shift_probability.to_numpy(),g,scope,seed)
                if support:
                    null=aggregate(nulls[:,j,m,use],g,scope)
                    p=float((1+(np.abs(null-expected)>=abs(obs-expected)-1e-12).sum())/(N_SHIFT+1))
                    status='ok'
                rate_table=raw[(raw.behavior==behavior)&(raw.role==role)&raw.eligible&(raw.occupied_time_s>=MIN_STATE_TIME_S)]
                if scope!='combined':rate_table=rate_table[rate_table.genotype==scope]
                rate=np.nan
                if len(rate_table) and (scope!='combined' or rate_table.genotype.nunique()==2):
                    rate=float(aggregate(rate_table.exact_call_rate_per_min.to_numpy(float),rate_table.genotype.to_numpy(),scope))
                rows.append(dict(behavior=behavior,role=role,metric=metric,scope=scope,n_sessions=n,n_WT=nw,n_HET=nh,
                    n_events_or_bins=int(selected.denominator.sum()),observed_probability=obs,
                    ci_low=lo,ci_high=hi,shift_probability=expected,difference=obs-expected,
                    difference_ci_low=dlo,difference_ci_high=dhi,
                    fold_probability=obs/expected if expected>0 else np.nan,p=p,
                    p_monte_carlo_se=float(np.sqrt(p*(1-p)/(N_SHIFT+1))) if np.isfinite(p) else np.nan,
                    exact_call_rate_per_min=rate,status=status))
    result=pd.DataFrame(rows)
    result['primary']=(result.role=='dyad')&(result.scope=='combined')&result.metric.isin(METRICS[:2])
    result['family']=np.where(result.primary,'primary_26',
        np.where((result.role=='dyad')&result.metric.isin(METRICS[:2]),'genotype_secondary_52',
        np.where(result.role=='dyad','before_after_secondary_78','actor_secondary_216')))
    for column,method in [('q_bh','fdr_bh'),('q_by','fdr_by'),('p_holm','holm')]:
        result[column]=np.nan
        for family,idx in result.groupby('family').groups.items():
            result.loc[idx,column]=multipletests(result.loc[idx,'p'].fillna(1),method=method)[1]
    # Actual actor family contains 9 behaviors x2actors x4metrics x3scopes=216.
    result['significant_bh']=result.q_bh<ALPHA
    result.to_csv(out/'probability_statistics.csv',index=False)
    ps.to_csv(out/'session_probabilities.csv',index=False)
    curve_table=pd.DataFrame(curves)
    curve_table.to_csv(out/'session_peri_onset_curves.csv',index=False)
    return result,ps,curve_table


def sensitivities(sessions:list,result:pd.DataFrame,out:Path):
    """Compare fixed local-block and episode-definition alternatives descriptively."""
    rows=[]
    settings=[('block_30s',30.,'onsets'),('block_120s',120.,'onsets'),
              ('strict_episodes',BLOCK_S,'onsets_strict'),('include_low_coverage',BLOCK_S,'onsets'),
              ('exclude_reused_stim',BLOCK_S,'onsets')]
    for label,block,key in settings:
        ss=[s for s in sessions if s['eligible'] or label=='include_low_coverage']
        if label=='exclude_reused_stim':ss=[s for s in ss if s['animal_id'] not in previous.REUSED_STIM_RESIDENTS]
        stats=[shift_statistics(s,block,0,onset_key=key) for s in ss]
        genotypes=np.array([s['genotype'] for s in ss])
        for j,(behavior,role) in enumerate(DEFINITIONS):
            if role!='dyad':continue
            for m,metric in enumerate(METRICS[:2]):
                obs=np.array([x['observed'][j,m] for x in stats])
                exp=np.array([x['expected'][j,m] for x in stats])
                den=np.array([x['denominator'][j,m] for x in stats])
                valid=(den*BIN_S>=MIN_STATE_TIME_S) if m==0 else den>0
                g=genotypes[valid]
                observed=expected=np.nan
                if len(g) and len(np.unique(g))==2:
                    observed=float(aggregate(obs[valid],g,'combined'))
                    expected=float(aggregate(exp[valid],g,'combined'))
                reference=result[result.primary&(result.behavior==behavior)&(result.metric==metric)].iloc[0]
                rows.append(dict(sensitivity=label,behavior=behavior,metric=metric,n_sessions=int(valid.sum()),
                    denominator=int(den[valid].sum()),observed_probability=observed,shift_probability=expected,
                    difference=observed-expected,primary_difference=reference.difference))
    pd.DataFrame(rows).to_csv(out/'sensitivity_statistics.csv',index=False)


def self_check():
    """Verify orientation, boundary handling and probabilities with synthetic traces."""
    a=np.array([[1,0,0,0.]])
    b=np.array([0,1,0,0.])
    assert np.allclose(xcorr(a,b),[[0,1,0,0]])
    binary=np.array([0,0,1,0,0],bool)
    assert window_hit(binary,-1,1).tolist()==[False,False,True,True,False]
    valid=np.array([1,1,0,1,1,1],bool)
    assert segments(valid,60)==[(0,2),(3,6)]
    active=np.array([0,1,1,0,1,1,0],bool);observed=np.ones(7,bool);dt=np.full(7,.1)
    assert frame_episodes(active,observed,dt,.2).tolist()==[1]
    observed[0]=False
    assert frame_episodes(active,observed,dt,.2).size==0
    # Fixed groups standardized equally even if session counts differ.
    assert aggregate(np.array([0.,0.,1.]),np.array(['WT','WT','HET']),'combined')==.5


def block_sensitivity_tests(out:Path,qa:pd.DataFrame) -> pd.DataFrame:
    """Check inferential robustness to the two predeclared local time scales.

    This secondary cache has its own fingerprint, so adding robustness checks
    never recomputes already validated primary probabilities or bootstrap CIs.
    All 52 combined tests form one separate multiplicity family.
    """
    path=out/'block_sensitivity_tests.csv';meta=out/'block_sensitivity_provenance.json'
    primary=json.loads((out/'provenance.json').read_text())['fingerprint']
    fp=dict(primary=primary,code=hashlib.sha256(inspect.getsource(block_sensitivity_tests).encode()).hexdigest(),
            blocks=BLOCK_SENSITIVITY,n_shift=N_SHIFT,seed=SEED)
    fp=json.loads(json.dumps(fp))
    if path.exists() and meta.exists() and json.loads(meta.read_text())==fp:
        print('Validated temporal-block sensitivity cache reused.',flush=True)
        return pd.read_csv(path)
    arrays=np.load(out/'traces.npz')
    labels=qa[qa.eligible]
    rows=[]
    with threadpool_limits(limits=1):
        for block in BLOCK_SENSITIVITY:
            observed=[];expected=[];denominators=[];nulls=[]
            for r in labels.itertuples():
                session={'animal_id':str(r.animal_id)}
                for name in ['counts','valid','flags','onsets']:session[name]=arrays[f'{r.animal_id}_{name}']
                stat=shift_statistics(session,block,N_SHIFT)
                observed.append(stat['observed'][:len(SOCIAL),:2])
                expected.append(stat['expected'][:len(SOCIAL),:2])
                denominators.append(stat['denominator'][:len(SOCIAL),:2])
                nulls.append(stat['null'][:,:len(SOCIAL),:2].copy())
            obs,exp,den=np.stack(observed),np.stack(expected),np.stack(denominators)
            null=np.stack(nulls,axis=-1)
            genotypes=labels.genotype.to_numpy()
            for j,behavior in enumerate(SOCIAL):
                for m,metric in enumerate(METRICS[:2]):
                    use=(den[:,j,m]*BIN_S>=MIN_STATE_TIME_S) if m==0 else den[:,j,m]>0
                    g=genotypes[use]
                    mean=baseline=p=np.nan
                    support=int(use.sum())>=MIN_SESSIONS and (m==0 or den[use,j,m].sum()>=MIN_EVENTS)
                    support&=min(int((g=='WT').sum()),int((g=='HET').sum()))>=2
                    if len(np.unique(g))==2:
                        mean=float(aggregate(obs[use,j,m],g,'combined'))
                        baseline=float(aggregate(exp[use,j,m],g,'combined'))
                        if support:
                            values=aggregate(null[:,j,m,use],g,'combined')
                            p=float((1+(np.abs(values-baseline)>=abs(mean-baseline)-1e-12).sum())/(N_SHIFT+1))
                    rows.append(dict(block_s=block,behavior=behavior,metric=metric,n_sessions=int(use.sum()),
                        n_events_or_bins=int(den[use,j,m].sum()),observed_probability=mean,
                        shift_probability=baseline,difference=mean-baseline,p=p,status='ok' if support else 'insufficient_support'))
            print(f'Temporal-block robustness: {block:g}s',flush=True)
    result=pd.DataFrame(rows)
    for method,col in [('fdr_bh','q_bh'),('fdr_by','q_by'),('holm','p_holm')]:
        result[col]=multipletests(result.p.fillna(1),method=method)[1]
    result.to_csv(path,index=False);meta.write_text(json.dumps(fp,indent=2),encoding='utf-8')
    return result


def main():
    """Cache numerical results before plotting and expose reproducible rerun modes."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--recompute',action='store_true');modes.add_argument('--figures-only',action='store_true')
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    self_check();fp=json.loads(json.dumps(fingerprint()))
    provenance=out/'provenance.json'
    names=['probability_statistics','session_probabilities','session_peri_onset_curves','raw_state_rates','source_qa','sensitivity_statistics']
    valid=provenance.exists() and json.loads(provenance.read_text())['fingerprint']==fp and all((out/f'{n}.csv').exists() for n in names)
    if args.figures_only and not valid:raise RuntimeError('No valid numeric cache; run without --figures-only first.')
    if valid and not args.recompute:print('Validated numerical cache reused.',flush=True)
    else:
        with threadpool_limits(limits=1):
            sessions,raw,qa=build_sessions(out)
            raw.to_csv(out/'raw_state_rates.csv',index=False);qa.to_csv(out/'source_qa.csv',index=False)
            result,ps,curves=analyse(sessions,raw,out)
            sensitivities(sessions,result,out)
        provenance.write_text(json.dumps({'fingerprint':fp,'observational_unit':'recording / dyad',
            'probability_bin_definition':'behavior occupies >=50% of observed frames assigned to a complete100ms bin',
            'probability_event_definition':'at least one call onset in[-1,+1)s of a100ms-quantized episode onset',
            'session_weighting':'equal animals within genotype, combined50%WT+50%HET',
            'null':'uniform independent cyclic shifts inside contiguous complete coverage and local60s blocks, including offsetzero',
            'limitations':['Local stationarity within blocks assumed; shift structure broken at block/coverage boundaries',
                'Behaviors overlap; the same call can associate with multiple behaviors',
                'Behavior actor is not identified acoustic caller',
                'Reused stimulus partners can induce dependence between recordings',
                '100ms quantization and rule-based behavioral labels'],
            'references':['https://pmc.ncbi.nlm.nih.gov/articles/PMC2972681/',
                'https://pmc.ncbi.nlm.nih.gov/articles/PMC3090783/',
                'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html']},indent=2),encoding='utf-8')
    data=[pd.read_csv(out/f'{name}.csv',dtype={'animal_id':str}) for name in names]
    from social_call_probability_figures import make_outputs
    make_outputs(out,*data)
    print(data[0][data[0].primary&(data[0].metric=='around_1s')].sort_values('observed_probability',ascending=False)[
        ['behavior','n_sessions','n_events_or_bins','observed_probability','shift_probability','difference','q_bh']].to_string(index=False),flush=True)


if __name__=='__main__':main()
