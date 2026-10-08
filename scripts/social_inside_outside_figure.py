"""Reproduce a six-panel WT/HET boxstrip from validated social-call traces.

Each point is one recording. The complementary inside/outside states use the
same completely observed 100 ms bins, so all three metrics share a denominator.
Raw files and the upstream cache are never modified by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import social_call_probability as source

# Analysis configuration. Change the behavior list here to redefine social.
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'lgdel_usv_analysis'/'social_call_probability'
OUTPUT=ROOT/'lgdel_usv_analysis'/'social_inside_outside'
BEHAVIORS=list(source.SOCIAL)
MIN_EXPOSURE_S=5.
N_PERMUTATIONS=49999
N_BOOTSTRAP=5000
SEED=20261008
ALPHA=.05
METRICS=['probability_pct','frequency_per_min','call_count']
STATES=['Inside','Outside']
GROUPS=['WT','HET']

# Figure configuration. Rendering edits reuse the numerical cache.
COLORS={'WT':'#327A9E','HET':'#E68A32'}
LABELS={'WT':'WT','HET':'Het'}
FIGSIZE=(10.8,7.2)
DPI=400
BOX_WIDTH=.36
POINT_SIZE=30
JITTER=.085
TITLES=['Call probability','Call frequency','Number of calls']
YLABELS=['Bins with ≥1 call (%)','Calls / min','Calls']
FORMATS=['png','pdf','svg']
FONT='DejaVu Sans'


def fingerprint() -> dict:
    """Verify upstream sources, then fingerprint only our numerical choices."""
    upstream=json.loads((INPUT/'provenance.json').read_text(encoding='utf-8'))
    current=json.loads(json.dumps(source.fingerprint()))
    if upstream['fingerprint']!=current:
        raise RuntimeError('Upstream cache is stale. Run social_call_probability.py first.')
    return dict(version=1,upstream=current,
        input_hashes={name:hashlib.sha256((INPUT/name).read_bytes()).hexdigest()
                      for name in ['traces.npz','source_qa.csv']},
        numerical_code=hashlib.sha256('\n'.join(inspect.getsource(f) for f in
            [extract,statistics]).encode()).hexdigest(),behaviors=BEHAVIORS,
        exposure=MIN_EXPOSURE_S,permutations=N_PERMUTATIONS,bootstrap=N_BOOTSTRAP,
        seed=SEED,metrics=METRICS,states=STATES,groups=GROUPS,alpha=ALPHA)


def extract() -> tuple[pd.DataFrame,pd.DataFrame]:
    """Partition every valid bin once, preserving calls, exposure and eligibility.

    Social means at least one specified dyad flag is active. Upstream flags
    require that particular behavior to occupy at least 50% of the 100 ms bin.
    The outside state is the complement, not missing tracking or solitude.
    """
    qa=pd.read_csv(INPUT/'source_qa.csv',dtype={'animal_id':str})
    traces=np.load(INPUT/'traces.npz')
    indices=[source.DEFINITIONS.index((behavior,'dyad')) for behavior in BEHAVIORS]
    rows=[];checks=[]
    for r in qa.itertuples():
        valid=traces[f'{r.animal_id}_valid']
        counts=traces[f'{r.animal_id}_counts']
        social=traces[f'{r.animal_id}_flags'][indices].any(axis=0)&valid
        masks={'Inside':social,'Outside':valid&~social}
        for state,mask in masks.items():
            n_bins=int(mask.sum());duration=n_bins*source.BIN_S
            calls=int(counts[mask].sum());hit=int((counts[mask]>0).sum())
            rows.append(dict(animal_id=r.animal_id,genotype=r.genotype,virus=r.virus,
                state=state,eligible=bool(r.eligible),coverage=r.coverage,
                exposure_s=duration,n_bins=n_bins,bins_with_call=hit,
                probability_pct=100*hit/n_bins if duration>=MIN_EXPOSURE_S else np.nan,
                frequency_per_min=60*calls/duration if duration>=MIN_EXPOSURE_S else np.nan,
                call_count=calls if duration>=MIN_EXPOSURE_S else np.nan))
        # Exhaustive, disjoint partition provides a meaningful conservation check.
        assert not (masks['Inside']&masks['Outside']).any()
        assert np.array_equal(masks['Inside']|masks['Outside'],valid)
        total=sum(int(counts[m].sum()) for m in masks.values())
        assert total==r.calls_in_complete_bins
        checks.append(dict(animal_id=r.animal_id,eligible=r.eligible,
            calls_conserved=total==r.calls_in_complete_bins,
            observed_bins_conserved=sum(int(m.sum()) for m in masks.values())==r.valid_bins))
    return pd.DataFrame(rows),pd.DataFrame(checks)


def statistics(data:pd.DataFrame) -> pd.DataFrame:
    """Six two-sided genotype tests, conditioned on virus, with one BH family.

    Shuffle genotype labels only within virus while preserving group sizes.
    Equal recording weights avoid call-level pseudoreplication. No normality
    assumption is needed, but genotype exchangeability within virus and
    independent recordings are assumptions. Effect CIs bootstrap recordings
    within each genotype/virus cell, retaining the observed cell composition.
    """
    rng=np.random.default_rng(SEED);rows=[]
    for state in STATES:
        for metric in METRICS:
            d=data[data.eligible&(data.state==state)].dropna(subset=[metric])
            values=d[metric].to_numpy(float);wt=(d.genotype=='WT').to_numpy()
            viruses=d.virus.to_numpy();nw=int(wt.sum());nh=int((~wt).sum())
            weights=np.empty((N_PERMUTATIONS,len(d)))
            # Independent permutations within each virus, with unchanged cell sizes.
            for virus in np.unique(viruses):
                idx=np.flatnonzero(viruses==virus);n=int(wt[idx].sum())
                ranks=np.argsort(rng.random((N_PERMUTATIONS,len(idx))),axis=1)
                labels=np.zeros((N_PERMUTATIONS,len(idx)),bool)
                labels[np.arange(N_PERMUTATIONS)[:,None],ranks[:,:n]]=True
                weights[:,idx]=np.where(labels,1/nw,-1/nh)
            effect=float(values[wt].mean()-values[~wt].mean())
            null=weights@values
            p=float((1+(np.abs(null)>=abs(effect)-1e-12).sum())/(N_PERMUTATIONS+1))
            boot=np.zeros(N_BOOTSTRAP)
            for group,sign,total in [('WT',1,nw),('HET',-1,nh)]:
                for virus in np.unique(viruses):
                    cell=d[(d.genotype==group)&(d.virus==virus)][metric].to_numpy()
                    if len(cell):boot+=sign*rng.choice(cell,(N_BOOTSTRAP,len(cell))).sum(axis=1)/total
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(state=state,metric=metric,n_wt=nw,n_het=nh,
                mean_wt=float(values[wt].mean()),mean_het=float(values[~wt].mean()),
                median_wt=float(np.median(values[wt])),median_het=float(np.median(values[~wt])),
                effect_wt_minus_het=effect,effect_ci_low=lo,effect_ci_high=hi,
                p=p,mc_standard_error=np.sqrt(p*(1-p)/(N_PERMUTATIONS+1)),
                test='two-sided virus-stratified genotype permutation; difference of means',
                assumptions='genotype exchangeability within virus; recording independence',
                normality_required=False,ci='genotype/virus-stratified recording bootstrap',
                correction_family='six genotype contrasts, three metrics x two states'))
    result=pd.DataFrame(rows)
    for method,col in [('fdr_bh','q_bh'),('fdr_by','q_by'),('holm','p_holm')]:
        result[col]=multipletests(result.p,method=method)[1]
    return result


def draw(data:pd.DataFrame,stats:pd.DataFrame,out:Path) -> None:
    """Render narrow colored boxes with every recording and corrected q values."""
    plt.rcParams.update({'font.family':FONT,'font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'axes.edgecolor':'#52616A',
        'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    d=data[data.eligible];rng=np.random.default_rng(SEED)
    fig,axes=plt.subplots(2,3,figsize=FIGSIZE,sharey='col')
    fig.subplots_adjust(left=.10,right=.975,bottom=.115,top=.73,wspace=.34,hspace=.47)
    fig.text(.10,.952,'SOCIAL CONTEXT & VOCAL OUTPUT',fontsize=10,color='#667780',weight='bold')
    fig.text(.10,.898,'When do mice vocalize?',fontsize=23,weight='bold',color='#172F3D')
    fig.text(.10,.848,'WT vs Het  ·  each dot = one recording  ·  partner present, 5–15 min',fontsize=10,color='#667780')
    for col,(metric,title,ylabel) in enumerate(zip(METRICS,TITLES,YLABELS)):
        peak=float(d[metric].max());ylim=max(peak*1.27,1.)
        for row,state in enumerate(STATES):
            ax=axes[row,col];ax.set_ylim(-ylim*.035,ylim)
            ax.set_axisbelow(True);ax.yaxis.grid(True,color='#EBEFF2',lw=.7)
            ax.spines['bottom'].set_color('#A0ADB5')
            for pos,group in enumerate(GROUPS):
                vals=d[(d.state==state)&(d.genotype==group)][metric].dropna().to_numpy()
                ax.boxplot([vals],positions=[pos],widths=BOX_WIDTH,patch_artist=True,
                    showfliers=False,medianprops={'color':'white','linewidth':2},
                    boxprops={'facecolor':COLORS[group],'edgecolor':COLORS[group],'alpha':.90},
                    whiskerprops={'color':'#52616A','linewidth':1},
                    capprops={'color':'#52616A','linewidth':1})
                # Fixed seeded jitter exposes individual recordings without hiding zeros.
                ax.scatter(pos+rng.uniform(-JITTER,JITTER,len(vals)),vals,s=POINT_SIZE,
                    facecolors='white',edgecolors=COLORS[group],linewidths=1.15,zorder=4)
            r=stats[(stats.state==state)&(stats.metric==metric)].iloc[0]
            q=f'{r.q_bh:.3f}' if r.q_bh>=.001 else '<0.001'
            text=f'q = {q}' if not q.startswith('<') else f'q {q}'
            if r.q_bh>=ALPHA:text+='  n.s.'
            y=ylim*.88;h=ylim*.025
            ax.plot([0,0,1,1],[y,y+h,y+h,y],color='#52616A',lw=.9)
            ax.text(.5,y+h*1.6,text,ha='center',va='bottom',fontsize=9,color='#344954')
            ax.set_xlim(-.48,1.48)
            ax.set_xticks([0,1],[f'WT\nn={int(r.n_wt)}',f'Het\nn={int(r.n_het)}'])
            ax.tick_params(axis='both',length=3,color='#A0ADB5');ax.set_ylabel(ylabel,labelpad=7)
            ax.text(0,1.045,f'{chr(65+row*3+col)}',transform=ax.transAxes,fontsize=11,weight='bold',color='#172F3D')
            ax.set_title(f'{state} social behavior',loc='right',fontsize=10,color='#667780',pad=10)
            if row==0:ax.text(.5,1.18,title,ha='center',transform=ax.transAxes,fontsize=12,weight='bold',color='#172F3D')
    fig.text(.10,.033,'Probability: ≥1 call / 100 ms. Frequency: calls per observed minute. Counts: unnormalized totals.',fontsize=8,color='#667780')
    fig.text(.10,.010,'Social = any of 13 dyad behavior flags; outside = their complement. Brackets: genotype contrasts, BH across 6 tests.',fontsize=8,color='#667780')
    for fmt in FORMATS:fig.savefig(out/f'social_inside_outside_boxstrip.{fmt}',dpi=DPI,bbox_inches='tight')
    plt.close(fig)


def write_report(out:Path,data:pd.DataFrame,stats:pd.DataFrame) -> None:
    """Deliver definitions and interpretation alongside complete CSV statistics."""
    lines=['# Calls inside and outside social behavior','',
        '23 eligible recordings: 12 WT and 11 HET. Each dot is one recording.',
        'Inside is the union of 13 dyad behavior flags in completely observed 100 ms bins. '
        'Each flag requires its behavior to occupy at least 50% of that bin. Outside is the complementary observed state, '
        'while the partner is still present. It does not mean alone, untracked, or proven absence of every possible interaction.',
        'Probability = fraction of bins containing at least one call onset. Frequency = total call onsets / observed minutes. '
        'Count = unnormalized call onsets. Multiple calls in a bin increase frequency/count, but probability only once.','',
        '| State | Metric | WT mean | Het mean | WT minus Het [95% CI] | BH q |',
        '|---|---|---:|---:|---|---:|']
    for r in stats.itertuples():lines.append(f'| {r.state} | {r.metric} | {r.mean_wt:.3f} | {r.mean_het:.3f} | '
        f'{r.effect_wt_minus_het:.3f} [{r.effect_ci_low:.3f}, {r.effect_ci_high:.3f}] | {r.q_bh:.5f} |')
    exposure=data[data.eligible].groupby(['state','genotype']).exposure_s.mean()
    lines+=['','Mean observed exposure (seconds):',exposure.to_string(),'',
        'Boxes show median and quartiles; whiskers extend to 1.5 IQR. Every recording is displayed, including zeros.',
        'Six two-sided genotype permutation tests preserve genotype counts within virus. BH, BY and Holm are exported. '
        'The assumed independent unit is a recording, with genotype exchangeability conditional on virus. '
        'No Gaussian assumption is required. Bootstrap CIs resample recordings within genotype/virus cells and are pointwise.',
        'Inside and outside are paired states of the same recording. The brackets only compare WT with Het within each state, '
        'not inside with outside. Counts depend on exposure, so use frequency for comparisons of vocal output per unit time.',
        'Two stimulus partners are reused in four recordings, so residual dependence is possible. '
        'The microphone measures dyad calls and does not identify the caller. Behavioral labels are tracking-based rules.','',
        'Source: validated social_call_probability trace cache, original partner window 300–900 s, '
        '95% recording coverage criterion. Raw data and metadata remain unchanged.']
    (out/'README_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main() -> None:
    """Separate reproducible numerical extraction/testing from figure rendering."""
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--recompute',action='store_true');modes.add_argument('--figures-only',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    fp=json.loads(json.dumps(fingerprint()));meta=out/'provenance.json'
    files=['session_metrics.csv','statistics.csv','validation.csv']
    valid=meta.exists() and json.loads(meta.read_text())['fingerprint']==fp and all((out/f).exists() for f in files)
    if args.figures_only and not valid:raise RuntimeError('No valid numerical cache; run without --figures-only.')
    if valid and not args.recompute:print('Validated numeric cache reused.',flush=True)
    else:
        data,checks=extract();stats=statistics(data)
        data.to_csv(out/files[0],index=False);stats.to_csv(out/files[1],index=False);checks.to_csv(out/files[2],index=False)
        meta.write_text(json.dumps(dict(fingerprint=fp,created_utc=datetime.now(timezone.utc).isoformat(),
            source=str(INPUT),unit='recording/dyad',bin_s=source.BIN_S,
            partition='union of selected per-behavior >=50%-occupied dyad bins vs observed complement'),indent=2),encoding='utf-8')
    data=pd.read_csv(out/files[0],dtype={'animal_id':str});stats=pd.read_csv(out/files[1])
    draw(data,stats,out);write_report(out,data,stats)
    print(stats[['state','metric','mean_wt','mean_het','p','q_bh']].to_string(index=False),flush=True)


if __name__=='__main__':main()
