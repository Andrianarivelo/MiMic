"""Four-condition WT/Het social-context boxstrip with mixed ANOVA and Holm p.

Analyze only 300-900 s. Include all 24 recordings as explicitly requested;
unobserved bins remain excluded. Never count inside/outside as independent
animals. Preserve the earlier 23-recording figure and its inference unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import social_inside_outside_figure as previous

# Analysis parameters: all 12 WT + 12 Het, as requested, on unchanged raw units.
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'lgdel_usv_analysis'/'social_context_mixed_anova'
METRICS=list(previous.METRICS)
GROUPS=['WT','HET']
CONTEXTS=['Inside','Outside']
INCLUDE_ALL_RECORDINGS=True
ALPHA=.05
N_BOOTSTRAP=5000
SEED=20261008
POSTHOC_FAMILY='all 18 pairwise comparisons across three metrics'
ANOVA_FAMILY='all 9 ANOVA effects across three metrics'

# Rendering parameters: outside is a lighter version of the genotype color.
COLORS={'WT':'#327A9E','HET':'#E68A32'}
LIGHT_COLORS={'WT':'#BED5E1','HET':'#F6D9BB'}
TITLES=['Call probability','Call frequency','Number of calls']
YLABELS=['Bins with ≥1 call (%)','Calls / min','Calls']
FIGSIZE=(12.6,8.0)
BOX_WIDTH=.34
POINT_SIZE=24
JITTER=.065
DPI=400
FORMATS=['png','pdf','svg']


def fingerprint() -> dict:
    """Keep numeric settings and sources independent of cosmetic revisions."""
    return dict(version=1,upstream=previous.fingerprint(),metrics=METRICS,
        include_all=INCLUDE_ALL_RECORDINGS,groups=GROUPS,contexts=CONTEXTS,
        n_bootstrap=N_BOOTSTRAP,seed=SEED,alpha=ALPHA,
        posthoc_family=POSTHOC_FAMILY,anova_family=ANOVA_FAMILY,
        code=hashlib.sha256('\n'.join(inspect.getsource(f) for f in
            [calculate,bootstrap_contrast]).encode()).hexdigest())


def bootstrap_contrast(a:np.ndarray,b:np.ndarray,paired:bool,rng:np.random.Generator) -> tuple[float,float]:
    """Estimate a raw mean difference CI with recording-level resampling."""
    if paired:
        delta=a-b;samples=rng.choice(delta,(N_BOOTSTRAP,len(delta))).mean(axis=1)
    else:
        samples=rng.choice(a,(N_BOOTSTRAP,len(a))).mean(axis=1)-rng.choice(b,(N_BOOTSTRAP,len(b))).mean(axis=1)
    return tuple(np.quantile(samples,[.025,.975]))


def calculate() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Fit two-level split-plot ANOVA without duplicating subject independence.

    Between-subject genotype is tested on recording means. Within-subject
    context and genotype x context are tested on paired Inside-Outside
    differences. Sum-coded Type III tests give equal-group marginal context
    means. For two within levels this is the classical mixed ANOVA; sphericity
    is automatic. Raw-scale normality and variance checks are exported, plus
    HC3 heteroscedasticity-robust F tests as a separate sensitivity, not a
    substitute for the requested classical tests.
    """
    data,validation=previous.extract()
    data['included']=True if INCLUDE_ALL_RECORDINGS else data.eligible
    rows=[];post=[];checks=[];sensitivity=[];rng=np.random.default_rng(SEED)
    cells=list(itertools.product(GROUPS,CONTEXTS))
    for metric in METRICS:
        d=data[data.included]
        wide=d.pivot(index=['animal_id','genotype'],columns='state',values=metric).dropna().reset_index()
        wide['subject_mean']=(wide.Inside+wide.Outside)/2
        wide['paired_difference']=wide.Inside-wide.Outside
        models={name:ols(f'{name} ~ C(genotype, Sum)',data=wide).fit()
                for name in ['subject_mean','paired_difference']}
        # Scaling the response by sqrt(2) multiplies both SS terms equally and
        # therefore cancels in F. df1=1 and df2=N-2 for each effect.
        effects=[('Genotype','subject_mean','C(genotype, Sum)'),
                 ('Context','paired_difference','Intercept'),
                 ('Genotype × context','paired_difference','C(genotype, Sum)')]
        for effect,key,term in effects:
            model=models[key];table=anova_lm(model,typ=3);r=table.loc[term]
            eta=float(r.sum_sq/(r.sum_sq+table.loc['Residual','sum_sq']))
            rows.append(dict(metric=metric,effect=effect,F=float(r.F),df1=float(r.df),df2=float(model.df_resid),
                p=float(r['PR(>F)']),partial_eta_squared=eta,n_recordings=len(wide),
                n_wt=int((wide.genotype=='WT').sum()),n_het=int((wide.genotype=='HET').sum()),
                scale='raw',model='two-way mixed ANOVA, sum-coded Type III, genotype between/context within',
                correction_family=ANOVA_FAMILY))
            robust=anova_lm(model,typ=3,robust='hc3').loc[term]
            sensitivity.append(dict(metric=metric,effect=effect,F=float(robust.F),p=float(robust['PR(>F)']),
                df1=float(robust.df),df2=float(model.df_resid),method='HC3 robust covariance F sensitivity, same raw-scale contrasts'))
        for key,model in models.items():
            residuals=model.resid.to_numpy();normal=stats.shapiro(residuals)
            a=wide[wide.genotype=='WT'][key].to_numpy();b=wide[wide.genotype=='HET'][key].to_numpy()
            variance=stats.levene(a,b,center='median')
            checks.extend([dict(metric=metric,component=key,assumption='normal residuals',
                test='Shapiro-Wilk',statistic=normal.statistic,p=normal.pvalue,flag=normal.pvalue<ALPHA),
                dict(metric=metric,component=key,assumption='equal genotype variances',
                test='Brown-Forsythe (median-centered Levene)',statistic=variance.statistic,p=variance.pvalue,flag=variance.pvalue<ALPHA)])
        # Six cell comparisons per metric. Same genotype is paired; between
        # genotype uses Welch to avoid imposing equal raw-scale variances.
        for (g1,c1),(g2,c2) in itertools.combinations(cells,2):
            paired=g1==g2
            a=wide[wide.genotype==g1][c1].to_numpy();b=wide[wide.genotype==g2][c2].to_numpy()
            t=stats.ttest_rel(a,b) if paired else stats.ttest_ind(a,b,equal_var=False)
            lo,hi=bootstrap_contrast(a,b,paired,rng)
            post.append(dict(metric=metric,group1=g1,context1=c1,group2=g2,context2=c2,
                n1=len(a),n2=len(b),paired=paired,t=float(t.statistic),df=float(t.df),p=float(t.pvalue),
                mean1=float(a.mean()),mean2=float(b.mean()),mean_difference=float(a.mean()-b.mean()),
                bootstrap_ci_low=lo,bootstrap_ci_high=hi,
                test='paired t' if paired else 'Welch independent t',
                correction_family=POSTHOC_FAMILY,scale='raw'))
    anova=pd.DataFrame(rows);posthoc=pd.DataFrame(post);assumptions=pd.DataFrame(checks);robust=pd.DataFrame(sensitivity)
    for table in [anova,posthoc,robust]:table['p_holm']=multipletests(table.p,method='holm')[1]
    data.to_csv(OUTPUT/'session_metrics.csv',index=False)
    validation.to_csv(OUTPUT/'partition_validation.csv',index=False)
    return data,anova,posthoc,assumptions,robust


def ptext(p:float) -> str:
    """Format adjusted p without ever referring to it as q."""
    return '<0.001' if p<.001 else f'{p:.3f}'


def validate(data:pd.DataFrame,anova:pd.DataFrame,post:pd.DataFrame,out:Path) -> None:
    """Cross-check mixed-model F against independent split-plot sums of squares."""
    checks=[]
    d=data[data.included]
    counts=d.groupby(['genotype','state']).size()
    expected=12 if INCLUDE_ALL_RECORDINGS else None
    checks.append(dict(check='paired cohort',passed=all(n==expected for n in counts) if expected else len(counts)==4,value=counts.to_dict()))
    checks.append(dict(check='three complete posthoc families',passed=len(post)==18,value=len(post)))
    for metric in METRICS:
        w=d.pivot(index=['animal_id','genotype'],columns='state',values=metric).dropna().reset_index()
        means=(w.Inside+w.Outside)/2;delta=w.Inside-w.Outside;n=len(w)
        grandmean=means.mean();granddelta=delta.mean()
        ss_g=ss_subject=ss_interaction=ss_error=0.
        for group in GROUPS:
            use=w.genotype==group;m=means[use];dd=delta[use]
            ss_g+=2*len(m)*(m.mean()-grandmean)**2
            ss_subject+=2*((m-m.mean())**2).sum()
            ss_interaction+=.5*len(dd)*(dd.mean()-granddelta)**2
            ss_error+=.5*((dd-dd.mean())**2).sum()
        direct={'Genotype':ss_g/(ss_subject/(n-2)),
                'Context':(.5*n*granddelta**2)/(ss_error/(n-2)),
                'Genotype × context':ss_interaction/(ss_error/(n-2))}
        for effect,F in direct.items():
            r=anova[(anova.metric==metric)&(anova.effect==effect)].iloc[0]
            checks.append(dict(check=f'{metric}: {effect} independent split-plot F',passed=bool(np.isclose(F,r.F,rtol=1e-10)),value=F))
    for name,table in [('ANOVA',anova),('posthoc',post)]:
        checks.append(dict(check=f'{name} Holm bounds',passed=bool(((table.p_holm>=table.p-1e-12)&(table.p_holm<=1)).all()),value=len(table)))
    result=pd.DataFrame(checks);result.to_csv(out/'analysis_validation.csv',index=False)
    if not result.passed.all():raise AssertionError(result.to_string())


def draw(data:pd.DataFrame,anova:pd.DataFrame,post:pd.DataFrame,assumptions:pd.DataFrame,out:Path) -> None:
    """Show the four conditions together with paired strips and Holm p labels."""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'axes.edgecolor':'#7A8992',
        'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=FIGSIZE)
    fig.subplots_adjust(left=.07,right=.975,bottom=.34,top=.73,wspace=.30)
    fig.text(.07,.95,'SOCIAL CONTEXT & VOCAL OUTPUT',fontsize=10,weight='bold',color='#667780')
    fig.text(.07,.901,'Inside versus outside social bouts',fontsize=22,weight='bold',color='#172F3D')
    fig.text(.07,.851,'Partner present only: 300–900 s   ·   WT n=12 / Het n=12   ·   each dot = one recording',fontsize=11,color='#667780')
    fig.text(.07,.803,'Solid color: inside     |     Light color: outside     |     Lines connect the same recording',fontsize=10,color='#667780')
    positions=[0,1,2.5,3.5];cells=list(itertools.product(GROUPS,CONTEXTS));rng=np.random.default_rng(SEED)
    # Nonoverlapping short brackets share a level; all six pairwise comparisons
    # appear, including the two cross-context/cross-genotype contrasts.
    levels={(0,1):0,(2,3):0,(0,2):1,(1,3):2,(1,2):3,(0,3):4}
    for j,(metric,title,ylabel) in enumerate(zip(METRICS,TITLES,YLABELS)):
        ax=axes[j];d=data[data.included];peak=max(float(d[metric].max()),1.)
        ax.set_ylim(-.055*peak,peak*1.74);ax.set_xlim(-.5,4)
        ax.yaxis.grid(True,color='#EAF0F3',lw=.7);ax.set_axisbelow(True)
        for k,g in enumerate(GROUPS):
            w=d[d.genotype==g].pivot(index='animal_id',columns='state',values=metric).dropna()
            offsets=rng.uniform(-JITTER,JITTER,len(w))
            for offset,(_,r) in zip(offsets,w.iterrows()):
                ax.plot(np.array(positions[2*k:2*k+2])+offset,[r.Inside,r.Outside],color=COLORS[g],alpha=.16,lw=.7,zorder=1)
            for ci,c in enumerate(CONTEXTS):
                pos=positions[2*k+ci];values=w[c].to_numpy();color=COLORS[g] if ci==0 else LIGHT_COLORS[g]
                ax.boxplot([values],positions=[pos],widths=BOX_WIDTH,patch_artist=True,showfliers=False,
                    boxprops={'facecolor':color,'edgecolor':COLORS[g],'linewidth':1},
                    medianprops={'color':'white' if ci==0 else COLORS[g],'linewidth':1.8},
                    whiskerprops={'color':COLORS[g],'linewidth':.8},capprops={'color':COLORS[g],'linewidth':.8})
                ax.scatter(pos+offsets,values,s=POINT_SIZE,facecolors='white',edgecolors=COLORS[g],linewidths=.9,zorder=3)
        t=post[post.metric==metric]
        for (i1,i2),level in levels.items():
            g1,c1=cells[i1];g2,c2=cells[i2]
            r=t[(t.group1==g1)&(t.context1==c1)&(t.group2==g2)&(t.context2==c2)].iloc[0]
            y=peak*(1.07+level*.13);h=peak*.022
            ax.plot([positions[i1],positions[i1],positions[i2],positions[i2]],[y,y+h,y+h,y],color='#667780',lw=.75)
            ax.text((positions[i1]+positions[i2])/2,y+h*1.7,f'pH {ptext(r.p_holm)}',
                ha='center',va='bottom',fontsize=7.5,color='#344954')
        ax.set_xticks(positions,['WT\nInside','WT\nOutside','Het\nInside','Het\nOutside'])
        ax.set_ylabel(ylabel);ax.set_title(title,fontsize=12,weight='bold',color='#172F3D',pad=16)
        ax.text(-.12,1.035,chr(65+j),transform=ax.transAxes,fontsize=11,weight='bold')
        ax.tick_params(length=3)
        # Put all three ANOVA terms below the plot, with F and adjusted p.
        aa=anova[anova.metric==metric]
        box=ax.get_position();ta=fig.add_axes([box.x0,.135,box.width,.135]);ta.axis('off')
        ta.text(0,1.09,'Two-way mixed ANOVA',fontsize=10,weight='bold',color='#344954')
        ta.text(0,.83,'Effect',fontsize=8,color='#667780');ta.text(.57,.83,'F(1,22)',fontsize=8,color='#667780');ta.text(.99,.83,'p Holm',ha='right',fontsize=8,color='#667780')
        for k,r in enumerate(aa.itertuples()):
            yy=.58-k*.26
            ta.text(0,yy,r.effect,fontsize=8,color='#344954')
            ta.text(.57,yy,f'{r.F:.2f}',fontsize=8,color='#344954')
            ta.text(.99,yy,ptext(r.p_holm),ha='right',fontsize=8,color='#344954')
    fig.text(.07,.079,'pH = Holm-adjusted p across all 18 cell comparisons. ANOVA p: Holm across all 9 effects. Raw units; paired context within recording.',fontsize=8,color='#667780')
    fig.text(.07,.051,'Probability: ≥1 call / 100 ms. Frequency: calls / observed min. Counts: raw totals. Outside remains within the partner-present period.',fontsize=8,color='#667780')
    caution='Raw-scale ANOVA assumptions show violations; treat inference as exploratory. HC3 sensitivity and complete diagnostics are exported.' if assumptions.flag.any() else 'Residual normality and genotype variance diagnostics exported; recording independence remains an assumption.'
    fig.text(.07,.023,caution,fontsize=8,color='#667780')
    for fmt in FORMATS:fig.savefig(out/f'social_context_mixed_anova.{fmt}',dpi=DPI,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def report(out:Path,data:pd.DataFrame,anova:pd.DataFrame,post:pd.DataFrame,assumptions:pd.DataFrame) -> None:
    """Explain cohort change, mixed design, posthoc family and diagnostics."""
    lines=['# Four-condition social-context figure','',
        'Both Inside and Outside use only 300–900 s, while the partner is present. '
        'The social definition is unchanged: union of 13 dyad behavior flags in completely observed 100 ms bins. '
        'Each per-behavior flag requires ≥50% occupancy. Outside is their observed complement, not the alone phase.',
        'All 12 WT and 12 Het recordings are included as explicitly requested. Session 31101 has 94.4% coverage '
        'and was excluded from the previous 23-recording figure. It is included here with only fully observed bins. '
        'Missing tracking is never treated as outside or zero calling. The old figure remains unchanged.',
        'Probability, frequency and counts retain the original raw units. No logarithm or normalization of counts is applied. '
        'Counts depend on time in each context; frequency uses the actual observed exposure. The microphone records the dyad.','',
        '## Two-way mixed ANOVA','',
        'Between factor: experimental-mouse genotype. Within factor: Inside/Outside, paired in each recording. '
        'Classical sum-coded Type III split-plot ANOVA is computed via subject means (genotype) and Inside−Outside differences '
        '(context and genotype×context). With two within levels sphericity is automatic. All tests have df1=1, df2=22. '
        'Equal-genotype marginal weighting defines the context effect. Virus is not a factor in this requested two-factor model. '
        'All nine effects share a Holm family; raw and adjusted p, F, df and partial eta squared are exported.','',
        '| Metric | Effect | F | Raw p | Holm p |',
        '|---|---|---:|---:|---:|']
    for r in anova.itertuples():lines.append(f'| {r.metric} | {r.effect} | {r.F:.4f} | {r.p:.6f} | {r.p_holm:.6f} |')
    lines+=['','## Multiple comparisons','',
        'All six pairs of the four cells per metric are tested (18 comparisons across the three metrics). '
        'Within genotype: paired t tests. Between genotypes: Welch independent t tests, including cross-context cell comparisons. '
        'All 18 share one Holm family, rather than selectively correcting only displayed significant tests. '
        'Every adjusted p is shown on the figure. Per-comparison raw effect differences and recording-bootstrap 95% CIs are exported. '
        'These CIs are pointwise and are not simultaneous Holm-adjusted intervals.','',
        '## Assumptions and sensitivity','',
        f'{int(assumptions.flag.sum())}/{len(assumptions)} residual-normality and equal-variance diagnostics flag violations at p<0.05. '
        'The raw outcomes are strongly skewed. The requested classical ANOVA and t tests therefore remain exploratory. '
        'Holm correction addresses multiple testing but does not repair distributional assumptions.',
        'HC3 heteroscedasticity-robust covariance F tests for the same nine contrasts are exported separately; '
        'they address unequal variances but are not an exact small-sample solution to extreme skew. '
        'No transformation is selected after inspecting results. Reused stimulus partners can induce residual dependence.',
        'Outputs: session_metrics.csv, mixed_anova.csv, posthoc_comparisons.csv, assumption_checks.csv, '
        'hc3_sensitivity.csv, partition_validation.csv, provenance.json, and PNG/PDF/SVG.',
        'Methods: [statsmodels Type III ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html), '
        '[SciPy paired t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html), '
        '[SciPy Welch t](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html).']
    (out/'README_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main() -> None:
    """Reproduce numerical cache or regenerate only its figure and report."""
    parser=argparse.ArgumentParser(description=__doc__)
    m=parser.add_mutually_exclusive_group();m.add_argument('--recompute',action='store_true');m.add_argument('--figures-only',action='store_true')
    args=parser.parse_args();out=OUTPUT;out.mkdir(parents=True,exist_ok=True)
    fp=json.loads(json.dumps(fingerprint()));meta=out/'provenance.json'
    names=['session_metrics','mixed_anova','posthoc_comparisons','assumption_checks','hc3_sensitivity']
    valid=meta.exists() and json.loads(meta.read_text())['fingerprint']==fp and all((out/f'{n}.csv').exists() for n in names)
    if args.figures_only and not valid:raise RuntimeError('Create numerical cache first by running without --figures-only.')
    if valid and not args.recompute:print('Validated mixed-ANOVA cache reused.',flush=True)
    else:
        tables=calculate()
        for name,table in zip(names,tables):table.to_csv(out/f'{name}.csv',index=False)
        meta.write_text(json.dumps(dict(fingerprint=fp,window_s=[300,900],n_wt=12,n_het=12,
            repeated_unit='recording',cohort_change='include 31101 per user request; fully observed bins only',
            model='two-way mixed ANOVA: genotype between, context within',scale='raw'),indent=2),encoding='utf-8')
    data,anova,post,assumptions,robust=[pd.read_csv(out/f'{n}.csv',dtype={'animal_id':str}) for n in names]
    validate(data,anova,post,out)
    draw(data,anova,post,assumptions,out);report(out,data,anova,post,assumptions)
    print(anova[['metric','effect','F','p','p_holm']].to_string(index=False))
    print(post[['metric','group1','context1','group2','context2','p_holm']].to_string(index=False))


if __name__=='__main__':main()
