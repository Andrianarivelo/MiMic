"""Compare dyad calls around A acting on B versus B acting on A.

A is tracked mouse1, the experimental resident; B is tracked mouse2, partner.
Paired recordings and shared local shifts separate direction from differences
in background calling. Behavioral actor is never treated as acoustic caller.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
import social_call_probability as source

# Numerical configuration, separate from rendering.
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'lgdel_usv_analysis'/'social_call_probability'
OUTPUT=ROOT/'lgdel_usv_analysis'/'directional_social_calls'
BEHAVIORS=['nose2anogenital','nose2body','chasing']
N_SHIFT=19999
N_BOOT=5000
SEED=20261009
BLOCK_S=60.
MIN_SESSIONS=5
MIN_EVENTS_PER_DIRECTION=20
SCOPES=['WT','HET','combined']
ALPHA=.05

# Editable figure styling.
COLORS={'WT':'#327A9E','HET':'#E68A32','combined':'#354F5F'}
TITLES={'nose2anogenital':'Anogenital sniffing','nose2body':'Body sniffing','chasing':'Chasing'}
FIGSIZE=(12.0,7.5)
BOX_WIDTH=.32
JITTER=.055
DPI=400
FORMATS=['png','pdf','svg']
CONTRAST_LIMITS=(-25.,85.)


def fingerprint() -> dict:
    """Validate upstream scientific cache and hash only numeric dependencies."""
    p=json.loads((INPUT/'provenance.json').read_text(encoding='utf-8'))['fingerprint']
    if p!=json.loads(json.dumps(source.fingerprint())):raise RuntimeError('Refresh social_call_probability.py first.')
    return dict(version=1,upstream=p,
        inputs={name:hashlib.sha256((INPUT/name).read_bytes()).hexdigest() for name in ['traces.npz','source_qa.csv']},
        code=hashlib.sha256('\n'.join(inspect.getsource(f) for f in [analyse,supported]).encode()).hexdigest(),
        behaviors=BEHAVIORS,n_shift=N_SHIFT,n_boot=N_BOOT,seed=SEED,block=BLOCK_S,
        min_sessions=MIN_SESSIONS,min_events=MIN_EVENTS_PER_DIRECTION,scopes=SCOPES,alpha=ALPHA)


def supported(g:np.ndarray,den:np.ndarray,scope:str) -> bool:
    """Require recordings and complete episode support in both directions."""
    ok=len(g)>=MIN_SESSIONS and (den.sum(axis=0)>=MIN_EVENTS_PER_DIRECTION).all()
    if scope=='combined':ok &= min(int((g=='WT').sum()),int((g=='HET').sum()))>=2
    return bool(ok)


def analyse() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Use identical offset draws for both directional outcomes within a session.

    The tested contrast is observed A-minus-B minus its exact local-shift
    expectation. Its joint null retains correlations from overlapping windows.
    Recordings lacking either direction are omitted from that behavior only.
    """
    qa=pd.read_csv(INPUT/'source_qa.csv',dtype={'animal_id':str});qa=qa[qa.eligible]
    arrays=np.load(INPUT/'traces.npz');rows=[];nulls=[]
    js=[[source.DEFINITIONS.index((b,r)) for r in ['resident','partner']] for b in BEHAVIORS]
    with threadpool_limits(limits=1):
        for r in qa.itertuples():
            session={'animal_id':r.animal_id}
            for name in ['counts','valid','flags','onsets']:session[name]=arrays[f'{r.animal_id}_{name}']
            t=source.shift_statistics(session,BLOCK_S,N_SHIFT)
            contrasts=np.full((N_SHIFT,len(BEHAVIORS)),np.nan)
            for j,(b,indices) in enumerate(zip(BEHAVIORS,js)):
                obs=t['observed'][indices,1];exp=t['expected'][indices,1];den=t['denominator'][indices,1]
                paired=bool((den>0).all())
                if paired:contrasts[:,j]=t['null'][:,indices[0],1]-t['null'][:,indices[1],1]
                rows.append(dict(animal_id=r.animal_id,genotype=r.genotype,virus=r.virus,behavior=b,paired=paired,
                    events_a=int(den[0]),events_b=int(den[1]),p_a=obs[0],p_b=obs[1],
                    baseline_a=exp[0],baseline_b=exp[1],raw_difference=obs[0]-obs[1],
                    excess_a=obs[0]-exp[0],excess_b=obs[1]-exp[1],
                    directional_excess=(obs[0]-exp[0])-(obs[1]-exp[1])))
            nulls.append(contrasts)
            print(f'Paired directional null: {r.animal_id}',flush=True)
    data=pd.DataFrame(rows);null=np.stack(nulls,axis=-1);result=[];rng=np.random.default_rng(SEED)
    for j,b in enumerate(BEHAVIORS):
        d=data[data.behavior==b].reset_index(drop=True)
        for scope in SCOPES:
            use=d.paired.to_numpy()&((d.genotype==scope).to_numpy() if scope!='combined' else True)
            sub=d[use];g=sub.genotype.to_numpy();den=sub[['events_a','events_b']].to_numpy()
            ok=supported(g,den,scope);row=dict(behavior=b,scope=scope,n_sessions=len(sub),
                n_wt=int((g=='WT').sum()),n_het=int((g=='HET').sum()),events_a=int(den[:,0].sum()),events_b=int(den[:,1].sum()),
                status='ok' if ok else 'insufficient_support',p=np.nan)
            for col in ['p_a','p_b','baseline_a','baseline_b','raw_difference','excess_a','excess_b','directional_excess']:
                row[col]=float(source.aggregate(sub[col].to_numpy(),g,scope)) if len(sub) else np.nan
            baseline=row['baseline_a']-row['baseline_b']
            if ok:
                values=source.aggregate(null[:,j,use],g,scope)
                row['p']=float((1+(np.abs(values-baseline)>=abs(row['directional_excess'])-1e-12).sum())/(N_SHIFT+1))
            # Paired bootstrap preserves the A/B measures of each recording.
            groups=['WT','HET'] if scope=='combined' else [scope]
            for col in ['raw_difference','directional_excess']:
                samples=np.zeros(N_BOOT)
                for group in groups:
                    x=sub[sub.genotype==group][col].to_numpy()
                    if not len(x):samples[:]=np.nan;break
                    samples+=rng.choice(x,(N_BOOT,len(x))).mean(axis=1)/len(groups)
                lo,hi=np.quantile(samples,[.025,.975]);row[col+'_ci_low']=lo;row[col+'_ci_high']=hi
            row['p_monte_carlo_se']=np.sqrt(row['p']*(1-row['p'])/(N_SHIFT+1))
            result.append(row)
    stats=pd.DataFrame(result)
    for method,col in [('fdr_bh','q_bh'),('fdr_by','q_by'),('holm','p_holm')]:stats[col]=multipletests(stats.p.fillna(1),method=method)[1]
    checks=pd.DataFrame([dict(check='eligible_recordings',passed=len(qa)==23,value=len(qa)),
        dict(check='direction_pairs',passed=len(data)==23*len(BEHAVIORS),value=len(data)),
        dict(check='same_recordings_both_directions',passed=bool(data[data.paired][['p_a','p_b']].notna().all().all()),value=int(data.paired.sum())),
        dict(check='probabilities_bounded',passed=bool(data[['p_a','p_b']].stack().dropna().between(0,1).all()),value='all defined estimates'),
        dict(check='effect_identity',passed=bool(np.allclose(data.directional_excess,data.excess_a-data.excess_b,equal_nan=True)),value='paired contrast'),
        dict(check='nine_test_family',passed=len(stats)==3*len(BEHAVIORS),value=len(stats)),
        dict(check='undefined_not_zero',passed=bool(data[~data.paired].directional_excess.isna().all()),value='absent direction omitted')])
    if not checks.passed.all():raise AssertionError(checks.to_string())
    return data,stats,checks


def draw(data:pd.DataFrame,stats:pd.DataFrame,out:Path) -> None:
    """Pair A-to-B/B-to-A boxstrips above local-baseline directional contrasts."""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'axes.edgecolor':'#7A8992','axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=FIGSIZE,gridspec_kw={'height_ratios':[1.15,1]})
    fig.subplots_adjust(left=.075,right=.975,bottom=.14,top=.72,wspace=.33,hspace=.63)
    fig.text(.075,.958,'DIRECTION OF INTERACTION',fontsize=10,weight='bold',color='#667780')
    fig.text(.075,.909,'Who is acting when calls occur?',fontsize=23,weight='bold',color='#172F3D')
    fig.text(.075,.856,'A = experimental mouse (WT or Het)    ·    B = partner',fontsize=12,color='#344954')
    fig.text(.075,.813,'A → B: A sniffs / chases B     |     B → A: B sniffs / chases A',fontsize=10,color='#667780')
    rng=np.random.default_rng(SEED)
    for j,b in enumerate(BEHAVIORS):
        ax=axes[0,j];ax.set_ylim(-4,112);ax.set_xlim(-.5,3.5)
        ax.yaxis.grid(True,color='#ECF0F2',lw=.7);ax.set_axisbelow(True)
        for k,group in enumerate(['WT','HET']):
            d=data[(data.behavior==b)&data.paired&(data.genotype==group)]
            positions=[2*k,2*k+1];xshift=rng.uniform(-JITTER,JITTER,len(d))
            for offset,(_,r) in zip(xshift,d.iterrows()):
                ax.plot(np.array(positions)+offset,100*np.array([r.p_a,r.p_b]),color=COLORS[group],alpha=.20,lw=.7,zorder=1)
            for direction,col in enumerate(['p_a','p_b']):
                pos=positions[direction];vals=100*d[col].to_numpy()
                ax.boxplot([vals],positions=[pos],widths=BOX_WIDTH,patch_artist=True,showfliers=False,
                    boxprops={'facecolor':COLORS[group],'edgecolor':COLORS[group],'alpha':.85 if direction==0 else .30},
                    medianprops={'color':'white' if direction==0 else COLORS[group],'linewidth':1.8},
                    whiskerprops={'color':COLORS[group],'linewidth':.8},capprops={'color':COLORS[group],'linewidth':.8})
                ax.scatter(pos+xshift,vals,s=24,facecolors='white',edgecolors=COLORS[group],linewidths=.9,zorder=3)
                ax.scatter(pos,100*d['baseline_a' if direction==0 else 'baseline_b'].mean(),s=36,marker='D',facecolors='none',edgecolors='#344954',linewidths=1,zorder=5)
            r=stats[(stats.behavior==b)&(stats.scope==group)].iloc[0]
            label=('WT' if group=='WT' else 'Het')+f' · n={len(d)}'+(' †' if r.status!='ok' else '')
            ax.text((positions[0]+positions[1])/2,106,label,ha='center',fontsize=9,color=COLORS[group],weight='bold')
        ax.set_xticks(range(4),['A → B','B → A','A → B','B → A']);ax.tick_params(length=3)
        ax.set_ylabel('P(≥1 call within ±1 s) (%)' if j==0 else '')
        ax.set_title(TITLES[b],fontsize=12,weight='bold',color='#172F3D',pad=17)
        ax.text(-.12,1.035,chr(65+j),transform=ax.transAxes,weight='bold',fontsize=11)
        lower=axes[1,j];lower.axvline(0,color='#B1BCC3',ls=':',lw=1)
        for i,scope in enumerate(SCOPES):
            r=stats[(stats.behavior==b)&(stats.scope==scope)].iloc[0]
            x=100*r.directional_excess;lo=100*r.directional_excess_ci_low;hi=100*r.directional_excess_ci_high
            supported_row=r.status=='ok'
            if supported_row:lower.errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',color=COLORS[scope],capsize=3,ms=5,lw=1.2)
            else:lower.scatter(x,i,s=30,facecolors='none',edgecolors=COLORS[scope],alpha=.6)
            text=f'q={r.q_bh:.3f}' if supported_row else 'descriptive †'
            lower.text(.98,i-.22,text,transform=lower.get_yaxis_transform(),ha='right',va='bottom',fontsize=8,color='#667780')
        shown=stats[stats.status=='ok']
        limits=(min(CONTRAST_LIMITS[0],100*shown.directional_excess_ci_low.min()-5),
                max(CONTRAST_LIMITS[1],100*shown.directional_excess_ci_high.max()+18))
        lower.set_ylim(2.6,-.6);lower.set_xlim(*limits)
        lower.set_yticks(range(3),['WT','Het','Combined'])
        lower.set_xlabel('A → B minus B → A excess (percentage points)',fontsize=8)
        lower.set_title('Direction contrast above local baseline',fontsize=10,color='#344954',pad=12)
        lower.text(-.12,1.045,chr(68+j),transform=lower.transAxes,weight='bold',fontsize=11)
        lower.spines['left'].set_visible(False);lower.tick_params(axis='y',length=0)
    fig.text(.075,.071,'Top: paired recordings, narrow quartile boxes; diamonds = local-shift reference. Bottom: paired effects with recording-bootstrap 95% CI.',fontsize=8,color='#667780')
    fig.text(.075,.044,'Positive effect: more call enrichment when A acts on B. q: joint local-shift tests, BH across 9 contrasts. † Insufficient support.',fontsize=8,color='#667780')
    fig.text(.075,.017,'The behavior identifies the actor, not the caller. These directional contexts may guide a caller hypothesis; they do not identify the vocal source.',fontsize=8,color='#344954')
    for fmt in FORMATS:fig.savefig(out/f'directional_social_calls.{fmt}',dpi=DPI,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def report(out:Path,stats:pd.DataFrame) -> None:
    """Record effect meaning, paired selection, multiplicity and caller limits."""
    lines=['# Qui agit lorsque les appels surviennent ?','',
        'A = souris expérimentale résidente (mouse1), B = partenaire (mouse2). Le génotype WT/Het est celui de A.',
        'Les panneaux supérieurs comparent la probabilité d’au moins un appel dans [-1,+1) s autour du début '
        'du comportement. A→B signifie que A réalise le comportement dirigé vers B, et inversement pour B→A.',
        'Chaque comparaison utilise uniquement les mêmes sessions ayant des fenêtres complètes dans les deux directions. '
        'Les appels et les fenêtres peuvent se chevaucher entre directions. Un appel ne doit pas être compté comme une observation indépendante.','',
        '| Comportement | Groupe | Sessions appariées | Épisodes A→B / B→A | P A→B (%) | P B→A (%) | Différence brute (points) | Contraste d’enrichissement (points) | q BH | Support |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in stats.itertuples():lines.append(f'| {TITLES[r.behavior]} | {r.scope} | {r.n_sessions} | {r.events_a}/{r.events_b} | '
        f'{100*r.p_a:.2f} | {100*r.p_b:.2f} | {100*r.raw_difference:.2f} | {100*r.directional_excess:.2f} | {r.q_bh:.4f} | {r.status} |')
    lines+=['','## Interprétation et méthode','',
        'Le contraste inférieur vaut (P A→B − référence A→B) − (P B→A − référence B→A). '
        'Il évalue une préférence de contexte au-delà du niveau local d’appels. Le test compare la différence observée '
        'à sa distribution obtenue par les mêmes décalages d’appels dans les deux directions.',
        '19 999 combinaisons de décalages cycliques uniformes dans les segments complètement observés de blocs de 60 s, '
        'incluant zéro. Hypothèse de stationnarité locale et invariance aux décalages. Les fronts de blocs interrompent la conservation des salves.',
        'Une session est une unité statistique. Moyennes non pondérées par le nombre d’épisodes. Combiné standardisé à 50 % WT et 50 % Het. '
        '5 000 bootstraps appariés de sessions, stratifiés par génotype; IC ponctuels. Les partenaires réutilisés peuvent créer une dépendance résiduelle.',
        'Au moins cinq sessions et 20 épisodes complets dans chaque direction sont requis par test; le combiné exige au moins deux sessions par génotype. '
        'La poursuite Het ne remplit pas ces seuils : pas de test concluant ni d’IC affiché. Les neuf tests forment une famille BH/BY/Holm secondaire exploratoire.',
        'Le contraste WT anogénital passe également BY. Les contrastes WT du corps et de la poursuite passent BH mais pas BY. '
        'L’IC bootstrap WT de poursuite est large et inclut zéro malgré le test conditionnel de décalages significatif : '
        'ces deux méthodes quantifient respectivement l’incertitude entre sessions et l’association temporelle conditionnelle aux traces observées. '
        'Un résultat significatif WT et non significatif Het ne démontre pas une interaction avec le génotype; celle-ci n’est pas testée ici.',
        'Les appels sont associés au contexte de l’initiateur ou du receveur du comportement. Un enrichissement A→B pourrait être compatible '
        'avec une vocalisation de A, une réponse de B ou les deux. Il ne distingue pas ces hypothèses. '
        'Cette figure ne démontre donc pas que l’acteur est le locuteur.',
        'Les labels comportementaux sont des règles issues du suivi. Fenêtre partenaire 300–900 s, ≥95 % couverture, '
        'épisodes et fenêtres complètes selon le pipeline amont. Onsets quantifiés à 100 ms. Aucun fichier brut ou métadonnée n’est modifié.']
    (out/'RAPPORT_DIRECTION_APPELS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main() -> None:
    """Cache new paired inference before rendering the direction figure."""
    parser=argparse.ArgumentParser(description=__doc__)
    m=parser.add_mutually_exclusive_group();m.add_argument('--recompute',action='store_true');m.add_argument('--figures-only',action='store_true')
    args=parser.parse_args();out=OUTPUT;out.mkdir(parents=True,exist_ok=True)
    fp=json.loads(json.dumps(fingerprint()));meta=out/'provenance.json'
    valid=meta.exists() and json.loads(meta.read_text())['fingerprint']==fp and all((out/f).exists() for f in ['session_directional_calls.csv','direction_statistics.csv','validation.csv'])
    if args.figures_only and not valid:raise RuntimeError('Run without --figures-only to create the numeric cache.')
    if valid and not args.recompute:print('Validated directional cache reused.',flush=True)
    else:
        data,stats,checks=analyse();data.to_csv(out/'session_directional_calls.csv',index=False)
        stats.to_csv(out/'direction_statistics.csv',index=False);checks.to_csv(out/'validation.csv',index=False)
        meta.write_text(json.dumps(dict(fingerprint=fp,actor_a='mouse1 experimental resident',actor_b='mouse2 partner',
            acoustic_unit='dyad',paired_selection='complete onset windows in both directions',
            source=str(INPUT),comparison='paired local-shift directional enrichment',family='9 exploratory direction tests'),indent=2),encoding='utf-8')
    data=pd.read_csv(out/'session_directional_calls.csv',dtype={'animal_id':str});stats=pd.read_csv(out/'direction_statistics.csv')
    draw(data,stats,out);report(out,stats)
    print(stats[['behavior','scope','n_sessions','p_a','p_b','directional_excess','q_bh','status']].to_string(index=False),flush=True)


if __name__=='__main__':main()
