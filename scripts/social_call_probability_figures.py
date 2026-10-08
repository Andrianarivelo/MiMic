"""Figures and French interpretation for conditional call probabilities.

Rendering is deliberately independent of numeric caches. All probabilities
refer to dyad call onsets, not an identified vocal source.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from social_vocal_correlation_figures import configure,save,BEH_LABELS,COLORS,number

# Editable rendering configuration.
FIGURE_DPI=320
GENOTYPES=['WT','HET']
GENOTYPE_OFFSETS={'WT':-.13,'HET':.13}
SMOOTH_BINS=5
FONT_SIZE=9


def ranked_order(stats:pd.DataFrame,metric:str,role='dyad') -> list[str]:
    """Rank only descriptively, placing behaviors with insufficient support last."""
    rows=stats[(stats.role==role)&(stats.scope=='combined')&(stats.metric==metric)].copy()
    rows['supported']=rows.status=='ok'
    return rows.sort_values(['supported','observed_probability'],ascending=False).behavior.tolist()


def probabilities(folder:Path,stats:pd.DataFrame,role:str) -> None:
    """Show genotype-specific probabilities and exact occupied-time call rates."""
    order=ranked_order(stats,'around_1s',role)
    fig,axes=plt.subplots(1,3,figsize=(12.8,7.4),layout='constrained',sharey=True)
    for ax,metric,title in zip(axes,['during_100ms','around_1s','exact_rate'],
        ['During behavior: any call / 100 ms','Around onset: any call in ±1 s','Descriptive occupied-time call rate']):
        for i,behavior in enumerate(order):
            for g in GENOTYPES:
                selected=stats[(stats.role==role)&(stats.behavior==behavior)&(stats.scope==g)&
                               (stats.metric==('during_100ms' if metric=='exact_rate' else metric))]
                row=selected.iloc[0]
                yy=i+GENOTYPE_OFFSETS[g]
                if metric=='exact_rate':
                    ax.scatter(row.exact_call_rate_per_min,yy,s=29,color=COLORS[g],alpha=.9)
                elif np.isfinite(row.observed_probability):
                    mean,lo,hi=100*row.observed_probability,100*row.ci_low,100*row.ci_high
                    ax.errorbar(mean,yy,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],
                                fmt='o',color=COLORS[g],ms=4,capsize=2,lw=.9,
                                alpha=1 if row.status=='ok' else .35)
                    ax.scatter(100*row.shift_probability,yy,marker='D',s=19,
                               facecolors='white',edgecolors=COLORS[g],linewidths=.7)
        ax.set_title(title,fontsize=10,pad=14)
        ax.set_xlabel('Probability (%)' if metric!='exact_rate' else 'Equal-session mean rate')
        ax.axvline(0,color='#dddddd',lw=.7)
        ax.set_ylim(len(order)-.5,-.5)
        ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(order)),[BEH_LABELS[b] for b in order])
    handles=[plt.Line2D([],[],marker='o',lw=0,color=COLORS[g],label=g) for g in GENOTYPES]
    handles+=[plt.Line2D([],[],marker='D',lw=0,markerfacecolor='white',color='#777777',label='Local-shift baseline')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3)
    fig.suptitle(f'Social behavior and dyad vocalization | {role}\n'
                 'Raw probability ranking; animals weighted equally within genotype',fontsize=12)
    save(fig,folder,f'probability_ranking_{role}')


def enrichment(folder:Path,stats:pd.DataFrame) -> None:
    """Distinguish common vocalization contexts from excess over local chance."""
    from social_call_probability import ALPHA
    order=ranked_order(stats,'around_1s')
    fig,axes=plt.subplots(1,2,figsize=(11,7.4),layout='constrained',sharey=True)
    for ax,metric,title in zip(axes,['during_100ms','around_1s'],
                              ['Excess during behavior','Excess within ±1 s of onset']):
        rows=stats[stats.primary&(stats.metric==metric)].set_index('behavior')
        for i,b in enumerate(order):
            r=rows.loc[b]
            color=COLORS['WT'] if r.difference>=0 else COLORS['HET']
            x,lo,hi=100*r.difference,100*r.difference_ci_low,100*r.difference_ci_high
            if np.isfinite(x):
                ax.errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',color=color,
                            ms=4,capsize=2,lw=.9,alpha=1 if r.status=='ok' else .3)
                if r.q_bh<ALPHA:ax.annotate('*',(x,i),xytext=(6,0),textcoords='offset points',va='center',fontweight='bold')
        ax.axvline(0,color='#888888',lw=.8,ls=':')
        ax.set_title(title,pad=14)
        ax.set_xlabel('Observed minus local-shift probability (percentage points)')
        ax.set_ylim(len(order)-.5,-.5)
        ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(order)),[BEH_LABELS[b] for b in order])
    fig.suptitle('Which associations exceed local calling propensity?\n'
                 f'Combined estimate: 50% WT + 50% Het | * BH q < {ALPHA:g}, one 26-test family',fontsize=12)
    save(fig,folder,'probability_above_local_baseline')


def curve_summary(curves:pd.DataFrame) -> pd.DataFrame:
    """Bootstrap equal-session peri-onset curves, preserving episode clustering."""
    from social_call_probability import N_BOOT,SEED,aggregate,MIN_SESSIONS,MIN_EVENTS
    rows=[]
    for (role,behavior),table in curves.groupby(['role','behavior'],sort=False):
        labels=table[['animal_id','genotype','events']].drop_duplicates().set_index('animal_id')
        table=table[table.events>0]
        if table.empty:continue
        obs=table.pivot(index='animal_id',columns='lag_s',values='probability')
        exp=table.pivot(index='animal_id',columns='lag_s',values='shift_probability').reindex(obs.index)
        labels=labels.loc[obs.index]
        # Smoothing is a mean of five 100ms probabilities, not P(any call/500ms).
        n_lag=obs.shape[1]//SMOOTH_BINS
        values=obs.to_numpy().reshape(len(obs),n_lag,SMOOTH_BINS).mean(axis=2)
        expected=exp.to_numpy().reshape(len(obs),n_lag,SMOOTH_BINS).mean(axis=2)
        lags=np.asarray(obs.columns).reshape(n_lag,SMOOTH_BINS).mean(axis=1)
        g=labels.genotype.to_numpy()
        for scope in ['combined','WT','HET']:
            use=np.ones(len(obs),bool) if scope=='combined' else g==scope
            if not use.any() or (scope=='combined' and len(np.unique(g[use]))<2):continue
            v,e,gs=values[use],expected[use],g[use]
            mean=aggregate(v.T,gs,scope)
            base=aggregate(e.T,gs,scope)
            rng=np.random.default_rng(SEED+sum(map(ord,role+behavior+scope)))
            idx=np.tile(np.arange(len(v)),(N_BOOT,1))
            for genotype in np.unique(gs):
                pos=np.flatnonzero(gs==genotype)
                idx[:,pos]=rng.choice(pos,size=(N_BOOT,len(pos)),replace=True)
            bootstrap=aggregate(v[idx].transpose(0,2,1),gs,scope)
            low,high=np.quantile(bootstrap,[.025,.975],axis=0)
            n_events=int(labels.loc[obs.index[use],'events'].sum())
            support=use.sum()>=MIN_SESSIONS and n_events>=MIN_EVENTS
            for k,lag in enumerate(lags):
                rows.append(dict(role=role,behavior=behavior,scope=scope,lag_s=lag,
                    probability=mean[k],ci_low=low[k],ci_high=high[k],shift_probability=base[k],
                    n_sessions=int(use.sum()),n_events=n_events,supported=support))
    return pd.DataFrame(rows)


def peri_plot(folder:Path,summary:pd.DataFrame,stats:pd.DataFrame) -> None:
    """Show the full fixed behavior set, with before/after timing as descriptive curves."""
    order=ranked_order(stats,'around_1s')
    fig,axes=plt.subplots(4,4,figsize=(13,10.8),layout='constrained')
    for ax,b in zip(axes.flat,order):
        for g in GENOTYPES:
            t=summary[(summary.role=='dyad')&(summary.behavior==b)&(summary.scope==g)].sort_values('lag_s')
            if t.empty:continue
            supported=bool(t.supported.iloc[0])
            ax.plot(t.lag_s,100*t.probability,color=COLORS[g],lw=1.2,alpha=1 if supported else .3)
            if supported:ax.fill_between(t.lag_s,100*t.ci_low,100*t.ci_high,color=COLORS[g],alpha=.14,lw=0)
            ax.plot(t.lag_s,100*t.shift_probability,color=COLORS[g],lw=.8,ls=':',alpha=1 if supported else .3)
        info=summary[(summary.role=='dyad')&(summary.behavior==b)&(summary.scope=='combined')]
        counts=[]
        for g in GENOTYPES:
            sub=summary[(summary.role=='dyad')&(summary.behavior==b)&(summary.scope==g)]
            if len(sub):counts.append(f'{g} n={int(sub.n_sessions.iloc[0])}'+('†' if not sub.supported.iloc[0] else ''))
        label=', '.join(counts) if counts else 'No complete episodes'
        ax.set_title(BEH_LABELS[b]+'\n'+label,fontsize=8.5,pad=8)
        ax.axvline(0,color='#777777',lw=.8,ls='--')
        ax.set_xlim(-5,5);ax.set_ylim(bottom=0)
        ax.set_xlabel('Time from onset (s)',fontsize=7.5)
        ax.set_ylabel('Call probability / 100 ms (%)',fontsize=7.5)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(order):]:ax.axis('off')
    handles=[plt.Line2D([],[],color=COLORS[g],label=g) for g in GENOTYPES]
    handles+=[plt.Line2D([],[],color='#777777',ls=':',label='Local-shift baseline')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3)
    fig.suptitle('Vocalization around the onset of every social behavior\n'
                 'Mean 100ms probabilities smoothed over 500ms; † faint curves = insufficient support',fontsize=12)
    save(fig,folder,'all_social_peri_onset_probability')


def sensitivity_plot(folder:Path,stats:pd.DataFrame,sensitivity:pd.DataFrame) -> None:
    """Benchmark local time scales and episode definitions without choosing a winner."""
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    choices=[('block_30s','#888888','o'),('block_120s','#a393bb','s'),
             ('strict_episodes',COLORS['HET'],'^'),('include_low_coverage',COLORS['WT'],'D'),
             ('exclude_reused_stim','#58936a','x')]
    for ax,metric,title in zip(axes,['during_100ms','around_1s'],['During behavior','Within ±1 s of onset']):
        for name,color,marker in choices:
            t=sensitivity[(sensitivity.sensitivity==name)&(sensitivity.metric==metric)]
            ax.scatter(100*t.primary_difference,100*t.difference,color=color,marker=marker,
                       s=25,alpha=.75,label=name.replace('_',' '))
        limits=np.array([*ax.get_xlim(),*ax.get_ylim()]);lo,hi=limits.min(),limits.max()
        ax.plot([lo,hi],[lo,hi],color='#aaaaaa',lw=.8,ls=':')
        ax.set_xlabel('Primary excess probability (percentage points)')
        ax.set_ylabel('Sensitivity excess probability (percentage points)')
        ax.set_title(title)
    axes[1].legend(fontsize=7,loc='best')
    fig.suptitle('Benchmark against fixed alternative temporal settings',fontsize=12)
    save(fig,folder,'local_shift_and_episode_sensitivity')


def validate(out:Path,stats:pd.DataFrame,ps:pd.DataFrame,curves:pd.DataFrame,raw:pd.DataFrame,qa:pd.DataFrame):
    """Validate denominators, probabilities, correction families and observation units."""
    checks=[]
    def check(name,ok,value):
        checks.append(dict(check=name,passed=bool(ok),value=str(value)))
        if not ok:raise AssertionError(f'{name}: {value}')
    check('recordings_unique',len(qa)==24 and qa.animal_id.nunique()==24,len(qa))
    from social_call_probability import MIN_COVERAGE,xcorr,window_hit
    check('eligible_recordings',int(qa.eligible.sum())==int((qa.coverage>=MIN_COVERAGE).sum()),int(qa.eligible.sum()))
    check('primary_family',int(stats.primary.sum())==26,int(stats.primary.sum()))
    check('secondary_families',stats.groupby('family').size().to_dict()==
          {'actor_secondary_216':216,'before_after_secondary_78':78,'genotype_secondary_52':52,'primary_26':26},stats.groupby('family').size().to_dict())
    check('probability_bounds',stats.observed_probability.dropna().between(-1e-9,1+1e-9).all(),len(stats))
    check('baseline_bounds',stats.shift_probability.dropna().between(-1e-9,1+1e-9).all(),len(stats))
    check('window_hits_not_more_than_events',(ps.numerator<=ps.denominator).all(),len(ps))
    check('no_missing_behavior_zero_imputation',ps.loc[ps.denominator==0,'observed_probability'].isna().all(),'undefined when no exposure')
    check('calls_not_added', (qa.calls_in_complete_bins<=qa.total_calls).all(),qa.calls_in_complete_bins.sum())
    check('occupied_duration_not_more_than_exposure',(raw.occupied_time_s<=raw.exposure_s+1e-8).all(),len(raw))
    check('fdr_not_smaller_than_p',(stats.loc[stats.p.notna(),'q_bh']>=stats.loc[stats.p.notna(),'p']-1e-12).all(),'all families')
    for family,table in stats.groupby('family'):
        table=table[table.p.notna()].sort_values('p')
        check('bh_monotone_'+family,(np.diff(table.q_bh)>=-1e-12).all(),len(table))
    # Validate nonlinear any-call event windows against direct enumeration of
    # every legal cyclic shift, not only a linear correlation identity.
    binary=np.zeros(64,bool);binary[[1,15,16,33,56]]=True
    onsets=np.zeros((1,64));onsets[0,[16,32,47]]=1
    hits=window_hit(binary,-10,10)
    table=xcorr(onsets,hits.astype(float))[0]
    manual=[]
    for shift in range(64):
        rotated=np.roll(binary,-shift)
        manual.append(sum(rotated[i-10:i+10].any() for i in [16,32,47]))
    check('nonlinear_shift_probability_matches_direct',np.allclose(table,manual),'all64 shifts')
    # Under the finite uniform shift group, the exact two-sided test cannot
    # reject more than alpha of the group orbit, including tied statistics.
    deviation=np.abs(table-table.mean())
    exact_p=(deviation[None,:]>=deviation[:,None]-1e-12).mean(axis=1)
    check('exact_shift_orbit_type1_control',(exact_p<=.05).mean()<=.05,'uniform64offset orbit')
    expected=.5*ps[ps.metric=='before_1s'].denominator.sum()+.5*ps[ps.metric=='after_1s'].denominator.sum()
    check('same_before_after_event_denominators',ps[ps.metric=='around_1s'].denominator.sum()==expected,int(expected))
    main=stats[(stats.metric=='around_1s')&(stats.role=='dyad')]
    before=stats[(stats.metric=='before_1s')&(stats.role=='dyad')]
    after=stats[(stats.metric=='after_1s')&(stats.role=='dyad')]
    joined=main.merge(before,on=['behavior','scope'],suffixes=('_around','_before')).merge(
        after[['behavior','scope','observed_probability']],on=['behavior','scope'])
    both=joined.observed_probability_before+joined.observed_probability-joined.observed_probability_around
    check('union_probability_identity',both.dropna().between(-1e-9,1+1e-9).all(),'before+after-around equals intersection')
    pd.DataFrame(checks).to_csv(out/'validation.csv',index=False)


def report(out:Path,stats:pd.DataFrame,ps:pd.DataFrame,raw:pd.DataFrame,qa:pd.DataFrame,sensitivity:pd.DataFrame):
    """Explain observed probability, local enrichment, timing and ranking uncertainty."""
    from social_call_probability import N_SHIFT,N_BOOT,ALPHA,MIN_STATE_TIME_S,MIN_SESSIONS,MIN_EVENTS
    primary=stats[stats.primary]
    around=primary[primary.metric=='around_1s'].sort_values('observed_probability',ascending=False)
    during=primary[primary.metric=='during_100ms'].sort_values('observed_probability',ascending=False)
    hits=primary[primary.q_bh<ALPHA]
    lines=['# Probabilité des vocalisations autour des comportements sociaux','',
        f'Analyse principale : {int(qa.eligible.sum())} sessions admissibles sur 24, fenêtre sociale 300-900 s.',
        f'{len(hits)} associations temporelles sur 26 tests principaux dépassent le seuil BH q<{ALPHA:g}.','',
        '## Deux probabilités différentes','',
        '- Pendant : probabilité d’au moins un début d’appel dans un intervalle de 100 ms où le comportement occupe au moins la moitié du temps.',
        '- Autour du début : probabilité d’au moins un appel dans [-1,+1) s autour du début d’un épisode. '
        'Le début est aligné au premier intervalle de 100 ms qui le contient.',
        '- Le taux exact d’appels pendant un comportement divise les débuts d’appels observés à l’intérieur des images actives par leur durée active réelle.',
        '- Ces probabilités portent sur les appels de la dyade. Résidente/partenaire désignent l’acteur du comportement, sans identifier l’émetteur vocal.','',
        '## Classement : au moins un appel à ±1 s du début','',
        '| Comportement | Sessions | Épisodes complets | P observée (%) | IC 95 % | P décalages (%) | Excès (points) | q BH |',
        '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in around.itertuples():
        suffix='' if r.status=='ok' else ' (support insuffisant)'
        lines.append(f'| {BEH_LABELS[r.behavior]}{suffix} | {r.n_sessions} | {r.n_events_or_bins} | '
            f'{number(100*r.observed_probability)} | [{number(100*r.ci_low)}; {number(100*r.ci_high)}] | '
            f'{number(100*r.shift_probability)} | {number(100*r.difference)} | {number(r.q_bh)} |')
    lines+=['','## Classement : vocalisation pendant le comportement','',
        '| Comportement | P appel / 100 ms (%) | P décalages (%) | Taux exact moyen (appels/min) | q BH |',
        '|---|---:|---:|---:|---:|']
    for r in during.itertuples():
        lines.append(f'| {BEH_LABELS[r.behavior]} | {number(100*r.observed_probability)} | '
            f'{number(100*r.shift_probability)} | {number(r.exact_call_rate_per_min)} | {number(r.q_bh)} |')
    lines+=['','## WT et HET : mêmes mesures, groupes séparés','',
        '| Comportement | WT : P à ±1 s (%) | HET : P à ±1 s (%) | WT : appels/min pendant | HET : appels/min pendant |',
        '|---|---:|---:|---:|---:|']
    for behavior in around.behavior:
        a=stats[(stats.role=='dyad')&(stats.behavior==behavior)&(stats.metric=='around_1s')].set_index('scope')
        d=stats[(stats.role=='dyad')&(stats.behavior==behavior)&(stats.metric=='during_100ms')].set_index('scope')
        wt=f'{number(100*a.loc["WT","observed_probability"])} (n={a.loc["WT","n_sessions"]})'+('†' if a.loc['WT','status']!='ok' else '')
        het=f'{number(100*a.loc["HET","observed_probability"])} (n={a.loc["HET","n_sessions"]})'+('†' if a.loc['HET','status']!='ok' else '')
        lines.append(f'| {BEH_LABELS[behavior]} | {wt} | '
            f'{het} | {number(d.loc["WT","exact_call_rate_per_min"])} | '
            f'{number(d.loc["HET","exact_call_rate_per_min"])} |')
    lines+=['','† Support insuffisant : estimation descriptive, pas de test concluant pour ce sous-groupe.','',
        '## Avant ou après le début du comportement ?','',
        '| Comportement | P dans [-1,0) s (%) | P dans [0,+1) s (%) | q avant | q après |',
        '|---|---:|---:|---:|---:|']
    for behavior in around.behavior:
        before=stats[(stats.role=='dyad')&(stats.scope=='combined')&(stats.behavior==behavior)&(stats.metric=='before_1s')].iloc[0]
        after=stats[(stats.role=='dyad')&(stats.scope=='combined')&(stats.behavior==behavior)&(stats.metric=='after_1s')].iloc[0]
        lines.append(f'| {BEH_LABELS[behavior]} | {number(100*before.observed_probability)} | '
                     f'{number(100*after.observed_probability)} | {number(before.q_bh)} | {number(after.q_bh)} |')
    blocktests=pd.read_csv(out/'block_sensitivity_tests.csv')
    lines+=['','## Robustesse aux échelles temporelles locales','',
        'Les 52 tests secondaires (13 comportements × deux probabilités × deux échelles) partagent une correction BH.',
        'Les interactions anogénitales et les approches sont enrichies à ±1 s aux trois échelles. Le suivi est sensible au choix de 30 s.',
        'La poursuite a la plus forte probabilité brute, mais son enrichissement ne passe aucune des trois corrections BH.','',
        '| Comportement | q BH, blocs 30 s | q BH, blocs 60 s (principal) | q BH, blocs 120 s |',
        '|---|---:|---:|---:|']
    for behavior in ['nose2anogenital','following','chasing','approach']:
        b=blocktests[(blocktests.behavior==behavior)&(blocktests.metric=='around_1s')].set_index('block_s')
        a=around[around.behavior==behavior].iloc[0]
        lines.append(f'| {BEH_LABELS[behavior]} | {number(b.loc[30.,"q_bh"])} | {number(a.q_bh)} | {number(b.loc[120.,"q_bh"])} |')
    lines+=['','## Méthode et hypothèses','',
        '- Durée analysée : 300-900 s; seules les images où les deux animaux sont observés sont utilisables. '
        'Les intervalles de 100 ms doivent être entièrement observés, y compris les portions d’images à leurs limites.',
        '- Seuil d’admissibilité des sessions fixé à 95 % de couverture. La session 31101 atteint 94,4 % et est exclue du principal, puis réintégrée en sensibilité.',
        '- Les marqueurs dyadiques sont l’union des deux acteurs. Les doublons frame/acteur sont agrégés une fois.',
        '- Épisodes : fusion des interruptions observées de ≤0,2 s, durée active minimale 0,2 s. '
        'Les interruptions ne sont pas ajoutées à la durée active; les trous de suivi interrompent les épisodes.',
        '- Les épisodes commencés à la limite de fenêtre ou juste après un trou de suivi sont censurés et exclus. '
        'La fenêtre ±1 s doit être entièrement observée et située dans le même bloc temporel local.',
        '- Null : décalage cyclique uniforme des appels, incluant le décalage nul, dans chaque segment observé '
        'à l’intérieur des blocs de 60 s. Le nombre d’appels local est préservé. Le même décalage est partagé entre comportements.',
        '- Hypothèse : stationnarité locale et invariance aux décalages dans ces segments. '
        'La structure des salves est préservée à l’intérieur des segments, mais pas à leurs frontières. '
        'Les blocs de 30 et 120 s sont testés dans une famille secondaire commune de 52 tests, sans sélection du réglage le plus favorable.',
        f'- La probabilité attendue est calculée en énumérant tous les décalages de chaque segment. '
        f'Le test bilatéral compare l’écart observé à {N_SHIFT:,} combinaisons de décalages indépendantes par session.',
        '- Pondération : moyenne des probabilités par session dans chaque génotype. Le combiné standardise à 50 % WT et 50 % HET, '
        'pour éviter que les souris très vocales ou le nombre de sessions d’un groupe dominent le classement.',
        f'- Support minimal des tests : {MIN_SESSIONS} sessions; {MIN_EVENTS} épisodes complets pour les fenêtres autour des débuts. '
        f'Pour pendant, chaque session doit contribuer au moins {MIN_STATE_TIME_S:g} s de comportement en intervalles complets. '
        'Le test combiné requiert également au moins deux sessions par génotype.',
        f'- IC : {N_BOOT:,} bootstraps des sessions, stratifiés par génotype. Les épisodes d’une session restent ensemble. '
        'Les IC sont ponctuels, sans correction simultanée; [0,0] lorsque tous les résultats sont nuls ne démontre pas une probabilité réelle exactement nulle.',
        '- Multiplicité : 26 tests principaux (13 comportements × deux probabilités), 52 tests secondaires WT/HET, '
        '78 tests secondaires avant/après, et 216 tests secondaires des acteurs. BH, BY et Holm sont tous exportés.',
        '- Les courbes ±5 s utilisent des fenêtres complètes et une moyenne de cinq probabilités /100 ms. '
        'Il ne s’agit pas de la probabilité d’au moins un appel dans 500 ms. Les courbes sont descriptives, sans test de chaque point.','',
        '## Limites pour interpréter le classement','',
        '- Les comportements se chevauchent. Un appel peut donc apparaître dans plusieurs comportements; les pourcentages ne doivent pas totaliser 100 %.',
        '- La plus grande probabilité brute n’implique pas l’enrichissement le plus élevé au-dessus des décalages locaux.',
        '- Un classement sur ces données ne prouve pas une différence statistique entre le premier et le deuxième comportement. '
        'Les IC et les tests concernent l’écart au niveau local de vocalisation, pas toutes les comparaisons comportement contre comportement.',
        '- Les labels comportementaux sont des règles géométriques sur le suivi vidéo, sans annotation manuelle indépendante.',
        '- Deux partenaires stimulus sont réutilisés entre sessions; une dépendance résiduelle peut affecter les IC et les tests entre dyades. '
        'Une sensibilité descriptive retire les quatre résidentes concernées.',
        '- Les appels sont détectés par le modèle existant. Les sorties brutes et les métadonnées ne sont pas modifiées.','',
        '## Fichiers','',
        '- `probability_statistics.csv` : estimations, IC, niveaux attendus, enrichissements, effectifs, p et corrections pour chaque comportement/rôle/groupe.',
        '- `session_probabilities.csv`, `raw_state_rates.csv` : toutes les mesures par session et leurs dénominateurs.',
        '- `session_peri_onset_curves.csv`, `peri_onset_curve_statistics.csv` : courbes par session et agrégation avec IC.',
        '- `sensitivity_statistics.csv` : blocs 30/120 s, épisodes stricts, réintégration de faible couverture, exclusion des partenaires réutilisés.',
        '- `block_sensitivity_tests.csv`, `block_sensitivity_provenance.json` : tests secondaires aux échelles 30/120 s et leur cache.',
        '- `source_qa.csv`, `validation.csv`, `provenance.json`, `traces.npz` : suivi, sources, hypothèses et cache reproductible.',
        '- `figures/` : chaque figure est exportée en PNG, PDF et SVG.','',
        '## Références méthodologiques','',
        '- [Louis et al. : non-stationnarité et génération de traces substitutives](https://pmc.ncbi.nlm.nih.gov/articles/PMC2972681/).',
        '- [Choix des méthodes de substitution et contrôle des faux positifs](https://pmc.ncbi.nlm.nih.gov/articles/PMC3090783/).',
        '- [SciPy : rééchantillonnage apparié et intervalles bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).','']
    (out/'RAPPORT_PROBABILITE_VOCALISATIONS.md').write_text('\n'.join(lines),encoding='utf-8')
    stats[stats.role=='dyad'].to_csv(out/'dyad_behavior_ranking.csv',index=False)


def make_outputs(out:Path,stats:pd.DataFrame,ps:pd.DataFrame,curves:pd.DataFrame,
                 raw:pd.DataFrame,qa:pd.DataFrame,sensitivity:pd.DataFrame):
    """Validate and render every requested probability view from immutable numeric results."""
    configure();validate(out,stats,ps,curves,raw,qa)
    from social_call_probability import block_sensitivity_tests
    block_sensitivity_tests(out,qa)
    folder=out/'figures';folder.mkdir(exist_ok=True)
    for role in ['dyad','resident','partner']:probabilities(folder,stats,role)
    enrichment(folder,stats)
    summary=curve_summary(curves)
    summary.to_csv(out/'peri_onset_curve_statistics.csv',index=False)
    peri_plot(folder,summary,stats)
    sensitivity_plot(folder,stats,sensitivity)
    report(out,stats,ps,raw,qa,sensitivity)
