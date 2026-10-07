"""White-background scientific figures and French report for correlation results.

Numeric calculations are owned by social_vocal_correlations.py. Editing style
here does not invalidate its source-stamped numerical cache.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from social_vocal_correlations import ALPHA

# Editable figure parameters and vocabulary.
COLORS={'WT':'#32799D','HET':'#DF892E'}
VIRUS_MARKERS={'ChrimsonR':'o','tdTomato':'s'}
DPI=320
FORMATS=('png','pdf','svg')
FONT=9
BEH_LABELS={
    'nose2anogenital':'Anogenital investigation','nose2body':'Body investigation',
    'nose2nose':'Nose-to-nose contact','oriented_toward':'Orientation toward partner',
    'following':'Following','chasing':'Chasing','approach':'Approach',
    'sidebyside':'Side-by-side','sidereside':'Side contact',
    'withdrawal_from_partner':'Withdrawal from partner',
    'withdrawal_after_contact':'Withdrawal after contact','escape':'Escape','fighting':'Fighting',
}
B_LABELS={'cumulative_duration_s':'Cumulative active duration (s)',
          'mean_episode_duration_s':'Mean active episode duration (s)',
          'episode_rate_per_min':'Social episodes / min'}
V_LABELS={'call_rate_per_min':'Calls / min','vocal_output_s_per_min':'Vocal output (s / min)',
          'median_call_duration_ms':'Median call duration (ms)',
          'median_frequency_khz':'Median frequency (kHz)','median_bandwidth_khz':'Median bandwidth (kHz)',
          'bout_rate_per_min':'Vocal bouts / min','mean_calls_per_bout':'Mean calls / bout',
          'class_entropy_bits':'Class diversity (bits)'}
V_SHORT=['Calls\n/ min','Output\n(s/min)','Length\n(ms)','Freq.\n(kHz)',
         'Bandw.\n(kHz)','Bouts\n/ min','Calls\n/ bout','Entropy\n(bits)']
SCOPES=['pooled','adjusted_genotype','adjusted_genotype_virus','WT','HET']
SCOPE_LABEL={'pooled':'Pooled, unadjusted','adjusted_genotype':'Adjusted: genotype',
             'adjusted_genotype_virus':'Adjusted: genotype + virus','WT':'WT only','HET':'Het only'}


def configure() -> None:
    """Set a compact, editable publication style consistent with the prior figure."""
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':FONT,'axes.titlesize':10,'axes.titleweight':'bold',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
        'axes.edgecolor':'#444444','axes.facecolor':'white','figure.facecolor':'white',
        'legend.frameon':False,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def save(fig,folder:Path,stem:str) -> None:
    """Export each canonical panel set to all three requested formats."""
    for ext in FORMATS:
        fig.savefig(folder/f'{stem}.{ext}',dpi=DPI,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def number(value:float,precision:int=3) -> str:
    """Print unavailable statistics explicitly and use scientific notation when small."""
    if not np.isfinite(value): return 'NA'
    return f'{value:.{precision}g}'


def heatmap(ax,rows:pd.DataFrame,row_keys:list,col_keys:list,row_field:str,
            title:str,show_stars:bool=True):
    """Display every planned association, with FDR significance and undefined cells."""
    values=np.full((len(row_keys),len(col_keys)),np.nan)
    significant=np.zeros_like(values,bool)
    for i,key in enumerate(row_keys):
        for j,vm in enumerate(col_keys):
            sub=rows[(rows[row_field]==key)&(rows.vocal_metric==vm)]
            if len(sub)==1:
                values[i,j]=sub.rho.iloc[0]
                significant[i,j]=sub.q_bh.iloc[0]<ALPHA
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#eeeeee')
    im=ax.imshow(values,cmap=cmap,vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(len(col_keys)),V_SHORT)
    ax.tick_params(axis='both',length=0)
    ax.set_yticks(range(len(row_keys)),[BEH_LABELS.get(key,key) for key in row_keys])
    ax.set_title(title,pad=12)
    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            value=values[i,j]
            text=f'{value:.2f}' if np.isfinite(value) else 'NA'
            if significant[i,j] and show_stars:text+='*'
            ax.text(j,i,text,ha='center',va='center',fontsize=7.5,
                    color='white' if np.isfinite(value) and abs(value)>.62 else '#222222')
    for spine in ax.spines.values():spine.set_visible(False)
    return im


def social_maps(folder:Path,results:pd.DataFrame) -> None:
    """Show all 312 primary dyad tests and separate actor sensitivity families."""
    from social_vocal_correlations import SOCIAL,BEHAVIOR_METRICS,VOCAL_METRICS
    for role in ['dyad','resident','partner']:
        rows=results[(results.role==role)&(results.scope=='adjusted_genotype_virus')]
        behaviors=[b for b in SOCIAL if b in set(rows.behavior)]
        fig,axes=plt.subplots(1,3,figsize=(18,7.5 if role=='dyad' else 6),layout='constrained')
        for ax,bm in zip(axes,BEHAVIOR_METRICS):
            im=heatmap(ax,rows[rows.behavior_metric==bm],behaviors,VOCAL_METRICS,'behavior',B_LABELS[bm])
        cb=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.72,pad=.015)
        cb.set_label('Conditional rank correlation')
        label={'dyad':'All social behaviors, dyad union','resident':'Resident as actor, sensitivity',
               'partner':'Partner as actor, sensitivity'}[role]
        if role!='dyad':label+=' | one combined 432-test actor family'
        fig.suptitle(label+f'\nAdjusted for genotype + virus | * BH q < {ALPHA:g}; NA = insufficient data',fontsize=12)
        save(fig,folder,f'all_social_{role}_heatmap')


def anogenital_map(folder:Path,results:pd.DataFrame) -> None:
    """Contrast unadjusted, genotype-conditioned and within-genotype associations."""
    from social_vocal_correlations import BEHAVIOR_METRICS,VOCAL_METRICS
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    rows=results[(results.role=='dyad')&(results.behavior=='nose2anogenital')]
    for ax,bm in zip(axes,BEHAVIOR_METRICS):
        im=heatmap(ax,rows[rows.behavior_metric==bm],SCOPES,VOCAL_METRICS,'scope',B_LABELS[bm])
        ax.set_yticklabels([SCOPE_LABEL[s] for s in SCOPES])
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,pad=.015,label='Rank correlation')
    fig.suptitle(f'Anogenital investigation vs dyad vocalization\n* BH q < {ALPHA:g} in the complete family for each scope',fontsize=12)
    save(fig,folder,'anogenital_all_parameters')


def scatter(ax,behavior:pd.DataFrame,vocal:pd.DataFrame,test:pd.Series) -> None:
    """Plot raw animal/session measurements, with conditional statistics labeled."""
    b=behavior[(behavior.role==test.role)&(behavior.behavior==test.behavior)]
    joined=b.merge(vocal,on=['animal_id','genotype','virus'],suffixes=('','_vocal'))
    joined=joined[joined.eligible&joined[test.behavior_metric].notna()&joined[test.vocal_metric].notna()]
    for (genotype,virus),table in joined.groupby(['genotype','virus']):
        ax.scatter(table[test.behavior_metric],table[test.vocal_metric],
                   color=COLORS[genotype],marker=VIRUS_MARKERS.get(virus,'o'),
                   s=33,edgecolor='white',linewidth=.55,zorder=3)
    ax.set_xlabel(B_LABELS[test.behavior_metric],fontsize=8)
    ax.set_ylabel(V_LABELS[test.vocal_metric],fontsize=8)
    ax.set_title(f'{BEH_LABELS[test.behavior]}\n'
                 f'conditional r = {number(test.rho,2)}, q = {number(test.q_bh,2)}, n = {test.n}',
                 fontsize=9,pad=10)
    ax.margins(.12)


def legend(fig) -> None:
    """Explain genotype colors and virus markers consistently across scatter plots."""
    handles=[Line2D([],[],marker='o',lw=0,color=COLORS[g],label=g) for g in ['WT','HET']]
    handles += [Line2D([],[],marker=m,lw=0,color='#777777',label=v) for v,m in VIRUS_MARKERS.items()]
    fig.legend(handles=handles,loc='outside lower center',ncol=4)


def scatter_sets(folder:Path,behavior:pd.DataFrame,vocal:pd.DataFrame,results:pd.DataFrame) -> None:
    """Use a fixed anogenital set, then clearly label data-selected exploratory examples."""
    from social_vocal_correlations import BEHAVIOR_METRICS
    primary=results[results.primary]
    fig,axes=plt.subplots(3,3,figsize=(12,10.5),layout='constrained')
    for i,bm in enumerate(BEHAVIOR_METRICS):
        for j,vm in enumerate(['call_rate_per_min','vocal_output_s_per_min','median_call_duration_ms']):
            test=primary[(primary.behavior=='nose2anogenital')&(primary.behavior_metric==bm)&(primary.vocal_metric==vm)].iloc[0]
            scatter(axes[i,j],behavior,vocal,test)
    fig.suptitle('Anogenital investigation: three fixed vocal outcomes\n'
                 'Each point is one dyad; statistics adjust for genotype and virus',fontsize=12)
    legend(fig);save(fig,folder,'anogenital_scatterplots')
    selected=primary[primary.status=='ok'].sort_values(['q_bh','p']).head(6)
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.7),layout='constrained')
    for ax,(_,test) in zip(axes.flat,selected.iterrows()):scatter(ax,behavior,vocal,test)
    fig.suptitle('Exploratory examples: six smallest primary adjusted p values\n'
                 'Selection is descriptive; the full 312-test family is exported',fontsize=12)
    legend(fig);save(fig,folder,'exploratory_association_scatterplots')


def adjustment_benchmark(folder:Path,results:pd.DataFrame,sensitivity:pd.DataFrame) -> None:
    """Benchmark pooled correlations against covariate adjustment and robustness settings."""
    keys=['behavior','behavior_metric','vocal_metric']
    primary=results[results.primary]
    pooled=results[(results.role=='dyad')&(results.scope=='pooled')]
    table=primary.merge(pooled,on=keys,suffixes=('_adjusted','_pooled'))
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    colors=np.where(table.behavior=='nose2anogenital',COLORS['HET'],'#a6a9ac')
    axes[0].scatter(table.rho_pooled,table.rho_adjusted,c=colors,s=17,alpha=.7)
    axes[0].plot([-1,1],[-1,1],color='#aaaaaa',lw=.8,ls=':')
    axes[0].axhline(0,color='#cccccc',lw=.7);axes[0].axvline(0,color='#cccccc',lw=.7)
    axes[0].set(xlim=(-1,1),ylim=(-1,1),xlabel='Pooled rank correlation',
                ylabel='Correlation adjusted for genotype + virus',title='Adjustment benchmark')
    axes[0].text(.03,.96,'Orange: anogenital pairs',transform=axes[0].transAxes,va='top',fontsize=8)
    for name,color,marker in [('episode_gap_0.2s','#777777','o'),
                             ('vocal_bout_gap_0.5s','#a893ba','s'),
                             ('exclude_reused_stim_partners',COLORS['HET'],'^'),
                             ('adjust_motion_and_distance',COLORS['WT'],'D'),
                             ('include_low_coverage_session','#548C66','x')]:
        subset=sensitivity[sensitivity.sensitivity==name]
        axes[1].scatter(subset.primary_rho,subset.rho,s=14,alpha=.45,
                        color=color,marker=marker,label=name.replace('_',' '))
    axes[1].plot([-1,1],[-1,1],color='#aaaaaa',lw=.8,ls=':')
    axes[1].set(xlim=(-1,1),ylim=(-1,1),xlabel='Primary conditional rank correlation',
                ylabel='Sensitivity coefficient (descriptive)',title='Parameter and confound sensitivity')
    axes[1].legend(loc='lower right',fontsize=6.5)
    fig.suptitle('All planned social-behavior / vocalization pairs',fontsize=12)
    save(fig,folder,'adjustment_and_sensitivity_benchmark')


def validation(out:Path,behavior:pd.DataFrame,vocal:pd.DataFrame,qa:pd.DataFrame,results:pd.DataFrame) -> pd.DataFrame:
    """Export explicit data and inferential QA checks, not just a pass message."""
    from social_vocal_correlations import SOCIAL,BEHAVIOR_METRICS,VOCAL_METRICS,N_PERM
    checks=[]
    def check(name,passed,value):
        checks.append(dict(check=name,passed=bool(passed),value=str(value)))
        if not passed:raise AssertionError(f'{name}: {value}')
    check('unique_session_ids',vocal.animal_id.nunique()==24 and len(vocal)==24,len(vocal))
    primary=results[results.primary]
    check('complete_primary_family',len(primary)==len(SOCIAL)*len(BEHAVIOR_METRICS)*len(VOCAL_METRICS),len(primary))
    check('no_duration_exceeds_exposure',(behavior.cumulative_duration_s<=behavior.exposure_s+1e-8).all(),'all rows')
    check('absent_behavior_mean_undefined',behavior.loc[behavior.episode_count==0,'mean_episode_duration_s'].isna().all(),'all absent rows')
    ok=results[results.status=='ok']
    check('rho_bounds',ok.rho.between(-1-1e-9,1+1e-9).all(),len(ok))
    check('p_bounds',ok.p.between(1/(N_PERM+1)-1e-12,1).all(),len(ok))
    check('adjustment_does_not_reduce_p',(ok.q_bh>=ok.p-1e-12).all(),'BH')
    check('matched_exposure_positive',(vocal.exposure_s>0).all(),vocal.exposure_s.min())
    check('source_genotypes_balanced',(vocal.groupby('genotype').size()==12).all(),vocal.groupby('genotype').size().to_dict())
    check('genotype_virus_balance',(vocal.groupby(['genotype','virus']).size()==6).all(),'4 groups of 6')
    pd.DataFrame(checks).to_csv(out/'validation.csv',index=False)
    return pd.DataFrame(checks)


def report(out:Path,behavior:pd.DataFrame,vocal:pd.DataFrame,qa:pd.DataFrame,
           results:pd.DataFrame,sensitivity:pd.DataFrame) -> None:
    """Write an honest French interpretation with all selection and limitations explicit."""
    from social_vocal_correlations import N_PERM,N_BOOT,MIN_COVERAGE,REUSED_STIM_RESIDENTS
    primary=results[results.primary]
    hits=primary[primary.q_bh<ALPHA].sort_values('q_bh')
    ago=primary[primary.behavior=='nose2anogenital']
    ago_hits=ago[ago.q_bh<ALPHA].sort_values('q_bh')
    best=ago.sort_values('p').head(6)
    lines=['# Corrélations entre interactions sociales et vocalisations','',
        '## Résultat principal','',
        f'{len(hits)} associations sur {len(primary)} comparaisons planifiées ont q BH < {ALPHA:g} après ajustement génotype + virus.',
        f'Pour les interactions anogénitales : {len(ago_hits)} sur {len(ago)}.',
        f'Avec la correction BY, robuste à une dépendance arbitraire des tests : {int((primary.q_by<ALPHA).sum())} associations.',
        f'Avec Holm (contrôle familial) : {int((primary.p_holm<ALPHA).sum())} associations.','',
        'Ces résultats sont exploratoires. Une corrélation ne démontre ni causalité ni émission par la souris résidente.','',
        '## Plan et mesures','',
        '- Unité : une session résidente-partenaire, 24 sessions, 12 WT et 12 HET.',
        '- Fenêtre : 300-900 s. Les débuts d’appels sont sélectionnés sur les images où les deux animaux sont observés. '
        'La durée vocale de ces appels est intersectée avec les intervalles effectivement observés. '
        'Les médianes acoustiques restent les caractéristiques des appels complets à début observé.',
        f'- Couverture minimale fixée : {100*MIN_COVERAGE:.0f} %. Sessions admissibles : {int(vocal.eligible.sum())}/24. '
        f'Couverture observée : {100*vocal.coverage.min():.2f} à {100*vocal.coverage.max():.2f} %.',
        f'- Appels exclus faute de couverture comportementale : {int(vocal.excluded_unobserved_calls.sum())}/{int(vocal.total_window_calls.sum())}.',
        '- Interaction dyadique = union des marqueurs des deux acteurs. Les contacts communs ne sont pas comptés deux fois.',
        '- Trois mesures : durée active cumulée, durée active moyenne par épisode, fréquence des épisodes par minute observée.',
        '- Huit mesures vocales : taux d’appels, durée vocale par minute, médianes de durée/fréquence/bande passante, taux de salves, appels par salve, diversité de classes (Shannon).',
        '- Épisode primaire : séquence continue de marqueurs vrais, sans fusion. Les trous de suivi interrompent un épisode. '
        'La sensibilité fusionne les interruptions observées de 0,2 s, sans ajouter ces interruptions à la durée active.',
        '- Salve vocale : appels successifs séparés de ≤ 0,25 s depuis la fin la plus tardive des appels précédents; '
        'les trous de suivi imposent une nouvelle salve. Sensibilité à 0,5 s.',
        '- Un comportement absent a une durée cumulée et une fréquence nulles, mais sa durée moyenne est indéfinie.',
        '- Les rôles résidente et partenaire forment une seule famille secondaire de 432 tests.','',
        '## Statistiques et hypothèses','',
        '- Coefficient principal : rangs moyens globaux, centrés dans chacune des quatre strates génotype × virus, puis corrélation des résidus.',
        f'- Test bilatéral : {N_PERM:,} permutations de la mesure vocale à l’intérieur de ces strates, p=(extrêmes+1)/(permutations+1).',
        '- Hypothèse nulle : indépendance conditionnelle et échangeabilité dans les strates. La normalité n’est pas requise. '
        'Les liens monotones sont visés; un lien en U pourrait être manqué.',
        '- Famille principale : toutes les 312 paires dyadiques, y compris anogénitales. Les paires non testables contribuent p=1 pour la correction. '
        'BH est complétée par BY et Holm; les familles secondaires sont identifiées dans les CSV.',
        '- Test non réalisé si n<10 ou si une mesure a moins de trois valeurs distinctes. Les rangs constants après ajustement sont signalés.',
        f'- IC : bootstrap apparié dans les strates, {N_BOOT:,} rééchantillonnages, percentile 95 %, intervalles ponctuels et non simultanés.',
        '- La suppression successive d’une session est calculée pour chaque test. Les réglages de sensibilité sont rapportés sans choisir le plus favorable.',
        '- Le mélange des génotypes peut produire une corrélation apparente; pooled, ajustement génotype seul et analyses WT/HET sont exportés.','',
        '## Interactions anogénitales : six plus petites p-values ajustées','',
        '| Mesure sociale | Mesure vocale | n | r conditionnel | IC 95 % | p | q BH |',
        '|---|---|---:|---:|---|---:|---:|']
    for r in best.itertuples():
        lines.append(f'| {B_LABELS[r.behavior_metric]} | {V_LABELS[r.vocal_metric]} | {r.n} | {number(r.rho)} | '
                     f'[{number(r.ci_low)}; {number(r.ci_high)}] | {number(r.p)} | {number(r.q_bh)} |')
    lines += ['','## Pourquoi le mélange des groupes change la conclusion anogénitale','',
              '| Mesure sociale | Mesure vocale | r brut | q brut | r ajusté | q ajusté |',
              '|---|---|---:|---:|---:|---:|']
    for bm in B_LABELS:
        for vm in ['call_rate_per_min','vocal_output_s_per_min']:
            r=ago[(ago.behavior_metric==bm)&(ago.vocal_metric==vm)].iloc[0]
            raw=results[(results.role=='dyad')&(results.behavior=='nose2anogenital')&
                        (results.behavior_metric==bm)&(results.vocal_metric==vm)&(results.scope=='pooled')].iloc[0]
            lines.append(f'| {B_LABELS[bm]} | {V_LABELS[vm]} | {number(raw.rho)} | {number(raw.q_bh)} | '
                         f'{number(r.rho)} | {number(r.q_bh)} |')
    lines += ['',f'## Associations sociales avec q BH < {ALPHA:g}','']
    if hits.empty:lines.append('Aucune association ne franchit le seuil corrigé. Les tendances nominales ne sont pas des résultats confirmés.')
    else:
        lines += ['| Comportement | Mesure sociale | Mesure vocale | n | r | q BH | q BY | Signe stable au retrait d’une session |',
                  '|---|---|---|---:|---:|---:|---:|---|']
        for r in hits.itertuples():
            lines.append(f'| {BEH_LABELS[r.behavior]} | {B_LABELS[r.behavior_metric]} | {V_LABELS[r.vocal_metric]} | '
                         f'{r.n} | {number(r.rho)} | {number(r.q_bh)} | {number(r.q_by)} | {r.loo_sign_stable} |')
    lines += ['','## Six premières tendances sociales, non confirmées si q dépasse le seuil','',
              '| Comportement | Mesure sociale | Mesure vocale | r | p | q BH |',
              '|---|---|---|---:|---:|---:|']
    for r in primary.sort_values('p').head(6).itertuples():
        lines.append(f'| {BEH_LABELS[r.behavior]} | {B_LABELS[r.behavior_metric]} | {V_LABELS[r.vocal_metric]} | '
                     f'{number(r.rho)} | {number(r.p)} | {number(r.q_bh)} |')
    actor=results[(results.role!='dyad')&(results.q_bh<ALPHA)].sort_values('p')
    lines += ['','## Analyses secondaires des acteurs','',
              f'{len(actor)} associations atteignent BH dans la famille commune des 432 tests résidente/partenaire. '
              'Ces résultats secondaires restent exploratoires, surtout s’ils ne résistent pas à BY.','',
              '| Acteur | Comportement | Mesure sociale | Mesure vocale | r | p | q BH | q BY |',
              '|---|---|---|---|---:|---:|---:|---:|']
    for r in actor.itertuples():
        lines.append(f'| {r.role} | {BEH_LABELS[r.behavior]} | {B_LABELS[r.behavior_metric]} | {V_LABELS[r.vocal_metric]} | '
                     f'{number(r.rho)} | {number(r.p)} | {number(r.q_bh)} | {number(r.q_by)} |')
    lines += ['','## Sensibilité du lien fréquence anogénitale / durée vocale','',
              '| Réglage | n | Coefficient conditionnel descriptif |',
              '|---|---:|---:|']
    focus=sensitivity[(sensitivity.behavior=='nose2anogenital')&
                      (sensitivity.behavior_metric=='episode_rate_per_min')&
                      (sensitivity.vocal_metric=='vocal_output_s_per_min')]
    for r in focus.itertuples():
        lines.append(f'| {r.sensitivity} | {r.n} | {number(r.rho)} |')
    lines += ['','## Limites et analyses de sensibilité','',
        '- Les marqueurs comportementaux proviennent de règles géométriques sur le suivi vidéo. Ils ne sont pas une annotation manuelle indépendante.',
        '- Les durées moyennes d’épisodes tronqués par une limite de fenêtre ou une perte de suivi peuvent être sous-estimées. '
        'Le nombre d’épisodes censurés est exporté pour chaque session/comportement.',
        f'- Deux partenaires sont réutilisés. Les sessions résidentes {", ".join(sorted(REUSED_STIM_RESIDENTS))} '
        'sont retirées dans une sensibilité descriptive; les IC/session et permutations ne modélisent pas une dépendance résiduelle entre ces dyades.',
        '- Le contrôle vitesse résidente + distance ajoute des covariables de rang en sensibilité descriptive; avec 24 sessions il ne peut exclure tous les facteurs confondants.',
        '- La réintégration descriptive de la session à faible couverture est exportée pour toutes les 312 paires principales. '
        'Elle ne change pas le seuil d’admissibilité de l’analyse principale.',
        '- Les p-values de permutation sont estimées par simulation; leur erreur standard Monte-Carlo est exportée. '
        'Un franchissement marginal du seuil, en particulier dans une analyse secondaire, doit être interprété avec prudence.',
        f'- La diversité vocale dépend du nombre d’appels. {int(vocal.entropy_low_call_count.sum())} sessions ont moins de 20 appels; '
        'leurs estimations de diversité et leurs caractéristiques acoustiques sont moins précises.',
        '- Les durées, fréquences, taux de salves et appels par salve sont mathématiquement dépendants. '
        'L’exploration complète et les corrections évitent de ne conserver que le paramètre le plus favorable.',
        '- Les données ont déjà servi à d’autres analyses de ce projet. Une confirmation nécessite des sessions indépendantes.',
        '- Ces corrélations portent sur les différences entre sessions. Une absence de corrélation entre sessions '
        'n’exclut pas un enrichissement des appels pendant un comportement à l’intérieur d’une session. '
        'Une alternative complémentaire est une analyse temporelle avec décalages circulaires préservant les salves et épisodes.',
        '', '## Fichiers','',
        '- `behavior_parameters.csv`, `vocal_parameters.csv` : toutes les valeurs par session.',
        '- `correlation_statistics.csv` : toutes les corrélations, p, corrections, IC, effectifs et diagnostics.',
        '- `sensitivity_statistics.csv` : segmentation, seuil de salve, mouvement/distance, partenaires réutilisés, session à faible couverture.',
        '- `source_qa.csv`, `validation.csv`, `provenance.json` : couverture, doublons, identité des sources, hypothèses et paramètres.',
        '- `figures/` : paramètres anogénitaux, nuages de points, cartes complètes sociales/rôles, benchmark de l’ajustement. Chaque figure existe en PNG, PDF et SVG.',
        '', '## Références méthodologiques','',
        '- [SciPy : Spearman et petits échantillons](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).',
        '- [SciPy : permutations de paires pour les corrélations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).',
        '- [Statsmodels : BH, BY et Holm](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html).','']
    (out/'RAPPORT_CORRELATIONS_SOCIALES.md').write_text('\n'.join(lines),encoding='utf-8')
    # A compact, complete focus table makes the initial question immediately reviewable.
    results[(results.behavior=='nose2anogenital')].to_csv(out/'anogenital_statistics.csv',index=False)
    # Keep descriptive distributions and missingness available beside inference.
    vocal.groupby(['genotype','virus'])[list(V_LABELS)].agg(['count','mean','median','std','min','max']).to_csv(out/'vocal_group_descriptives.csv')
    behavior.groupby(['role','behavior','genotype','virus'])[list(B_LABELS)].agg(['count','mean','median','std','min','max']).to_csv(out/'behavior_group_descriptives.csv')


def make_outputs(out:Path,behavior:pd.DataFrame,vocal:pd.DataFrame,qa:pd.DataFrame,
                 results:pd.DataFrame,sensitivity:pd.DataFrame) -> None:
    """Validate every numeric artifact before writing the final graphics and report."""
    configure()
    validation(out,behavior,vocal,qa,results)
    folder=out/'figures';folder.mkdir(exist_ok=True)
    social_maps(folder,results)
    anogenital_map(folder,results)
    scatter_sets(folder,behavior,vocal,results)
    adjustment_benchmark(folder,results,sensitivity)
    report(out,behavior,vocal,qa,results,sensitivity)
