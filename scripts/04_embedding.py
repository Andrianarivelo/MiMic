"""MODULE D - Embedding geometry + individual/genotype classification.

Loads calls_dataset.npz. Uses DINOv2 spectrogram-card embeddings (primary) and the
z-scored acoustic feature vector (baseline). Produces:
  fig04_umap_dinov2.png  - UMAP(emb_dinov2), 3 panels: genotype / mouse / phase
  fig04_umap_features.png- UMAP(acoustic feat), 2 panels: genotype / mouse
  fig04_confusion_mouse.png - 6-class "which mouse" RF classifier (StratifiedKFold CV),
                              row-normalised confusion + balanced-accuracy bars
  fig04_crossphase.png   - individual-signature generalisation: train on female-phase,
                           test on male-phase for the two high-count WT mice

HONESTY: WT dominate call counts (WT n=559 vs Het n=77); Het per-mouse n is 15-24.
Balanced accuracy + stratified CV are used throughout so that the 2 WT mice cannot
inflate scores by sheer count. Chance lines are drawn explicitly.
"""
import sys
sys.path.insert(0, '/home/andry/UVS/scripts')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import common as C
C.apply_style()

import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay)

RNG = 0
np.random.seed(RNG)

# ---------------------------------------------------------------- load
D = np.load('/home/andry/UVS/scripts/calls_dataset.npz', allow_pickle=True)
Xemb = D['emb_dinov2'].astype(float)      # [N,384]
Xfeat = D['feat'].astype(float)           # [N,10] z-scored acoustic
mouse = D['mouse'].astype(str)
geno = D['geno'].astype(str)
phase = D['phase'].astype(str)
N = len(mouse)

MICE = ['31096', '31097', '31098', '31099', '31101', '31102']  # WT,WT,Het,Het,Het,Het
MOUSE_GENO = {'31096': 'WT', '31097': 'WT', '31098': 'Het',
              '31099': 'Het', '31101': 'Het', '31102': 'Het'}
mouse_color = {m: (C.WT_COLOR if MOUSE_GENO[m] == 'WT' else C.HET_COLOR) for m in MICE}
# distinct markers so mice of same genotype are separable in the UMAP
mouse_marker = {'31096': 'o', '31097': 's', '31098': '^', '31099': 'v',
                '31101': 'D', '31102': 'P'}
phase_color = {'alone': '#6a51a3', 'female': '#e6550d', 'male': '#31a354'}

print('N =', N, '| WT =', (geno == 'WT').sum(), 'Het =', (geno == 'Het').sum())
for m in MICE:
    print(f'  {m} ({MOUSE_GENO[m]}): {(mouse==m).sum()}')


# ---------------------------------------------------------------- UMAP helper
def run_umap(X, seed=RNG, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric='euclidean', random_state=seed)
    return reducer.fit_transform(X)


def scatter_by(ax, emb, labels, order, colors, markers=None, title='', legend=True):
    for lab in order:
        m = labels == lab
        if not m.any():
            continue
        mk = markers[lab] if markers else 'o'
        ax.scatter(emb[m, 0], emb[m, 1], s=16, c=colors[lab], marker=mk,
                   alpha=0.75, linewidths=0.2, edgecolors='white', label=str(lab))
    ax.set_title(title)
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(loc='best', fontsize=8, markerscale=1.3, ncol=1)


# ================================================================ FIG 1: UMAP dinov2
print('\n[1] UMAP(emb_dinov2) ...')
emb2 = run_umap(Xemb)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
# panel A: genotype
scatter_by(axes[0], emb2, geno, ['WT', 'Het'], C.GENO_COLOR,
           title='DINOv2 UMAP - by genotype')
# panel B: mouse (colour=genotype, marker=individual)
for m in MICE:
    sel = mouse == m
    axes[1].scatter(emb2[sel, 0], emb2[sel, 1], s=18, c=mouse_color[m],
                    marker=mouse_marker[m], alpha=0.75, linewidths=0.2,
                    edgecolors='white', label=f'{m} ({MOUSE_GENO[m]})')
axes[1].set_title('DINOv2 UMAP - by individual')
axes[1].set_xlabel('UMAP-1'); axes[1].set_ylabel('UMAP-2')
axes[1].set_xticks([]); axes[1].set_yticks([])
axes[1].legend(loc='best', fontsize=7.5, markerscale=1.2)
# panel C: phase
scatter_by(axes[2], emb2, phase, C.PHASE_ORDER, phase_color,
           title='DINOv2 UMAP - by phase')
fig.suptitle('Fig 04 - DINOv2 spectrogram-card embedding (UMAP)', fontweight='bold')
fig.tight_layout()
C.save_fig(fig, 'fig04_umap_dinov2.png')
plt.close(fig)


# ================================================================ FIG 2: UMAP feat
print('[2] UMAP(acoustic feat) ...')
embf = run_umap(Xfeat)
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
scatter_by(axes[0], embf, geno, ['WT', 'Het'], C.GENO_COLOR,
           title='Acoustic-feature UMAP - by genotype')
for m in MICE:
    sel = mouse == m
    axes[1].scatter(embf[sel, 0], embf[sel, 1], s=18, c=mouse_color[m],
                    marker=mouse_marker[m], alpha=0.75, linewidths=0.2,
                    edgecolors='white', label=f'{m} ({MOUSE_GENO[m]})')
axes[1].set_title('Acoustic-feature UMAP - by individual')
axes[1].set_xlabel('UMAP-1'); axes[1].set_ylabel('UMAP-2')
axes[1].set_xticks([]); axes[1].set_yticks([])
axes[1].legend(loc='best', fontsize=7.5, markerscale=1.2)
fig.suptitle('Fig 04 - Acoustic-feature embedding (UMAP baseline)', fontweight='bold')
fig.tight_layout()
C.save_fig(fig, 'fig04_umap_features.png')
plt.close(fig)


# ================================================================ classifiers
print('[3] Classifiers (StratifiedKFold CV on emb_dinov2) ...')
# min class count = 15 (mouse 31102) -> 5-fold is safe (>=3 per fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)


def make_rf():
    return RandomForestClassifier(n_estimators=400, class_weight='balanced_subsample',
                                  random_state=RNG, n_jobs=-1)


# --- 6-class "which mouse" ---
y_mouse = mouse.copy()
ypred_mouse = cross_val_predict(make_rf(), Xemb, y_mouse, cv=skf, n_jobs=-1)
bacc_mouse = balanced_accuracy_score(y_mouse, ypred_mouse)
# per-fold balanced accuracy for an honest spread
bacc_mouse_folds = cross_val_score(make_rf(), Xemb, y_mouse, cv=skf,
                                   scoring='balanced_accuracy', n_jobs=-1)
cm = confusion_matrix(y_mouse, ypred_mouse, labels=MICE)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
chance_mouse = 1.0 / 6

# --- 2-class genotype ---
bacc_geno_folds = cross_val_score(make_rf(), Xemb, geno, cv=skf,
                                  scoring='balanced_accuracy', n_jobs=-1)
ypred_geno = cross_val_predict(make_rf(), Xemb, geno, cv=skf, n_jobs=-1)
bacc_geno = balanced_accuracy_score(geno, ypred_geno)
chance_geno = 0.5

# --- baseline on acoustic feat, for context ---
bacc_mouse_feat = cross_val_score(make_rf(), Xfeat, y_mouse, cv=skf,
                                  scoring='balanced_accuracy', n_jobs=-1).mean()
bacc_geno_feat = cross_val_score(make_rf(), Xfeat, geno, cv=skf,
                                 scoring='balanced_accuracy', n_jobs=-1).mean()

print(f'  MOUSE (6-cls) balanced acc = {bacc_mouse:.3f}  '
      f'(folds {bacc_mouse_folds.mean():.3f}+-{bacc_mouse_folds.std():.3f})  chance={chance_mouse:.3f}')
print(f'  GENO  (2-cls) balanced acc = {bacc_geno:.3f}  '
      f'(folds {bacc_geno_folds.mean():.3f}+-{bacc_geno_folds.std():.3f})  chance={chance_geno:.3f}')
print(f'  [baseline feat] mouse={bacc_mouse_feat:.3f}  geno={bacc_geno_feat:.3f}')


# ================================================================ FIG 3: confusion + bars
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6),
                         gridspec_kw={'width_ratios': [1.15, 1]})
# confusion matrix (row-normalised)
ax = axes[0]
disp = ConfusionMatrixDisplay(cm_norm, display_labels=[f'{m}\n{MOUSE_GENO[m]}' for m in MICE])
disp.plot(ax=ax, cmap='viridis', colorbar=True, values_format='.2f')
ax.set_title(f'Which-mouse confusion (row-normalised)\nbalanced acc = {bacc_mouse:.2f}  '
             f'(chance {chance_mouse:.2f})')
ax.set_xlabel('predicted mouse'); ax.set_ylabel('true mouse')
for txt in ax.texts:
    txt.set_fontsize(8)
ax.grid(False)

# balanced-accuracy bars vs chance
ax = axes[1]
names = ['mouse\n(6-class)', 'genotype\n(2-class)']
vals = [bacc_mouse, bacc_geno]
errs = [bacc_mouse_folds.std(), bacc_geno_folds.std()]
chances = [chance_mouse, chance_geno]
xpos = np.arange(len(names))
bars = ax.bar(xpos, vals, yerr=errs, capsize=5, width=0.55,
              color=['#3182bd', '#756bb1'], edgecolor='#222', linewidth=0.8, zorder=3)
for x, c in zip(xpos, chances):
    ax.hlines(c, x - 0.32, x + 0.32, color='#d62728', lw=2.2, zorder=4)
    ax.text(x + 0.34, c, f'chance {c:.2f}', va='center', ha='left',
            fontsize=8.5, color='#d62728')
for x, v, e in zip(xpos, vals, errs):
    ax.text(x, v + e + 0.02, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(names)
ax.set_ylabel('balanced accuracy (5-fold CV)')
ax.set_ylim(0, 1.0)
ax.set_title('DINOv2 classification vs chance')
fig.suptitle('Fig 04 - Individual & genotype decoding from DINOv2 embeddings',
             fontweight='bold')
fig.tight_layout()
C.save_fig(fig, 'fig04_confusion_mouse.png')
plt.close(fig)


# ================================================================ FIG 4: cross-phase
print('[4] Cross-phase generalisation (train female -> test male, WT high-count) ...')
WT2 = ['31096', '31097']
sel_wt = np.isin(mouse, WT2)
tr = sel_wt & (phase == 'female')
te = sel_wt & (phase == 'male')
Xtr, ytr = Xemb[tr], mouse[tr]
Xte, yte = Xemb[te], mouse[te]
print(f'  train(female) n={tr.sum()}  test(male) n={te.sum()}')
print('   train counts:', {m: int((ytr == m).sum()) for m in WT2})
print('   test  counts:', {m: int((yte == m).sum()) for m in WT2})

clf = make_rf().fit(Xtr, ytr)
yhat = clf.predict(Xte)
acc_cp = (yhat == yte).mean()
bacc_cp = balanced_accuracy_score(yte, yhat)
chance_cp = 0.5
cm_cp = confusion_matrix(yte, yhat, labels=WT2)
cm_cp_norm = cm_cp / cm_cp.sum(axis=1, keepdims=True)

# within-phase reference: 5-fold CV on female-phase only (same 2 WT mice)
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
bacc_within = cross_val_score(make_rf(), Xtr, ytr, cv=skf2,
                              scoring='balanced_accuracy', n_jobs=-1).mean()

print(f'  cross-phase acc={acc_cp:.3f}  balanced acc={bacc_cp:.3f}  chance={chance_cp:.2f}')
print(f'  within-female-phase CV balanced acc (reference) = {bacc_within:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2),
                         gridspec_kw={'width_ratios': [1, 1]})
# confusion of the cross-phase test
ax = axes[0]
disp = ConfusionMatrixDisplay(cm_cp_norm, display_labels=[f'{m}\nWT' for m in WT2])
disp.plot(ax=ax, cmap='cividis', colorbar=True, values_format='.2f')
ax.set_title(f'Train female -> test male\ncross-phase balanced acc = {bacc_cp:.2f}')
ax.set_xlabel('predicted mouse'); ax.set_ylabel('true mouse')
ax.grid(False)

# bars: within-phase CV vs cross-phase vs chance
ax = axes[1]
names = ['within female\n(5-fold CV)', 'cross-phase\n(female->male)']
vals = [bacc_within, bacc_cp]
bars = ax.bar([0, 1], vals, width=0.5, color=['#9ecae1', '#3182bd'],
              edgecolor='#222', linewidth=0.8, zorder=3)
ax.hlines(chance_cp, -0.4, 1.4, color='#d62728', lw=2.2, ls='--', zorder=4,
          label=f'chance {chance_cp:.2f}')
for x, v in zip([0, 1], vals):
    ax.text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(names)
ax.set_ylabel('balanced accuracy'); ax.set_ylim(0, 1.05)
ax.set_title('Does an individual signature generalise across phases?')
ax.legend(loc='lower right', fontsize=9)
fig.suptitle('Fig 04 - Cross-phase individual-signature generalisation (WT 31096 vs 31097)',
             fontweight='bold')
fig.tight_layout()
C.save_fig(fig, 'fig04_crossphase.png')
plt.close(fig)


# ================================================================ summary
print('\n================ MODULE D SUMMARY ================')
print(f'Individual (6-class) balanced acc = {bacc_mouse:.3f}  vs chance {chance_mouse:.3f}  '
      f'-> {bacc_mouse/chance_mouse:.1f}x chance')
print(f'Genotype  (2-class) balanced acc = {bacc_geno:.3f}  vs chance {chance_geno:.3f}')
print(f'Cross-phase (female->male, WT) balanced acc = {bacc_cp:.3f}  vs chance {chance_cp:.2f}  '
      f'(raw acc {acc_cp:.3f})')
print(f'Acoustic-feature baseline: mouse={bacc_mouse_feat:.3f}  geno={bacc_geno_feat:.3f}')
print('==================================================')
