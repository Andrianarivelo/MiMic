"""Injected real-USV recovery in unchanged WT/Het recording backgrounds.

Same verified whistle library and randomized schedule in every recording.
Received amplitude is digital band-limited RMS dBFS, not calibrated source SPL.
Original 60 s detector context and 0.15 s halo are retained. Both identifier
and final noise-classifier recovery are exported, with matched sham trials.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy import signal
from scipy.optimize import linear_sum_assignment
from scipy.integrate import trapezoid
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

# Configurable scientific parameters. Raw recordings are never written.
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vocalpy_engine'))
sys.path.insert(0,str(ROOT))
import attr_common as metadata
OUT=ROOT/'lgdel_usv_analysis'/'detector_recovery'
SR=384000
CONTEXT_SECONDS=60.15
CONTEXT_STARTS=[360.,540.,720.]
BAND=(45000.,125000.)
TEMPLATE_IDS=['31102_179','31078_427','31337_164','31318_71']
SNR_LEVELS=[-10.,-5.,0.,5.,10.,15.,20.]
AMPLITUDE_LEVELS=[-75.,-70.,-65.,-60.,-55.,-50.,-45.]
ISOLATED_LEVELS={'snr':5.,'amplitude':-60.}
IOU_THRESHOLD=.30
GUARD_S=.30
MIN_NOISE_RMS=1e-7
SEED=20261008
N_BOOT=4000
N_PERM=19999
TORCH_THREADS=2
COLORS={'WT':'#327A9E','HET':'#E68A32'}
FORMATS=['png','pdf','svg']
DPI=320


def wav(animal:str) -> Path:
    """Resolve the current data root, not an old machine's drive letter."""
    return ROOT.parent/animal/'baseline'/'1'/f'{animal}_1_baseline.wav'


def stamp() -> dict:
    """Fingerprint original sources, templates, engine/checkpoints and settings."""
    paths=[]
    for a in metadata.GENOTYPE_MAP:
        paths.extend([wav(a),wav(a).parent/f'{a}_1_baseline_outputs'/f'{a}_1_baseline_stats.csv',
                      wav(a).parent/f'{a}_1_baseline_outputs'/'parameters.yml'])
    paths.extend([ROOT/'lgdel_usv_analysis/attribution/call_features.csv',
                  ROOT/'lgdel_usv_analysis/social_call_probability/traces.npz',
                  ROOT.parent/'pykaboo_trial_plan_completed.xlsx'])
    engine=list((ROOT/'vocalpy_engine/vocalpy').rglob('*.py'))
    engine+=list((ROOT/'vocalpy_engine/vocalpy/nn/pretrained').glob('*.pth.tar'))
    return dict(version=1,sources={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths},
        engine={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in engine},
        code=hashlib.sha256('\n'.join(inspect.getsource(f) for f in
            [wav,bandpass,rms,template_library,prepare,match,Detector,run]).encode()).hexdigest(),templates=TEMPLATE_IDS,
        snr=SNR_LEVELS,amplitude=AMPLITUDE_LEVELS,isolated=ISOLATED_LEVELS,sr=SR,
        contexts=CONTEXT_STARTS,seconds=CONTEXT_SECONDS,band=BAND,iou=IOU_THRESHOLD,guard=GUARD_S,
        minimum_noise_rms=MIN_NOISE_RMS,seed=SEED,
        genotype_map=metadata.GENOTYPE_MAP)


def bandpass(x:np.ndarray) -> np.ndarray:
    """Use the same defined USV band to measure injected signal and background."""
    return signal.sosfiltfilt(signal.butter(4,BAND,btype='bandpass',fs=SR,output='sos'),x)


def rms(x:np.ndarray) -> float:
    """RMS amplitude, without independently normalizing any mixed recording."""
    return float(np.sqrt(np.mean(np.square(x))))


def template_library() -> tuple[dict,pd.DataFrame]:
    """Denoise visually audited real whistles, retaining their frequency tracks.

    A soft Wiener spectral mask estimates source noise from both flanks. It
    suppresses injecting the donor's background along with its call. The fixed
    four-template library is shared across genotypes; it is not attributed to
    either animal in donor dyads. Original and cleaned waveforms are retained.
    """
    features=pd.read_csv(ROOT/'lgdel_usv_analysis/attribution/call_features.csv',dtype={'animal_id':str})
    library={};rows=[];archive={}
    for call in TEMPLATE_IDS:
        r=features[features.call_id==call].iloc[0];a=str(r.animal_id)
        pad=.05;begin=int((r.start_s-pad)*SR);end=int((r.end_s+pad)*SR)
        x,_=sf.read(wav(a),start=begin,stop=end,dtype='float64')
        f,t,Z=signal.stft(x,fs=SR,nperseg=512,noverlap=384,boundary='zeros')
        relative_start=r.start_s-begin/SR;relative_end=r.end_s-begin/SR
        context=(t<relative_start-.01)|(t>relative_end+.01)
        noise=np.median(abs(Z[:,context])**2,axis=1)
        gain=np.clip(1-noise[:,None]/(abs(Z)**2+1e-24),0,1)
        gain*=((f>=BAND[0])&(f<=BAND[1]))[:,None]
        gain*=((t>=relative_start-.003)&(t<=relative_end+.003))[None,:]
        _,clean=signal.istft(Z*gain,fs=SR,nperseg=512,noverlap=384)
        lo=max(0,int((relative_start-.003)*SR));hi=int((relative_end+.003)*SR)
        raw=x[lo:hi];clean=clean[lo:hi]
        active=(int(.003*SR),min(len(clean),int((.003+r.end_s-r.start_s)*SR)))
        signal_rms=rms(bandpass(clean)[active[0]:active[1]])
        if signal_rms<=0:raise ValueError('Empty cleaned template')
        library[call]=dict(wave=clean/signal_rms,active=active)
        archive[call+'_raw']=raw;archive[call+'_clean']=clean
        rows.append(dict(template_id=call,source_animal=a,source_context_genotype=metadata.GENOTYPE_MAP[a],
            source_start_s=r.start_s,source_end_s=r.end_s,duration_ms=1000*(r.end_s-r.start_s),
            ridge_peak_snr_db=r.rf_amp_peak_db,source_band_rms_dbfs=20*np.log10(signal_rms),
            verified='visual inspection: continuous narrow-band whistle; actor/caller unknown',
            processing='45-125kHz soft spectral Wiener mask; active RMS normalization'))
    np.savez_compressed(OUT/'template_waveforms.npz',**archive)
    manifest=pd.DataFrame(rows);manifest.to_csv(OUT/'template_manifest.csv',index=False)
    return library,manifest


def prepare() -> None:
    """Select real chunks/time slots independently of injected recovery outcomes."""
    library,manifest=template_library();rng=np.random.default_rng(SEED)
    traces=np.load(ROOT/'lgdel_usv_analysis/social_call_probability/traces.npz')
    plan=pd.read_excel(ROOT.parent/'pykaboo_trial_plan_completed.xlsx');plan['Animal ID']=plan['Animal ID'].astype(str)
    plan=plan.set_index('Animal ID');backgrounds=[];trials=[]
    # Balance the sampled middle chunks across each experimental genotype.
    starts={}
    for group in ['WT','HET']:
        animals=[a for a,g in metadata.GENOTYPE_MAP.items() if g==group]
        choices=np.array(CONTEXT_STARTS*4);rng.shuffle(choices)
        starts.update(dict(zip(animals,choices)))
    design=list(range(len(TEMPLATE_IDS)*len(SNR_LEVELS)));rng.shuffle(design)
    schedule=[(TEMPLATE_IDS[k%len(TEMPLATE_IDS)],k//len(TEMPLATE_IDS)) for k in design]
    work=OUT/'work';work.mkdir(exist_ok=True)
    for a,g in metadata.GENOTYPE_MAP.items():
        start=starts[a];x,sr=sf.read(wav(a),start=int(start*SR),stop=int((start+CONTEXT_SECONDS)*SR),dtype='float64')
        if sr!=SR or x.ndim!=1:raise ValueError('Unexpected audio sampling/channel format')
        y=bandpass(x)
        proposed=start
        if rms(y)<=MIN_NOISE_RMS:
            for alternative in CONTEXT_STARTS:
                x,sr=sf.read(wav(a),start=int(alternative*SR),stop=int((alternative+CONTEXT_SECONDS)*SR),dtype='float64')
                y=bandpass(x)
                if rms(y)>MIN_NOISE_RMS:start=alternative;break
            else:raise RuntimeError(f'No measurable recorded background: {a}')
        calls=pd.read_csv(wav(a).parent/f'{a}_1_baseline_outputs'/f'{a}_1_baseline_stats.csv')
        valid=traces[a+'_valid'];positions=[]
        # One slot in each of 28 consecutive time strata, with deterministic
        # fallback candidates. Exclude known calls and unobserved tracking.
        for i in range(len(schedule)):
            pool=np.arange(1.,58.,.05)
            candidates=pool[np.argsort(abs(pool-(1.5+i*2)))]
            selected=None
            for local in candidates:
                absolute=start+local;bin0=int((absolute-.1-300)/.1);bin1=int((absolute+.15-300)/.1)+1
                no_call=not ((calls['start(s)']<absolute+.12+GUARD_S)&(calls['end(s)']>absolute-GUARD_S)).any()
                separated=all(abs(local-old)>.4 for old in positions)
                aa=library[schedule[i][0]]['active'];sample=int(local*SR)
                measurable=rms(y[sample+aa[0]:sample+aa[1]])>MIN_NOISE_RMS
                if no_call and separated and measurable and valid[bin0:bin1].all():selected=local;break
            if selected is None:raise RuntimeError(f'No clean observed injection slot {a}/{i}')
            positions.append(selected)
        sf.write(work/f'{a}_sham.wav',x,SR,subtype='FLOAT')
        slots=[]
        for i,(call,level_index) in enumerate(schedule):
            template=library[call];begin=int(positions[i]*SR);active=template['active']
            noise_rms=rms(y[begin+active[0]:begin+active[1]])
            slots.append(dict(slot=i,template_id=call,start_sample=begin,noise_rms=noise_rms,
                truth_start=start+(begin+active[0])/SR,truth_end=start+(begin+active[1])/SR,level_index=level_index))
        for mode,levels in [('snr',SNR_LEVELS),('amplitude',AMPLITUDE_LEVELS)]:
            for variant in ['pooled','isolated']:
                mixture=x.copy();included=[]
                for slot in slots:
                    level=levels[slot['level_index']]
                    if variant=='isolated' and level!=ISOLATED_LEVELS[mode]:continue
                    amp=slot['noise_rms']*10**(level/20) if mode=='snr' else 10**(level/20)
                    wave=library[slot['template_id']]['wave']*amp;begin=slot['start_sample']
                    mixture[begin:begin+len(wave)]+=wave
                    included.append(dict(animal_id=a,genotype=g,virus=str(plan.loc[a,'virus']),mode=mode,
                        variant=variant,target_level=level,received_amplitude_dbfs=20*np.log10(amp),
                        received_snr_db=20*np.log10(amp/slot['noise_rms']),context_start_s=start,
                        **{k:v for k,v in slot.items() if k!='level_index'}))
                peak=float(abs(mixture).max())
                if peak>=1:raise ValueError('Injection would clip; mixture is not silently rescaled')
                sf.write(work/f'{a}_{mode}_{variant}.wav',mixture,SR,subtype='FLOAT')
                for row in included:row['mixture_peak']=peak
                trials.extend(included)
        backgrounds.append(dict(animal_id=a,genotype=g,virus=str(plan.loc[a,'virus']),start_s=start,
            end_s=start+CONTEXT_SECONDS,band_rms_dbfs=20*np.log10(rms(y)),
            proposed_start_s=proposed,substituted_zero_background=start!=proposed,
            fraction_exact_zero_samples=float((x==0).mean()),
            natural_calls_in_context=int(((calls['start(s)']>=start)&(calls['start(s)']<start+60)).sum()),
            injection_slots=len(slots),sham_file=str(work/f'{a}_sham.wav')))
        print(f'Prepared {a} {g}: chunk {start:g}s, background {20*np.log10(rms(y)):.1f} dBFS',flush=True)
    pd.DataFrame(trials).to_csv(OUT/'injection_manifest.csv',index=False)
    pd.DataFrame(backgrounds).to_csv(OUT/'background_manifest.csv',index=False)


def match(trials:pd.DataFrame,intervals:list) -> np.ndarray:
    """One-to-one temporal-IoU matching, not nearest sound regardless of overlap."""
    result=np.zeros(len(trials),bool)
    if not intervals or not len(trials):return result
    a=trials[['truth_start','truth_end']].to_numpy();b=np.asarray(intervals)
    intersection=np.maximum(0,np.minimum(a[:,1,None],b[None,:,1])-np.maximum(a[:,0,None],b[None,:,0]))
    union=np.maximum(a[:,1,None],b[None,:,1])-np.minimum(a[:,0,None],b[None,:,0])
    iou=intersection/np.maximum(union,1e-12)
    # Prefer maximum valid match cardinality, then maximize overlap quality.
    score=np.where(iou>=IOU_THRESHOLD,min(len(a),len(b))+1+iou,0)
    ii,jj=linear_sum_assignment(-score);result[ii]=iou[ii,jj]>=IOU_THRESHOLD
    return result


class Detector:
    """Reuse validated classifier weights while running unchanged engine methods."""
    def __init__(self):
        import torch
        from vocalpy.nn.classifier import VocalClassifier
        self.torch=torch;torch.set_num_threads(TORCH_THREADS)
        self.classifier=VocalClassifier('noise',np.empty((0,)),batch_size=32)
    def run(self,path:Path,params:dict,context_start:float) -> tuple[list,list]:
        from vocalpy.pipelines.mouse import Mouse
        from vocalpy.nn import datasets
        from vocalpy.nn.datasets import create_array_from_list_of_vocals
        animal=Mouse('mouse',params)
        # Original middle-bin number preserves engine timestamp offsets.
        chunk=[str(path),'','','',SR,60,int(context_start/60)+1,0,int(CONTEXT_SECONDS*SR)]
        with threadpool_limits(limits=1):lov=animal.identifier(chunk)
        candidate=[(float(v.start),float(v.end)) for v in lov.vocals_in_recording]
        if lov.number_of_vocals:
            images=create_array_from_list_of_vocals(lov)
            self.classifier.dataset=self.classifier.create_dataset(images)
            self.classifier.dataloader=datasets.create_dataloader(self.classifier.dataset,32)
            predictions=self.classifier.classify_list_of_vocals(lov)
            lov.remove_vocals_classified_as_noise(predictions)
        retained=[(float(v.start),float(v.end)) for v in lov.vocals_in_recording]
        return candidate,retained


def run() -> None:
    """Resume detection per recording/file without repeating finished expensive jobs."""
    backgrounds=pd.read_csv(OUT/'background_manifest.csv',dtype={'animal_id':str})
    manifest=pd.read_csv(OUT/'injection_manifest.csv',dtype={'animal_id':str})
    detector=Detector();detfolder=OUT/'detections';detfolder.mkdir(exist_ok=True)
    for r in backgrounds.itertuples():
        params=yaml.safe_load((wav(r.animal_id).parent/f'{r.animal_id}_1_baseline_outputs'/'parameters.yml').read_text())
        if params['bin_size']!=60 or params['segmenter']:raise ValueError('Unsupported original detector configuration')
        filetypes=['sham','snr_pooled','amplitude_pooled','snr_isolated','amplitude_isolated']
        for filetype in filetypes:
            cache=detfolder/f'{r.animal_id}_{filetype}.json'
            if cache.exists():continue
            begin=time.time();candidate,retained=detector.run(OUT/'work'/f'{r.animal_id}_{filetype}.wav',params,r.start_s)
            cache.write_text(json.dumps(dict(candidate=candidate,retained=retained,params=params)),encoding='utf-8')
            print(f'{r.animal_id} {r.genotype} {filetype}: {len(candidate)} candidates / {len(retained)} retained, {time.time()-begin:.1f}s',flush=True)
        sham=json.loads((detfolder/f'{r.animal_id}_sham.json').read_text())
        rows=[]
        for (mode,variant),t in manifest[manifest.animal_id==r.animal_id].groupby(['mode','variant'],sort=False):
            t=t.copy();pred=json.loads((detfolder/f'{r.animal_id}_{mode}_{variant}.json').read_text())
            for stage in ['candidate','retained']:
                t['sham_'+stage]=match(t,sham[stage])
                t['detected_'+stage]=match(t,pred[stage])
                t['new_'+stage]=t['detected_'+stage]&~t['sham_'+stage]
            rows.append(t)
        pd.concat(rows).to_csv(OUT/f'trials_{r.animal_id}.csv',index=False)
    pd.concat([pd.read_csv(OUT/f'trials_{a}.csv',dtype={'animal_id':str}) for a in metadata.GENOTYPE_MAP]).to_csv(OUT/'recovery_trials.csv',index=False)


def statistics() -> None:
    """Session-weighted curves, cluster bootstrap CIs and two primary AUC tests."""
    trials=pd.read_csv(OUT/'recovery_trials.csv',dtype={'animal_id':str});rows=[]
    for (a,g,v,m,variant,level),t in trials.groupby(['animal_id','genotype','virus','mode','variant','target_level']):
        # Shams with a retained overlapping detection cannot establish new-call
        # recovery. Remove those slots from BOTH identifier/final denominators.
        use=t[~t.sham_retained]
        rows.append(dict(animal_id=a,genotype=g,virus=v,mode=m,variant=variant,target_level=level,
            n_injected=len(t),n_eligible=len(use),n_sham_retained=int(t.sham_retained.sum()),
            p_identifier=float(use.detected_candidate.mean()) if len(use) else np.nan,
            p_novel_identifier=float(use.new_candidate.mean()) if len(use) else np.nan,
            probability=float(use.new_retained.mean()) if len(use) else np.nan,
            amplitude_dbfs=float(use.received_amplitude_dbfs.mean()),snr_db=float(use.received_snr_db.mean())))
    sessions=pd.DataFrame(rows);sessions.to_csv(OUT/'session_recovery.csv',index=False)
    rng=np.random.default_rng(SEED);curve=[];tests=[];aucs=[];sensitivity=[]
    for mode in ['amplitude','snr']:
        t=sessions[(sessions['mode']==mode)&(sessions.variant=='pooled')]
        wide=t.pivot(index=['animal_id','genotype','virus'],columns='target_level',values='probability').dropna().reset_index()
        levels=AMPLITUDE_LEVELS if mode=='amplitude' else SNR_LEVELS
        values=wide[levels].to_numpy();g=wide.genotype.to_numpy();virus=wide.virus.to_numpy()
        # Label permutations preserve genotype counts within original virus.
        weights=np.zeros((N_PERM,len(wide)))
        for vv in np.unique(virus):
            idx=np.flatnonzero(virus==vv);nwt=int((g[idx]=='WT').sum())
            ranks=np.argsort(rng.random((N_PERM,len(idx))),axis=1);labels=np.zeros((N_PERM,len(idx)),bool)
            labels[np.arange(N_PERM)[:,None],ranks[:,:nwt]]=True
            weights[:,idx]=np.where(labels,-1/(g=='WT').sum(),1/(g=='HET').sum())
        for j,level in enumerate(levels):
            a=values[g=='WT',j];b=values[g=='HET',j];effect=float(b.mean()-a.mean());null=weights@values[:,j]
            p=float((1+(abs(null)>=abs(effect)-1e-12).sum())/(N_PERM+1))
            tests.append(dict(mode=mode,target_level=level,effect_het_minus_wt=effect,p=p,n_wt=len(a),n_het=len(b)))
            for group,x in [('WT',a),('HET',b)]:
                boot=rng.choice(x,(N_BOOT,len(x))).mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
                curve.append(dict(mode=mode,target_level=level,genotype=group,probability=float(x.mean()),ci_low=lo,ci_high=hi,n_recordings=len(x)))
        area=trapezoid(values,x=np.asarray(levels),axis=1)/(levels[-1]-levels[0])
        a=area[g=='WT'];b=area[g=='HET'];effect=float(b.mean()-a.mean());null=weights@area
        boots=rng.choice(b,(N_BOOT,len(b))).mean(axis=1)-rng.choice(a,(N_BOOT,len(a))).mean(axis=1)
        lo,hi=np.quantile(boots,[.025,.975]);p=float((1+(abs(null)>=abs(effect)-1e-12).sum())/(N_PERM+1))
        aucs.append(dict(mode=mode,auc_wt=float(a.mean()),auc_het=float(b.mean()),effect_het_minus_wt=effect,
            ci_low=lo,ci_high=hi,p=p,n_wt=len(a),n_het=len(b),family='two primary normalized curve-AUC tests'))
        for a_id,g_id in zip(wide.animal_id,g):
            pooled=t[(t.animal_id==a_id)&(t.target_level==ISOLATED_LEVELS[mode])].iloc[0]
            isolated=sessions[(sessions.animal_id==a_id)&(sessions['mode']==mode)&(sessions.variant=='isolated')].iloc[0]
            sensitivity.append(dict(animal_id=a_id,genotype=g_id,mode=mode,level=ISOLATED_LEVELS[mode],
                pooled_probability=pooled.probability,isolated_probability=isolated.probability,
                difference=isolated.probability-pooled.probability))
    primary=pd.DataFrame(aucs);primary['p_holm']=multipletests(primary.p,method='holm')[1]
    secondary=pd.DataFrame(tests);secondary['p_holm']=multipletests(secondary.p,method='holm')[1]
    primary.to_csv(OUT/'primary_auc_statistics.csv',index=False);secondary.to_csv(OUT/'level_statistics.csv',index=False)
    pd.DataFrame(curve).to_csv(OUT/'curve_statistics.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'isolated_level_sensitivity.csv',index=False)


def main():
    """Run or resume injection, exact detection, inference and final figures."""
    global OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','detect','report','all'],default='all')
    p.add_argument('--figures-only',action='store_true')
    p.add_argument('--recompute',action='store_true',help='Repeat detector recovery from existing validated injection WAVs')
    p.add_argument('--output-dir',type=Path,default=OUT,help='Separate derived output directory for a new experiment version')
    args=p.parse_args()
    OUT=args.output_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    fp=json.loads(json.dumps(stamp()));meta=OUT/'provenance.json'
    if meta.exists() and json.loads(meta.read_text())['fingerprint']!=fp:
        raise RuntimeError('Numeric fingerprint changed. Use a new output/version, preserving previous results.')
    if args.figures_only:
        if not meta.exists():raise RuntimeError('No validated experiment cache')
    elif args.stage in ['prepare','all'] and not (OUT/'injection_manifest.csv').exists():
        prepare();meta.write_text(json.dumps(dict(fingerprint=fp,experiment='injected verified-USV recovery',
            amplitude='digital active-support 45-125kHz RMS dBFS',snr='injected clean band RMS / same-slot background band RMS',
            context='genuine original 60s middle bin +0.15s forward halo',
            pooled='28 calls, four templates x seven levels, same schedule across cohorts',
            sensitivity='single transition level at identical positions without other injected levels',
            environment=sys.executable),indent=2),encoding='utf-8')
    if args.recompute:
        if args.figures_only or args.stage not in ['detect','all'] or not meta.exists():
            raise ValueError('--recompute requires validated injections and stage detect/all')
        # Delete only named generated prediction caches. Preserve all audio,
        # manifests, prior statistics, and every original input.
        for animal_id in metadata.GENOTYPE_MAP:
            for variant in ['sham','snr_pooled','amplitude_pooled','snr_isolated','amplitude_isolated']:
                cache=OUT/'detections'/f'{animal_id}_{variant}.json'
                if cache.exists():cache.unlink()
    if not args.figures_only and args.stage in ['detect','all']:run()
    if args.figures_only or args.stage in ['report','all']:
        statistics()
        from detector_recovery_figures import make_outputs
        make_outputs(OUT)
        # Inference and styling can evolve without invalidating expensive,
        # unchanged injected waveforms and detector predictions.
        analysis=dict(seed=SEED,n_bootstrap=N_BOOT,n_permutations=N_PERM,
            inference_sha256=hashlib.sha256(inspect.getsource(statistics).encode()).hexdigest(),
            renderer_sha256=hashlib.sha256((ROOT/'scripts/detector_recovery_figures.py').read_bytes()).hexdigest(),
            python=sys.version,executable=sys.executable,
            numpy=np.__version__,pandas=pd.__version__)
        (OUT/'analysis_provenance.json').write_text(json.dumps(analysis,indent=2),encoding='utf-8')


if __name__=='__main__':main()
