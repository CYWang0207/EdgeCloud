"""Gate a cloud visual teacher before any Adapter distillation.

Fits a frozen BoxCars prototype head from clean training features, then reports
the teacher's clean/illumination/blur/noise accuracy on the same deterministic
camera corruptions used by the edge benchmark.  Do not run Adapter training
unless this result is convincingly above the edge baseline.
"""
import argparse, json, os, random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from boxcars_dataset import BoxCarsMultiView
from boxcars_camera_drift_dataset import PairedBoxCarsCameraDrift

IM = torch.tensor((.485,.456,.406)).view(1,1,3,1,1); IS = torch.tensor((.229,.224,.225)).view(1,1,3,1,1)
CM = torch.tensor((.48145466,.4578275,.40821073)).view(1,1,3,1,1); CS = torch.tensor((.26862954,.26130258,.27577711)).view(1,1,3,1,1)

def parse_args():
    p=argparse.ArgumentParser(description="Benchmark cloud visual teacher before distillation")
    p.add_argument('--dataset-path',required=True); p.add_argument('--model-path',required=True); p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--prototype-per-class',type=int,default=128,help='0 uses all clean train tracks')
    p.add_argument('--max-eval-samples',type=int,default=0,
                   help='quick gate only: evaluate a fixed random subset per condition; 0 uses all validation data')
    p.add_argument('--logit-scale',type=float,default=20.); return p.parse_args()

def make_base(path, split):
    return BoxCarsMultiView(path,split,'make',4,transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))

def stratified_clean_subset(dataset, per_class):
    """Avoid scanning an imbalanced 13k-track set just to build prototypes."""
    if not per_class: return dataset
    groups=[[] for _ in dataset.classes]
    for index, (_vehicle, label) in enumerate(dataset.base.samples): groups[label].append(index)
    chooser=random.Random(42); selected=[]
    for label, indices in enumerate(groups):
        chooser.shuffle(indices)
        if len(indices)<per_class: raise RuntimeError(f'class {label} has only {len(indices)} train tracks')
        selected.extend(indices[:per_class])
    chooser.shuffle(selected)
    return Subset(dataset, selected)

def teacher_feature(model, images, mask, device):
    pixels=((images*IS+IM).clamp(0,1)-CM)/CS; b,v=pixels.shape[:2]
    out=model(pixels.reshape(b*v,*pixels.shape[2:]).to(device,torch.bfloat16))
    h=out[0] if isinstance(out,(tuple,list)) else out.last_hidden_state
    h=h[:,0] if h.ndim==3 else h; h=h.reshape(b,v,-1).float(); w=mask.to(device,h.dtype).unsqueeze(-1)
    return F.normalize((h*w).sum(1)/w.sum(1).clamp_min(1),dim=-1)

@torch.no_grad()
def fit_prototypes(model, loader, classes, per_class, device):
    sums=counts=None
    for i,batch in enumerate(loader):
        h=teacher_feature(model,batch[0],batch[2],device); y=batch[3].to(device)
        if sums is None: sums=torch.zeros(len(classes),h.shape[1],device=device); counts=torch.zeros(len(classes),device=device)
        for c in y.unique():
            take=(y==c)
            if per_class: take &= counts[c] < per_class
            if take.any(): sums[c]+=h[take].sum(0); counts[c]+=take.sum()
        if i%50==0: print(f'prototype batches={i} counts={counts.int().tolist()}',flush=True)
        if per_class and bool((counts>=per_class).all()): break
    if (counts==0).any(): raise RuntimeError(f'missing classes: {(counts==0).nonzero().flatten().tolist()}')
    return F.normalize(sums/counts[:,None],dim=-1), counts.cpu()

@torch.no_grad()
def accuracy(model, loader, prototypes, device, clean):
    correct=count=0
    for i,batch in enumerate(loader):
        h=teacher_feature(model,batch[0] if clean else batch[1],batch[2],device)
        y=batch[3].to(device); correct+=(h@prototypes.T).argmax(1).eq(y).sum().item(); count+=len(y)
        if i%50==0: print(f'eval batches={i} samples={count}',flush=True)
    return correct/max(count,1),count

def fixed_subset(dataset, size, seed=123):
    if not size or size >= len(dataset): return dataset
    return Subset(dataset, random.Random(seed).sample(range(len(dataset)), size))

def main():
    a=parse_args(); device=torch.device('cuda')
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel,'all_tied_weights_keys'): PreTrainedModel.all_tied_weights_keys=property(lambda _self:{})
    from transformers import AutoModel
    model=AutoModel.from_pretrained(a.model_path,dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True).to(device).eval()
    train=make_base(a.dataset_path,'train'); val=make_base(a.dataset_path,'validation')
    common=dict(batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
    clean_pairs=PairedBoxCarsCameraDrift(train,('illumination',),clean_probability=1.,seed=42,return_metadata=True)
    proto,counts=fit_prototypes(model,DataLoader(stratified_clean_subset(clean_pairs,a.prototype_per_class),shuffle=True,**common),train.classes,a.prototype_per_class,device)
    results={'teacher':'InternViT-6B-224px','prototype_per_class':a.prototype_per_class,'prototype_counts':counts.tolist(),'split':'validation','accuracy':{}}
    specs={'clean':None,'illumination_1.0':('illumination',1.),'motion_blur_0.8':('motion_blur',.8),'sensor_noise_0.6':('sensor_noise',.6)}
    for name,spec in specs.items():
        ds=PairedBoxCarsCameraDrift(val,('illumination',),clean_probability=0.,seed=123,return_metadata=True) if spec is None else PairedBoxCarsCameraDrift(val,(spec[0],),clean_probability=0.,seed=123,fixed_drift=spec[0],fixed_severity=spec[1],return_metadata=True)
        ds=fixed_subset(ds,a.max_eval_samples)
        acc,n=accuracy(model,DataLoader(ds,shuffle=False,**common),proto,device,clean=spec is None)
        results['accuracy'][name]={'value':acc,'samples':n}; print(f'{name}: {acc:.4%} ({n} tracks)',flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True)
    with open(a.output,'w') as f: json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
