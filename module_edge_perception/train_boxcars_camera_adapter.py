"""Train a new, isolated BoxCars camera-degradation AdaptFormer checkpoint."""
import argparse, os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from adaptformer import attach_adaptformer, count_adapter_parameters, freeze_backbone, save_adapter_checkpoint, set_adapter_enabled
from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from boxcars_camera_drift_dataset import DRIFTS, PairedBoxCarsCameraDrift
from model import EarlyFusionMultiViewViT

def types(v):
    x=tuple(s.strip() for s in v.split(",") if s.strip())
    if not x or set(x)-set(DRIFTS): raise argparse.ArgumentTypeError(f"types must be from {DRIFTS}")
    return x
def fixed(v):
    out={}
    for item in v.split(","):
        k,eq,x=item.partition("=")
        if not eq or k not in DRIFTS or not 0<=float(x)<=1: raise argparse.ArgumentTypeError("fixed severities must be TYPE=0..1")
        out[k]=float(x)
    return out
def ap():
    p=argparse.ArgumentParser(); p.add_argument("--dataset-path",required=True);p.add_argument("--baseline-checkpoint",required=True);p.add_argument("--save-dir",required=True)
    p.add_argument("--task",choices=VALID_TASKS,default="make");p.add_argument("--drift-types",type=types,default=("illumination","motion_blur","sensor_noise"));p.add_argument("--drift-weights",type=float,nargs="+",default=(.3,.3,.4));p.add_argument("--fixed-severities",type=fixed,default={"illumination":.8,"motion_blur":.6,"sensor_noise":.6});p.add_argument("--clean-probability",type=float,default=.2)
    p.add_argument("--feature-weight",type=float,default=.25);p.add_argument("--consistency-weight",type=float,default=.2);p.add_argument("--r",type=int,default=32);p.add_argument("--epochs",type=int,default=8);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--lr",type=float,default=2e-4);p.add_argument("--weight-decay",type=float,default=.05);p.add_argument("--seed",type=int,default=42);p.add_argument("--max-train-batches",type=int,default=0);p.add_argument("--max-val-batches",type=int,default=0)
    return p.parse_args()
def load(m,path):
    x=torch.load(path,map_location="cpu");m.load_state_dict(x.get("model",x.get("state_dict",x)),strict=True)
def loader(a,split,clean):
    tf=[transforms.Resize((224,224))]+([transforms.RandomHorizontalFlip()] if split=="train" else [])+[transforms.ToTensor()]
    base=BoxCarsMultiView(a.dataset_path,split,a.task,4,transforms.Compose(tf))
    ds=PairedBoxCarsCameraDrift(base,a.drift_types,clean_probability=clean,seed=a.seed+(0 if split=="train" else 1_000_003),drift_weights=a.drift_weights,fixed_severities=a.fixed_severities)
    return ds,DataLoader(ds,batch_size=a.batch_size,shuffle=split=="train",num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
@torch.no_grad()
def evaluate(model,dl,limit):
    model.eval();co=dr=to=0
    for i,(clean,corrupt,mask,y) in enumerate(dl):
        if limit and i>=limit:break
        clean,corrupt,mask,y=clean.cuda(),corrupt.cuda(),mask.cuda(),y.cuda()
        with torch.autocast("cuda",dtype=torch.bfloat16): a=model(clean,view_mask=mask);b=model(corrupt,view_mask=mask)
        co+=(a.argmax(1)==y).sum().item();dr+=(b.argmax(1)==y).sum().item();to+=y.numel()
    return co/max(to,1),dr/max(to,1)
def main():
    a=ap();torch.manual_seed(a.seed); assert len(a.drift_types)==len(a.drift_weights)
    train,tl=loader(a,"train",a.clean_probability); val,vl=loader(a,"validation",0.)
    m=EarlyFusionMultiViewViT("vit_small_patch16_224",4,len(train.classes),pretrained=False);load(m,a.baseline_checkpoint);attach_adaptformer(m,a.r);freeze_backbone(m);m.cuda()
    opt=torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=a.lr,weight_decay=a.weight_decay);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);os.makedirs(a.save_dir,exist_ok=True);best=-1
    print(f"camera-mixture train={len(train)} val={len(val)} adapter={count_adapter_parameters(m)} types={a.drift_types} fixed={a.fixed_severities}",flush=True)
    for ep in range(a.epochs):
        m.train();total=correct=n=0
        for i,(clean,corrupt,mask,y) in enumerate(tl):
            if a.max_train_batches and i>=a.max_train_batches:break
            clean,corrupt,mask,y=clean.cuda(),corrupt.cuda(),mask.cuda(),y.cuda();set_adapter_enabled(m,False)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):cl,cf=m(clean,view_mask=mask,return_features=True)
            set_adapter_enabled(m,True);opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):lo,fe=m(corrupt,view_mask=mask,return_features=True);loss=F.cross_entropy(lo,y)+a.consistency_weight*F.kl_div(F.log_softmax(lo.float(),1),F.softmax(cl.float(),1),reduction="batchmean")+a.feature_weight*(1-F.cosine_similarity(fe.float(),cf.float(),dim=1).mean())
            loss.backward();opt.step();total+=loss.item()*y.numel();correct+=(lo.argmax(1)==y).sum().item();n+=y.numel()
        sched.step();clean_acc,drift_acc=evaluate(m,vl,a.max_val_batches);meta={"dataset":"boxcars116k","task":a.task,"drift_types":a.drift_types,"drift_weights":a.drift_weights,"fixed_severities":a.fixed_severities,"epoch":ep+1,"validation_clean_accuracy":clean_acc,"validation_mixture_accuracy":drift_acc,"r":a.r,"baseline_checkpoint":a.baseline_checkpoint}
        print(f"epoch={ep+1}/{a.epochs} loss={total/max(n,1):.4f} train={correct/max(n,1):.4f} clean={clean_acc:.4f} mixture={drift_acc:.4f}",flush=True);save_adapter_checkpoint(os.path.join(a.save_dir,"latest.pth"),m,**meta)
        if drift_acc>best:best=drift_acc;save_adapter_checkpoint(os.path.join(a.save_dir,"best.pth"),m,**meta)
if __name__=="__main__":main()
