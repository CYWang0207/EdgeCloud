"""Full split comparison for a calibrated BoxCars camera-mixture adapter."""
import argparse, json, os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from adaptformer import attach_adaptformer, load_adapter_checkpoint, set_adapter_enabled
from boxcars_dataset import BoxCarsMultiView, VALID_TASKS
from boxcars_camera_drift_dataset import DRIFTS, PairedBoxCarsCameraDrift
from model import EarlyFusionMultiViewViT

def spec(v):
    k,eq,x=v.partition("=")
    if not eq or k not in DRIFTS or not 0<=float(x)<=1: raise argparse.ArgumentTypeError("spec must be TYPE=0..1")
    return k,float(x)
def ap():
    p=argparse.ArgumentParser();p.add_argument("--dataset-path",required=True);p.add_argument("--baseline-checkpoint",required=True);p.add_argument("--adapter-checkpoint",required=True);p.add_argument("--output-json",required=True);p.add_argument("--task",choices=VALID_TASKS,default="make");p.add_argument("--split",choices=("validation","test"),default="validation");p.add_argument("--corruption-specs",type=spec,nargs="+",default=(("illumination",.8),("motion_blur",.6),("sensor_noise",.6)));p.add_argument("--batch-size",type=int,default=16);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--seed",type=int,default=123);return p.parse_args()
def base_load(m,path):
    x=torch.load(path,map_location="cpu");m.load_state_dict(x.get("model",x.get("state_dict",x)),strict=True)
@torch.no_grad()
def run(m,dl,clean=False):
    c=n=0
    for b in dl:
        x=(b[0] if clean else b[1]).cuda();mask,y=b[2].cuda(),b[3].cuda()
        with torch.autocast("cuda",dtype=torch.bfloat16):z=m(x,view_mask=mask)
        c+=(z.argmax(1)==y).sum().item();n+=y.numel()
    return c/max(n,1)
def main():
    a=ap();tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]);base=BoxCarsMultiView(a.dataset_path,a.split,a.task,4,tf);m=EarlyFusionMultiViewViT("vit_small_patch16_224",4,len(base.classes),pretrained=False);base_load(m,a.baseline_checkpoint);attach_adaptformer(m,32);m.cuda().eval()
    def dl(k,s): return DataLoader(PairedBoxCarsCameraDrift(base,(k,),fixed_drift=k,fixed_severity=s,seed=a.seed),batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=True,persistent_workers=a.num_workers>0)
    first=dl(*a.corruption_specs[0]);set_adapter_enabled(m,False);out={"dataset":"boxcars116k","split":a.split,"calibrated_corruptions":[f"{k}={s}" for k,s in a.corruption_specs],"baseline":{"clean":run(m,first,True)},"adapter":{}}
    for k,s in a.corruption_specs:out["baseline"][f"{k}_{s}"]=run(m,dl(k,s))
    load_adapter_checkpoint(m,a.adapter_checkpoint,device="cpu");set_adapter_enabled(m,True);out["adapter"]["clean"]=run(m,first,True)
    for k,s in a.corruption_specs:out["adapter"][f"{k}_{s}"]=run(m,dl(k,s))
    for key,val in out["baseline"].items():out.setdefault("delta_pp",{})[key]=(out["adapter"][key]-val)*100
    os.makedirs(os.path.dirname(a.output_json) or ".",exist_ok=True)
    with open(a.output_json,"w",encoding="utf-8") as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
