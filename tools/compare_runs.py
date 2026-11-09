"""Compare two two-epoch logger files without treating early Dice as final accuracy."""
import argparse
import json
from pathlib import Path


def compare(reference, candidate, dice_gap_pp=0.5, loss_gap=0.01):
    rows=[]
    for i in range(2):
        a,b=reference['train_loss'][i],candidate['train_loss'][i]
        gap=abs(reference['base_dice'][i]-candidate['base_dice'][i])*100
        relative=abs(a-b)/max(abs(a),1e-12)
        focused=candidate.get('focused_loss',candidate.get('mask_polish_refined_loss'))[i]
        rois=candidate['train_roi_per_image'][i]
        ok=gap<=dice_gap_pp and relative<=loss_gap and focused>0 and abs(rois-1)<1e-6
        rows.append(dict(epoch=i+1,reference_loss=a,candidate_loss=b,
                         reference_base_dice=reference['base_dice'][i],
                         candidate_base_dice=candidate['base_dice'][i],
                         dice_gap_pp=gap,relative_loss_gap=relative,
                         focused_loss=focused,train_roi_per_image=rois,passed=ok))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--max-dice-gap-pp',type=float,default=0.5)
    p.add_argument('--max-relative-loss-gap',type=float,default=0.01)
    a=p.parse_args()
    rows=compare(json.loads(a.reference.read_text())['data'],
                 json.loads(a.candidate.read_text())['data'],
                 a.max_dice_gap_pp,a.max_relative_loss_gap)
    print(json.dumps(rows,indent=2))
    if not all(r['passed'] for r in rows): raise SystemExit('Two-epoch check failed')


if __name__=='__main__': main()
