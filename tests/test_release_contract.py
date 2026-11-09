"""CPU checks for the portable release contract; no training is launched."""
import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rescale.config import DATASETS, load_config, apply_overrides
from tools.compare_runs import compare


class ReleaseContract(unittest.TestCase):
    def test_internal_imports_resolve_after_reorganization(self):
        for path in ROOT.rglob('*.py'):
            if '.git' in path.parts or '__pycache__' in path.parts:
                continue
            package='.'.join(path.relative_to(ROOT).parts[:-1])
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if not isinstance(node,ast.ImportFrom):
                    continue
                name=importlib.util.resolve_name('.'*node.level+(node.module or ''),package) if node.level else node.module
                if name and (name=='rescale' or name.startswith('rescale.')):
                    target=ROOT.joinpath(*name.split('.'))
                    self.assertTrue(target.with_suffix('.py').is_file() or (target/'__init__.py').is_file(),f'{path}: {name}')

    def test_all_fold_indexes_and_channels(self):
        counts={'CVC-ClinicDB':612,'ISIC2018':2594,'BUSI':647,'PAPILA':488,'PTX498':498,'FUSeg2021':1010}
        for name,count in counts.items():
            with self.subTest(dataset=name):
                m=json.loads((ROOT/'manifests'/f'{name}.json').read_text())
                self.assertEqual(len(m['channel_names']),3)
                ids={s['id'] for s in m['samples']}
                self.assertEqual(len(ids),count)
                folds=json.loads((ROOT/'splits'/f'{name}.json').read_text())['folds']
                self.assertEqual(len(folds),5)
                seen=[]
                for f in folds:
                    self.assertFalse(set(f['train'])&set(f['val']))
                    self.assertEqual(set(f['train'])|set(f['val']),ids)
                    seen.extend(f['val'])
                self.assertEqual(set(seen),ids)
                self.assertEqual(len(seen),count)

    def test_shared_config_preserves_model_and_training_across_datasets(self):
        base=load_config(str(ROOT/'configs/rescale.yaml'))
        self.assertEqual(list((ROOT/'configs').glob('*.yaml')), [ROOT/'configs/rescale.yaml'])
        for dataset in DATASETS:
            cfg=load_config(str(ROOT/'configs/rescale.yaml'), dataset=dataset)
            for key in ('model','training','innovations','metrics'):
                self.assertEqual(cfg[key],base[key])
            self.assertEqual(cfg['data']['manifest_json'],f'data/{dataset}/manifest.json')
            self.assertEqual(cfg['data']['splits_json'],f'splits/{dataset}.json')
            self.assertEqual(cfg['output']['save_dir'],f'outputs/{dataset}')
        self.assertEqual(base['training']['num_epochs'],200)
        self.assertTrue(base['innovations']['cuda']['enabled'])
        self.assertTrue(base['innovations']['recursion']['train_one_roi_per_image'])

    def test_dataset_validation_and_explicit_path_override(self):
        path=str(ROOT/'configs/rescale.yaml')
        with self.assertRaisesRegex(ValueError,'Unknown dataset'):
            load_config(path,dataset='unknown')
        cfg=apply_overrides(load_config(path,dataset='BUSI'),['output.save_dir=custom/run'])
        self.assertEqual(cfg['data']['splits_json'],'splits/BUSI.json')
        self.assertEqual(cfg['output']['save_dir'],'custom/run')

    def test_comparison_rejects_missing_focused_training(self):
        reference={'train_loss':[1.,.9],'base_dice':[.3,.4]}
        broken={**reference,'focused_loss':[0.,0.],'train_roi_per_image':[0.,0.]}
        self.assertFalse(any(r['passed'] for r in compare(reference,broken)))
        valid={**reference,'focused_loss':[.5,.4],'train_roi_per_image':[1.,1.]}
        self.assertTrue(all(r['passed'] for r in compare(reference,valid)))


if __name__=='__main__': unittest.main()
