"""Bind the supplied sample indexes to local dataset files without changing folds."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['CVC-ClinicDB','ISIC2018','BUSI','PAPILA','PTX498','FUSeg2021'])
    p.add_argument('--root', required=True, type=Path)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here/'manifests'/f'{a.dataset}.json').read_text())
    missing = []
    for sample in manifest['samples']:
        for key in ('image', 'label'):
            path = a.root.resolve()/sample[key]
            if not path.is_file():
                missing.append(str(path))
            sample[key] = str(path)
    if missing:
        raise FileNotFoundError(f'{len(missing)} files missing; first examples: {missing[:5]}')
    ids = {s['id'] for s in manifest['samples']}
    splits = json.loads((here/'splits'/f'{a.dataset}.json').read_text())
    validation = []
    for fold in splits['folds']:
        train, val = set(fold['train']), set(fold['val'])
        if train & val or train | val != ids:
            raise ValueError('Fold does not partition the supplied sample IDs')
        validation.extend(fold['val'])
    if len(validation) != len(ids) or set(validation) != ids:
        raise ValueError('Each image must appear in one validation fold')
    output = here/'data'/a.dataset/'manifest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Bound {len(ids)} images and verified five existing folds: {output}')


if __name__ == '__main__':
    main()
