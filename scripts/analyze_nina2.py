import json

with open(r'c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\_nina_eos9\birdclef-2026-eos-9.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find dataset_sources in metadata
if 'metadata' in nb:
    meta = nb['metadata']
    if 'kaggle' in meta:
        print('Kaggle metadata:', json.dumps(meta['kaggle'], indent=2)[:2000])

# Find all dataset/model references in code cells
import re
all_refs = set()
for cell in nb['cells']:
    src = ''.join(cell['source'])
    # Find kaggle dataset references
    for m in re.finditer(r'(?:hellodave|hideyuki|tucker|nina|tonylica|rishikesh|mtoshidesu|google)[/\w\-.]+', src):
        all_refs.add(m.group())
    # Find MODEL_DIR references
    for m in re.finditer(r'MODEL_DIR\s*=\s*[^\n]+', src):
        print(f'MODEL_DIR: {m.group()[:150]}')

print(f'\nDataset/model refs found:')
for r in sorted(all_refs):
    print(f'  {r}')

# Print the TAX_SMOOTHING function
for cell in nb['cells']:
    src = ''.join(cell['source'])
    if 'def f_TAX_SMOOTHING_POSTPROC' in src:
        idx = src.find('def f_TAX_SMOOTHING_POSTPROC')
        print(f'\n=== TAX_SMOOTHING_POSTPROC ===')
        print(src[idx:idx+2000])
        break
