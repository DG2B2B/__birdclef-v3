import json

with open(r'c:\Users\Utilisateur\Mon Drive\Kaggle\__birdclef-v3\_nina_eos9\birdclef-2026-eos-9.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Cells: {len(nb["cells"])}')

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    keywords = ['solutions', 'Model_', 'weight', 'ensemble', 'type_add', 'rank', 'direct', 'add2', 'add3']
    if any(k in src for k in keywords):
        # Print first 200 chars
        first_line = src.strip().split('\n')[0][:150]
        print(f'Cell {i} ({cell["cell_type"]}): {first_line}')

# Also show the solutions cell
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'solutions = {' in src:
        print(f'\n=== SOLUTIONS CELL {i} ===')
        print(src[:800])
        break

# Show blend functions
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'def rank_1_add2' in src or 'def direct_add2' in src:
        print(f'\n=== BLEND CELL {i} (first 600 chars) ===')
        # Find the function
        idx = src.find('def ')
        print(src[idx:idx+600])
        break
