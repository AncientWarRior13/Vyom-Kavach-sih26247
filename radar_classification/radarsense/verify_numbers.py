import re

with open('Final_report.md', 'r', encoding='utf-8') as f:
    text = f.read()

snippets = [
    '30.4%', '34.2%', '91.6%', '0.0073', '0.0014', '0.868', '0.54', '0.06', '47.7%', '0.849', '0.882', '92.2%'
]

for s in snippets:
    matches = [m.start() for m in re.finditer(re.escape(s), text)]
    print(f'Match for "{s}": {len(matches)} occurrences')
