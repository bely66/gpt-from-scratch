import pandas as pd

char_lines = 'Rick'
df = pd.read_csv('data/RickAndMortyScripts.csv')

lines = df[df['name'] == char_lines]['line'].values

with open(f'data/{char_lines}_lines.txt', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')