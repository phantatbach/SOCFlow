import pandas as pd

def get_coordinates(token, model):
    df = pd.read_csv(f'../input/{token}.tsne.30.tsv', sep='\t')

    # # Get the coordinates for each token
    # df['x'] = df[model + '.x']
    # df['y'] = df[model + '.y']

    # Get the coordinates for each token
    df['x'] = pd.to_numeric(df[model + '.x'], errors='coerce').round(4)  # Convert to numeric and round to 4 decimals
    df['y'] = pd.to_numeric(df[model + '.y'], errors='coerce').round(4)
    
    df = df[['_id', 'x', 'y']]
    df.to_csv(f'../input/{token}-{model}.tsne.30.tsv', sep='\t', index=False)
    print(f'The coordinates of {token}-{model} were extracted')

def get_senses(token, model):
    df = pd.read_csv(f'../input/{token}-{model}.tsne.30.tsv', sep='\t')
    sense_df = pd.read_csv(f'../input/{token}.variables.tsv', sep='\t')
    df['senses'] = df['_id'].map(sense_df.set_index('_id')['senses'])
    df.to_csv(f'../input/{token}-{model}.tsne.30.tsv', sep='\t', index=False)
    print(f'The senses of {token}-{model} were added')

def get_model_data(token, model):
    get_coordinates(token, model)
    get_senses(token, model)
    print('The model data was extracted')