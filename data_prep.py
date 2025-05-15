import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

# Fields we absolutely need for modeling and consistency checks
REQUIRED = [
    'Player', 'Team', 'Age', 'GP', 'Min', 'PTS', 'AST', 'REB',
    'STL', 'BLK', 'TOV', '+/-', 'FP'
]

# Features used as input to the model
FEATURES = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'GP', 'Min']
TARGET = 'FP'

def load_stats(path="nba_stats.tsv"):
    df = pd.read_csv(path, sep='\t')
    logging.info(f"Loaded {len(df)} rows from {path}")
    
    # Ensure necessary columns exist
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    
    # Drop rows with missing required values
    df = df.dropna(subset=REQUIRED)
    
    return df

def prepare_training_data(df):
    """
    Extracts feature matrix X and target vector y for training.
    """
    X = df[FEATURES].values
    y = df[TARGET].values
    return X, y

if __name__ == '__main__':
    df = load_stats()
    X, y = prepare_training_data(df)
    pd.to_pickle((X, y), 'training_data.pkl')
    logging.info("Saved processed training data to training_data.pkl")