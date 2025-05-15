import numpy as np
import pandas as pd
from utils import mdn_sample, mdn_loss
import tensorflow as tf

# Load saved MDN model - reload it to ensure we get the updated CSV with SALARY column
df = pd.read_csv('nba_stats.csv')

# Add SALARY column if it doesn't exist
if 'SALARY' not in df.columns:
    # Create a mock salary column based on fantasy points (FP) if it exists
    if 'FP' in df.columns:
        df['SALARY'] = (df['FP'] * 500).clip(3000, 12000).astype(int)
    else:
        # Use PTS as a proxy if FP doesn't exist
        df['SALARY'] = (df['PTS'] * 500).clip(3000, 12000).astype(int)
    
    # Save the updated DataFrame back to CSV
    df.to_csv('nba_stats.csv', index=False)

# When loading model with custom loss, we need to create the loss function with our n_components
n_components = 5
model = tf.keras.models.load_model('mdn_model.keras', custom_objects={'loss': mdn_loss(n_components)})

FEATURES = ['PTS','AST','REB','STL','BLK','TOV','GP','MIN']

# Helper function to convert TensorFlow tensors to NumPy arrays if needed
def to_numpy(tensor):
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()  # If it's a TensorFlow tensor
    else:
        return tensor  # If it's already a NumPy array

# Sample per-player fantasy points distribution
def project_fp(pid, n=5000):
    # Use 'Player' instead of 'PLAYER' to match the CSV column name
    row = df[df.Player == pid]
    if row.empty:
        raise ValueError(f"Player {pid} not found in dataframe")
    row = row.iloc[0]
    
    # Extract the features, handling lowercase column names in the CSV
    features_data = []
    for feature in FEATURES:
        # Try both cases of the feature name
        if feature in row:
            features_data.append(row[feature])
        elif feature.title() in row:  # e.g., 'pts' -> 'Pts'
            features_data.append(row[feature.title()])
        elif feature.lower() in row:  # e.g., 'PTS' -> 'pts'
            features_data.append(row[feature.lower()])
        else:
            raise ValueError(f"Feature {feature} not found for player {pid}")
    
    x = np.array(features_data).reshape(1, -1).astype(np.float32)
    
    # Predict returns a single output: concatenated MDN parameters (pi, mu, sigma)
    y_pred = model.predict(x)
    
    # Split the output into mixture components
    pi_logits = y_pred[:, :n_components]
    mu = y_pred[:, n_components:2*n_components]
    sigma_logits = y_pred[:, 2*n_components:3*n_components]
    
    # Convert to appropriate formats - handling both TensorFlow and NumPy types
    pi = to_numpy(tf.nn.softmax(pi_logits))[0]
    mu = to_numpy(mu)[0]
    sigma = np.exp(to_numpy(sigma_logits)[0]) + 1e-6
    
    # Sample from mixture distribution
    return mdn_sample(pi, mu, sigma, n)

# Simulate season totals
def simulate_season(players, n_sim=10000):
    results = np.zeros((n_sim, len(players)))
    for i, p in enumerate(players):
        pts = project_fp(p, n_sim)
        # Use 'Player' instead of 'PLAYER'
        gp_row = df[df.Player == p]
        if gp_row.empty:
            raise ValueError(f"Player {p} not found in dataframe")
        gp = gp_row.GP.iloc[0]
        results[:, i] = pts * gp
    return results.sum(axis=1)

# Optimize lineup under salary cap
def optimize(players, budget):
    values = [int(np.mean(project_fp(p)) * 100) for p in players]
    
    # Get costs for all players
    # Use 'Player' instead of 'PLAYER'
    costs = [int(df[df.Player == p].SALARY.iloc[0]) for p in players]
    
    from ortools.algorithms.python import knapsack_solver
    
    # Create solver - using the correct constant name
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'opt')
    
    # Initialize solver with correct format
    solver.init(values, [costs], [budget])
    
    # Solve and get selected players
    solver.solve()
    selected = [players[i] for i in range(len(players)) if solver.best_solution_contains(i)]
    
    return selected