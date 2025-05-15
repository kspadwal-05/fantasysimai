import numpy as np
import pandas as pd
import tensorflow as tf
from utils import mdn_sample, mdn_loss
from ortools.algorithms import pywrapknapsack_solver

# Load saved MDN model
df = pd.read_csv('nba_stats.tsv', sep='\t')
model = tf.keras.models.load_model('mdn_model', custom_objects={'loss': mdn_loss(5)})
FEATURES = ['PTS','AST','REB','STL','BLK','TOV','GP','MIN']

# Sample per-player fantasy points distribution
def project_fp(pid, n=5000):
    row = df[df.PLAYER==pid].iloc[0]
    x = row[FEATURES].values.reshape(1,-1)
    pi,mu,sig = model.predict(x)
    return mdn_sample(pi[0],mu[0],sig[0],n)

# Simulate season totals (e.g. 82 games)
def simulate_season(players, n_sim=10000):
    results = np.zeros((n_sim,len(players)))
    for i,p in enumerate(players):
        pts = project_fp(p,n_sim)
        gp = df[df.PLAYER==p].GP.iloc[0]
        results[:,i]=pts*gp
    return results.sum(axis=1)

# Optimize lineup under salary cap
def optimize(players, budget):
    values=[int(np.mean(project_fp(p))*100) for p in players]
    costs=[[int(df[df.PLAYER==p].SALARY.iloc[0]) for p in players]]
    solver=pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND,'opt')
    solver.Init(values,costs,[budget]); solver.Solve()
    return [players[i] for i in range(len(players)) if solver.BestSolutionContains(i)]