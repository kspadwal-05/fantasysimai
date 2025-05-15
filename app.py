from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from simulate import simulate_season, optimize

app=Flask(__name__)
df=pd.read_csv('nba_stats.tsv', sep='\t')

@app.route('/',methods=['GET','POST'])
def index():
    players=df.PLAYER.unique()
    if request.method=='POST':
        roster=request.form.getlist('players')
        sims=simulate_season(roster)
        picks=optimize(roster,budget=60000)
        summary={'mean':round(np.mean(sims),1),'5th':round(np.percentile(sims,5),1),'95th':round(np.percentile(sims,95),1)}
        return render_template('results.html',sims=sims.tolist(),summary=summary,picks=picks,player_map=df.set_index('PLAYER')['PLAYER'].to_dict(),salary_map=df.set_index('PLAYER')['SALARY'].to_dict())
    return render_template('index.html',players=players)

if __name__=='__main__': app.run(debug=True)