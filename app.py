from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from simulate import simulate_season, optimize

app = Flask(__name__)
df = pd.read_csv('nba_stats.csv')

# We need to ensure we have a SALARY column for optimization
# It needs to be added before simulate.py imports the data
if 'SALARY' not in df.columns:
    # Create a mock salary column based on fantasy points (FP) if it exists
    if 'FP' in df.columns:
        df['SALARY'] = (df['FP'] * 500).clip(3000, 12000).astype(int)
    else:
        # Use PTS as a proxy if FP doesn't exist
        df['SALARY'] = (df['PTS'] * 500).clip(3000, 12000).astype(int)
    
    # Save the updated DataFrame back to CSV to ensure simulate.py sees the SALARY column
    df.to_csv('nba_stats.csv', index=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    players = df['Player'].unique()  # Use 'Player' instead of 'PLAYER'
    if request.method == 'POST':
        roster = request.form.getlist('players')
        
        # Validate that at least one player is selected
        if not roster:
            return render_template('index.html', players=players, error="Please select at least one player")
        
        # Run simulations and optimization
        sims = simulate_season(roster)
        picks = optimize(roster, budget=60000)
        
        # Calculate summary statistics
        summary = {
            'mean': round(np.mean(sims), 1),
            '5th': round(np.percentile(sims, 5), 1),
            '95th': round(np.percentile(sims, 95), 1)
        }
        
        # Create player mapping using Team and Player name information
        player_map = {}
        for p in roster:
            player_row = df[df['Player'] == p]  # Use 'Player' instead of 'PLAYER'
            if not player_row.empty:
                team = player_row['Team'].iloc[0] if 'Team' in df.columns else ''
                player_map[p] = f"{p} ({team})" if team else p
            else:
                player_map[p] = p
        
        # Create salary map
        salary_map = df.set_index('Player')['SALARY'].to_dict()  # Use 'Player' instead of 'PLAYER'
        
        return render_template(
            'results.html',
            sims=sims.tolist(),
            summary=summary,
            picks=picks,
            player_map=player_map,
            salary_map=salary_map
        )
    
    return render_template('index.html', players=players)
    
if __name__ == '__main__':
    app.run(debug=True)