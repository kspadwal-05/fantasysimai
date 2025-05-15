# FantasySim AI for NBA Fantasy Leagues

Predicts fantasy points distributions and optimizes lineups using advanced ML.

## Data Requirements
Provide `nba_stats.csv` with columns:

```
PLAYER,TEAM,AGE,GP,W,L,MIN,PTS,FGM,FGA,FG%,3PM,3PA,3P%,FTM,FTA,FT%,OREB,DREB,REB,AST,TOV,STL,BLK,PF,FP,DD2,TD3,+/−,SALARY
```

- **Features**: PTS, AST, REB, STL, BLK, TOV, GP, MIN
- **Target**: FP (fantasy points per game)
- **Salary**: integer cost for lineup optimization

## Setup & Run
```bash
pip install -r requirements.txt
python data_prep.py
python model_train.py
python simulate.py
python app.py
Open http://127.0.0.1:5000
```