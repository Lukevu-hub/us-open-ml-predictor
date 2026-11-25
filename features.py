"""
Shared data cleaning and feature engineering for the US Open ML Predictor.
Imported by both the training notebook and the Streamlit app, so features
are always computed the same way (no train/serve skew).
"""
import os

import numpy as np
import pandas as pd

# Features compared between the two players (model sees P1 - P2)
PLAYER_FEATURES = ['log_rank', 'player_rank_points', 'player_ht', 'acc_played_minutes',
                   'avg_1st_srv_in_last_10', 'avg_1st_srv_won_last_10',
                   'avg_bp_save_rate_last_10', 'avg_ace_per_game_last_10', 'win_rate_last_10']

# Final model inputs, in the exact order the saved models expect
FEATURE_COLS = ['diff_' + f for f in PLAYER_FEATURES] + ['h2h_win_rate', 'h2h_matches']


def load_raw(data_dir, years=(2023, 2024, 2025, 2026)):
    """Read the yearly CSV files and combine them (one row per match)."""
    dfs = [pd.read_csv(os.path.join(data_dir, f'{y}.csv')) for y in years]
    return pd.concat(dfs, ignore_index=True)


def clean_matches(df):
    """Remove incomplete matches, fill missing minutes, add a unique match_id."""
    # 1. Remove
    df = df.copy()
    df = df[df['match_num'].notnull()]
    df = df[~df['tourney_name'].str.contains('Davis')]                                  # Not include Davis CUP
    df = df[df['w_SvGms'].notnull() & df['l_SvGms'].notnull()]                          # Remove missing points data
    df = df[~df['score'].str.contains('W/O|RET|DEF', na=False)]                         # Remove matches with retirement/ walkover/ Default
    df['tourney_name'] = df['tourney_name'].replace('Rio de Janeiro', 'Rio De Janeiro') # Standardize tournament name
    
    # 2. Fill missing minutes
    def count_sets(score_str):
        if pd.isna(score_str):
            return None
        return len(str(score_str).split())  
    df['set_count'] = df['score'].apply(count_sets)
    avg_minutes_by_set = df.groupby('set_count')['minutes'].mean().round(0).reset_index()
    def fill_minutes(row):
        if pd.isna(row['minutes']):
            return avg_minutes_by_set[avg_minutes_by_set['set_count'] == row['set_count']]['minutes'].values[0]
        return row['minutes']
    df['minutes'] = df.apply(fill_minutes, axis=1)
    df = df.drop(columns=['set_count'])
    # 3. Reset index and add match_id
    df = df.reset_index(drop=True)                                                      # Reset index after cleaning
    df['match_id'] = np.arange(len(df)) + 1                                             # Add match_id column
    return df



def convert_to_long_format(df):
    """Two rows per match, one per player (result = 1 winner, 0 loser)."""
    # 1. Maintain common columns for both Winner and Loser groups
    common_cols = ['match_id','tourney_id','tourney_name', 'tourney_date','match_num','surface', 'tourney_level','score','minutes']

    # 2. Create Winner group by removing 'winner_' and 'w_'
    winner_cols = [c for c in df.columns if c.startswith('winner_') or c.startswith('w_')]
    df_winner = df[common_cols + winner_cols].copy()
    df_winner.columns = [c.replace('winner_', 'player_').replace('w_', 'player_') for c in df_winner.columns]

    # 3. Compute result column = 1 (Win) and create opponent_id
    df_winner['result'] = 1
    df_winner['opponent_id'] = df['loser_id']
    df_winner['opponent_name'] = df['loser_name']

    # 4. Create Loser Group by removing 'loser_' and 'l_'
    loser_cols = [c for c in df.columns if c.startswith('loser_') or c.startswith('l_')]
    df_loser = df[common_cols + loser_cols].copy()
    df_loser.columns = [c.replace('loser_', 'player_').replace('l_', 'player_') for c in df_loser.columns]

    # 5. Compute result column = 0 (Loss) and create opponent_id
    df_loser['result'] = 0
    df_loser['opponent_id'] = df['winner_id']
    df_loser['opponent_name'] = df['winner_name']

    # 6. Concatenate winner and loser tables
    df_long = pd.concat([df_winner, df_loser], ignore_index=True)

    # 7. Sort by match_id to ensure that winner and loser of the same match are adjacent
    df_long = df_long.sort_values(by=['match_id']).reset_index(drop=True)

    return df_long


def add_fatigue(df):
    """Minutes already played in the same tournament before each match."""
    df = df.copy()

    # 1. Sort by tournament, player and match order
    df = df.sort_values(by=['tourney_id', 'player_id', 'match_num'])

    # 2. Minutes played BEFORE each match: shift(1) excludes the current match
    def acc_minutes(series):
        return series.shift(1).fillna(0).cumsum()

    # 3. Accumulate within each tournament, per player
    df['acc_played_minutes'] = df.groupby(['tourney_id', 'player_id'])['minutes'].transform(acc_minutes)
    return df


def add_form(df):
    """Serve ratios, last-10-match rolling form, win rate and log rank."""
    df = df.copy()
    # 1. Sorting Data
    df = df.sort_values(by=['player_id', 'tourney_date','tourney_id','match_num'])

    # 2. New feature with Clipping (Max = 1)
    df['1st_srv_in'] = (df['player_1stIn'] / df['player_svpt']).clip(upper=1)
    df['1st_srv_won'] = (df['player_1stWon'] / df['player_1stIn']).clip(upper=1)
    df['bp_save_rate'] = (df['player_bpSaved'] / df['player_bpFaced']).clip(upper=1)
    df['ace_per_game'] = df['player_ace'] / df['player_SvGms'] 

    ratio_cols = ['1st_srv_in', '1st_srv_won', 'bp_save_rate', 'ace_per_game']
    df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaN


    # 3. Apply to features
    features_to_roll = ['1st_srv_in', '1st_srv_won', 'bp_save_rate', 'ace_per_game', 'minutes']
    for col in features_to_roll:
        df[f'avg_{col}_last_10'] = df.groupby('player_id')[col].transform(
            lambda s: s.shift(1).rolling(window=10, min_periods=1).mean())
    #4. New Features
    df['win_rate_last_10']= df.groupby('player_id')['result'].transform(
        lambda s: s.shift(1).rolling(window=10, min_periods=1).mean())
    df['log_rank']= np.log(df['player_rank'].replace(0, np.nan))  # Replace 0 with NaN to avoid log(0) issues
    return df


def add_h2h(df):
    """Head-to-head record before each match."""
    df = df.copy()
    #1. Sorting:
    df = df.sort_values(by=['player_id', 'opponent_id', 'tourney_date', 'tourney_id', 'match_num']).reset_index(drop=True)

    # 2. Total H2H Wins
    df['h2h_wins'] = df.groupby(['player_id', 'opponent_id'])['result'].transform(lambda x: x.shift(1).fillna(0).cumsum())

    # 3. Total H2H matches
    df['h2h_total_matches'] = df.groupby(['player_id', 'opponent_id']).cumcount() 

    # 4. H2H Win Rate
    df['h2h_win_rate'] = np.where(
        df['h2h_total_matches'] > 0,
        df['h2h_wins'] / df['h2h_total_matches'],
        0.5 ) # If never meet 50-50%
    return df


def build_match_dataset(df, max_rank=100, seed=42):
    """One row per match: random P1/P2, features = P1 - P2, target = 1 if P1 won."""
    # 1. Split into winner rows and loser rows, aligned by match_id
    w = df[df['result'] == 1].set_index('match_id').sort_index()
    l = df[df['result'] == 0].set_index('match_id').sort_index()

    # 2. Keep matches where BOTH players are ranked <= max_rank
    keep = (w['player_rank'] <= max_rank) & (l['player_rank'] <= max_rank)
    w, l = w[keep], l[keep]

    # 3. Randomly choose who is P1 (fixed seed -> same result every run)
    rng = np.random.default_rng(seed)
    flip = rng.random(len(w)) < 0.5          # True -> the LOSER becomes P1

    # 4. P1 / P2 feature arrays (PLAYER_FEATURES is defined at the top of this file)
    W = w[PLAYER_FEATURES].to_numpy(float)
    L = l[PLAYER_FEATURES].to_numpy(float)
    P1 = np.where(flip[:, None], L, W)
    P2 = np.where(flip[:, None], W, L)

    # 5. Build the match table
    matches = pd.DataFrame(P1 - P2, columns=['diff_' + f for f in PLAYER_FEATURES])
    matches['h2h_win_rate'] = np.where(flip, l['h2h_win_rate'], w['h2h_win_rate'])  # P1's view
    matches['h2h_matches'] = w['h2h_total_matches'].values   # same for both players
    matches['target'] = np.where(flip, l['result'], w['result'])  # P1's result: 1 = P1 won

    # 6. Info columns (for checking and for the time split)
    matches['tourney_date'] = w['tourney_date'].values
    matches['tourney_id'] = w['tourney_id'].values
    matches['match_num'] = w['match_num'].values
    matches['p1_name'] = np.where(flip, l['player_name'], w['player_name'])
    matches['p2_name'] = np.where(flip, w['player_name'], l['player_name'])

    # 7. Sort by time
    return matches.sort_values(['tourney_date', 'tourney_id', 'match_num']).reset_index(drop=True) 


def prepare(data_dir):
    """Raw CSVs -> long table with all pre-match features."""
    df = clean_matches(load_raw(data_dir))
    df = convert_to_long_format(df)
    return add_h2h(add_form(add_fatigue(df)))