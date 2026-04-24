import streamlit as st
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib 

# --- Page Configuration ---
st.set_page_config(page_title="US Open 2026 AI Predictor", page_icon="🎾", layout="wide")

IOC_FLAGS = {
    'ITA': '🇮🇹', 'ESP': '🇪🇸', 'GER': '🇩🇪', 'USA': '🇺🇸', 'GBR': '🇬🇧',
    'SRB': '🇷🇸', 'AUS': '🇦🇺', 'RUS': '🇷🇺', 'DEN': '🇩🇰', 'NOR': '🇳🇴',
    'CZE': '🇨🇿', 'FRA': '🇫🇷', 'KAZ': '🇰🇿', 'CAN': '🇨🇦', 'GRE': '🇬🇷',
    'ARG': '🇦🇷', 'NED': '🇳🇱', 'JPN': '🇯🇵', 'POL': '🇵🇱', 'CHN': '🇨🇳',
    'BUL': '🇧🇬', 'CRO': '🇭🇷', 'SUI': '🇨🇭'
}

def get_flag(ioc):
    return IOC_FLAGS.get(ioc, '🏳️')

def convert_to_long_format(df):
    common_cols = ['tourney_id', 'tourney_name', 'tourney_date', 'match_num', 'minutes', 'score']
    
    winner_cols = [c for c in df.columns if c.startswith('winner_') or c.startswith('w_')]
    df_winner = df[common_cols + winner_cols].copy()
    df_winner.columns = [c.replace('winner_', 'player_').replace('w_', 'player_') for c in df_winner.columns]
    df_winner['result'] = 1
    df_winner['opponent_name'] = df['loser_name']

    loser_cols = [c for c in df.columns if c.startswith('loser_') or c.startswith('l_')]
    df_loser = df[common_cols + loser_cols].copy()
    df_loser.columns = [c.replace('loser_', 'player_').replace('l_', 'player_') for c in df_loser.columns]
    df_loser['result'] = 0
    df_loser['opponent_name'] = df['winner_name']

    return pd.concat([df_winner, df_loser], ignore_index=True)

# --- Data Initialization & Pipeline ---
@st.cache_resource
def initialize_system():
    base_dir = os.path.dirname(__file__)
    base_path = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'models') 
    
    try:
        dfs = []
        for year in [2023, 2024, 2025, 2026]:
            path = os.path.join(base_path, f"{year}.csv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
            else:
                print(f"Warning: File {year}.csv not found at {path}")
        
        if not dfs:
            raise FileNotFoundError("No data files found in the 'data' directory.")
            
        df_raw = pd.concat(dfs, ignore_index=True)
        df = convert_to_long_format(df_raw)
    
        # 1. Feature Engineering
        df = df.sort_values(by=['tourney_id', 'player_name', 'match_num'])
        df['acc_played_minutes'] = df.groupby(['tourney_id', 'player_name'])['minutes'].transform(lambda x: x.shift(1).fillna(0).cumsum())
        
        df['1st_srv_in'] = df['player_1stIn'] / df['player_svpt']
        df['1st_srv_won'] = df['player_1stWon'] / df['player_1stIn']
        df['bp_save_rate'] = df['player_bpSaved'] / df['player_bpFaced']
        df['ace_per_game'] = df['player_ace'] / df['player_SvGms']
        # df.fillna(0, inplace=True) # This is too broad and causes errors on string columns.

        df = df.sort_values(by=['player_name', 'tourney_date', 'match_num'])
        roll_cols = ['1st_srv_in', '1st_srv_won', 'bp_save_rate', 'ace_per_game', 'minutes']
        for col in roll_cols:
            df[f'avg_{col}_last_10'] = df.groupby('player_name')[col].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())

        # 2. H2H Feature Creation
        df['h2h_wins'] = df.groupby(['player_name', 'opponent_name'])['result'].transform(lambda x: x.shift(1).fillna(0).cumsum())
        df['h2h_total_matches'] = df.groupby(['player_name', 'opponent_name']).cumcount()
        df['h2h_win_rate'] = np.where(df['h2h_total_matches'] > 0, df['h2h_wins'] / df['h2h_total_matches'], 0.5)
        # df.fillna(0, inplace=True) # This is also too broad.

        # Targeted fill for numeric columns that may have NaNs from calculations, avoiding errors on string columns.
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        cols_to_fill = ['1st_srv_in', '1st_srv_won', 'bp_save_rate', 'ace_per_game'] + \
                       [f'avg_{col}_last_10' for col in roll_cols]
        df[cols_to_fill] = df[cols_to_fill].fillna(0)
        
        # 3. Create O(1) H2H Cache (Dictionary) for fast lookup during simulation
        h2h_latest = df.drop_duplicates(subset=['player_name', 'opponent_name'], keep='last')
        h2h_cache = {}
        for _, row in h2h_latest.iterrows():
            p1 = row['player_name']
            p2 = row['opponent_name']
            if p1 not in h2h_cache:
                h2h_cache[p1] = {}
            total = row['h2h_total_matches'] + 1
            wins = row['h2h_wins'] + row['result']
            h2h_cache[p1][p2] = {'total': total, 'win_rate': wins / total}

        # 4. Create Latest Stats Cache per player
        latest_stats = df.drop_duplicates(subset=['player_name'], keep='last').set_index('player_name')

        # 5. Model Loading and Feature Definition
        ml_features = ['player_ht', 'player_rank', 'player_rank_points', 'acc_played_minutes',
                       'avg_1st_srv_won_last_10', 'avg_bp_save_rate_last_10', 'avg_ace_per_game_last_10',
                       'h2h_total_matches', 'h2h_win_rate']
        
        try:
            model = joblib.load(os.path.join(model_dir, 'best_xgb_model.pkl'))
            scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            imputer = joblib.load(os.path.join(model_dir, 'imputer.pkl'))
        except Exception as e:
            st.error(f"Error loading models: {e}. Ensure .pkl files exist in the 'models' directory.")
            st.stop()

        # 6. Load Players List
        try:
            players_df = pd.read_csv(os.path.join(base_path, 'Players.csv'), header=None, names=['seed', 'name'])
            players_df['name'] = players_df['name'].str.strip()
            players_df = players_df[players_df['name'].str.upper() != 'NAME']
        except Exception as e:
            st.error(f"Error loading Players.csv: {e}")
            players_df = pd.DataFrame(columns=['seed', 'name'])

        # CẬP NHẬT: Trả về thêm imputer
        return df, model, scaler, imputer, players_df, ml_features, latest_stats, h2h_cache
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

# --- Logic Functions (Vectorized) ---
def predict_batch_proba(match_pairs, latest_stats, h2h_cache, model, scaler, imputer, ml_features, sim_minutes=None):
    """Predict probabilities for a batch of matches using matrix multiplication."""
    if not match_pairs:
        return []
        
    X_batch_p1 = np.zeros((len(match_pairs), len(ml_features)))
    X_batch_p2 = np.zeros((len(match_pairs), len(ml_features)))
    
    for i, (p1, p2) in enumerate(match_pairs):
        # Player 1 data 
        p1_data = latest_stats.loc[p1].copy() if p1 in latest_stats.index else pd.Series(dtype=float)
        
        h2h_info_p1 = h2h_cache.get(p1, {}).get(p2, {'total': 0, 'win_rate': 0.5})
        p1_data['h2h_total_matches'] = h2h_info_p1['total']
        p1_data['h2h_win_rate'] = h2h_info_p1['win_rate']
        
        if sim_minutes is not None:
            p1_data['acc_played_minutes'] = sim_minutes.get(p1, 0)
        else:
            p1_data['acc_played_minutes'] = 0
            
        # Player 2 data
        p2_data = latest_stats.loc[p2].copy() if p2 in latest_stats.index else pd.Series(dtype=float)
        
        h2h_info_p2 = h2h_cache.get(p2, {}).get(p1, {'total': 0, 'win_rate': 0.5})
        p2_data['h2h_total_matches'] = h2h_info_p2['total']
        p2_data['h2h_win_rate'] = h2h_info_p2['win_rate']
        
        if sim_minutes is not None:
            p2_data['acc_played_minutes'] = sim_minutes.get(p2, 0)
        else:
            p2_data['acc_played_minutes'] = 0
        
        # Fill in the feature vectors
        for j, feat in enumerate(ml_features):
            X_batch_p1[i, j] = p1_data.get(feat, np.nan)
            X_batch_p2[i, j] = p2_data.get(feat, np.nan)
            
    # Predict for Player 1
    X_imputed_p1 = imputer.transform(X_batch_p1)
    X_scaled_p1 = scaler.transform(X_imputed_p1)
    probs_p1 = model.predict_proba(X_scaled_p1)[:, 1]

    # Predict for Player 2
    X_imputed_p2 = imputer.transform(X_batch_p2)
    X_scaled_p2 = scaler.transform(X_imputed_p2)
    probs_p2 = model.predict_proba(X_scaled_p2)[:, 1]
    
    # Normalize probabilities to make the prediction symmetric
    sum_probs = probs_p1 + probs_p2
    probs = np.where(sum_probs == 0, 0.5, probs_p1 / sum_probs)
    
    return probs

def run_monte_carlo(draw_list, n_iter, latest_stats, h2h_cache, model, scaler, imputer, ml_features):
    results = {p: 0 for p in draw_list}
    prog_bar = st.progress(0)
    
    for i in range(n_iter):
        if i % max(1, n_iter // 10) == 0: prog_bar.progress(i/n_iter)
        current_round = list(draw_list)
        np.random.shuffle(current_round) # Shuffle players for a random draw each iteration
        
        # Track simulated minutes to simulate tournament fatigue properly
        sim_minutes = {p: 0 for p in draw_list}
        
        while len(current_round) > 1:
            next_round = []
            match_pairs = []
            
            # Create match pairs for the current round
            for j in range(0, len(current_round) - 1, 2):
                match_pairs.append((current_round[j], current_round[j+1]))
                
            probs = predict_batch_proba(match_pairs, latest_stats, h2h_cache, model, scaler, imputer, ml_features, sim_minutes)
            
            # Determine winners
            for idx, (p1, p2) in enumerate(match_pairs):
                winner = p1 if np.random.random() < probs[idx] else p2
                next_round.append(winner)
                
                # Increment fatigue for both players (average match ~120 mins)
                sim_minutes[p1] += 120
                sim_minutes[p2] += 120
                
            # Handle the case of an odd number of players (Bye)
            if len(current_round) % 2 != 0: 
                next_round.append(current_round[-1])
                
            current_round = next_round
            
        if current_round: 
            results[current_round[0]] += 1
            
    prog_bar.empty()
    return pd.DataFrame(list(results.items()), columns=['Player', 'Wins']).sort_values('Wins', ascending=False)

# --- APP FLOW ---
df, model, scaler, imputer, players_df, ml_features, latest_stats, h2h_cache = initialize_system()

st.title("🏆 US Open AI Predictor")
st.markdown(f"**Author**: Luke VU | **Data:** {len(df)} Matches  |  **Model:** XGBoost")
# --- Player Selection ---

all_db_players = sorted(df['player_name'].unique().tolist())
if 'selected_players' not in st.session_state:
    csv_names = players_df['name'].tolist()
    valid_default = [name for name in csv_names if name in all_db_players]
    st.session_state['selected_players'] = valid_default

# --- Tab Definition ---
tabs = st.tabs(["📖 Readme", "👥 1. Player Selection", "📊 2. Tournament Prediction", "⚔️ 3. Head to Head Prediction"])

# --- TAB 0: README  ---
with tabs[0]:
    st.header("Welcome!")
    st.markdown("This application uses Machine Learning (XGBoost) and Stochastic Modeling (Monte Carlo) to predict tennis match outcomes. The prediction relies on a standardized feature matrix capturing 4 core pillars:")  
    st.markdown("""
    * **Physical & Ranking:** Height (`player_ht`), ATP Rank (`player_rank`), and Rank Points (`player_rank_points`).
    * **Tournament Fatigue:** Accumulated minutes played (`acc_played_minutes`) in the current tournament to penalize exhausted players.
    * **Technical Attributes (Last 10 Games):** 1st Serve Win %(`avg_1st_srv_won_last_10`), Break Point Save % (`avg_bp_save_rate_last_10`), and Aces per game (`avg_ace_per_game_last_10`) to capture current momentum.
    * **Head-to-Head (H2H):** Total historical encounters and specific H2H Win Rate (`h2h_total_matches`, `h2h_win_rate`) between two players.""")

    st.markdown("#### Navigation")
    st.markdown("""
    | Tab     | Function | Explain|
    | :--- | :--- | :--- |
    | **👥 1. Player Selection** | Build your tournament pool. | Select up to 32 players from the pool to initialize the tournament. |
    | **📊 2. Tournament Prediction** | Predict the champion. | Runs 100-1000 times of Monte Carlo to simulate upsets and bracket progression and find the winner. |
    | **⚔️ 3. Head to Head** | Quick 1v1 prediction. | To find out the win probability between two selected players.| |
    """)
    
# --- TAB 1: PLAYER MANAGEMENT ---
with tabs[1]:
    remaining_options = [p for p in all_db_players if p not in st.session_state['selected_players']]
    selected = st.multiselect(
        "Select Players for the Simulation:",
        options=st.session_state['selected_players'] + remaining_options,
        default=st.session_state['selected_players'],
        max_selections=32
    )
    st.session_state['selected_players'] = selected
    st.info(f"Selected: {len(selected)} / 32 players.")

# --- TAB 2: TOURNAMENT SIMULATION ---
with tabs[2]:
    active_list = st.session_state['selected_players']
    if len(active_list) < 2:
        st.error("Please select more players in the 'Player Selection' tab.")
    else:
        st.markdown("**How it works**: The model predicts the winner of entire tournament for N times. \n\n"
                     "The winner probability is simply the percentage of total times they lift the trophy (e.g., 250 wins in 1,000 times = 25%).")
        st.markdown("**Choose iterations**: 100: Quick preview and 1,000: Slower, but more accurate.")
        
        col1, btn_col, msg_col = st.columns([2, 1, 2])
        
        with col1:
            n_sim = st.select_slider("Iterations", options=[100, 200, 500, 1000], value=100, label_visibility="collapsed")
            
        with btn_col:
            start_sim = st.button("🚀 Start ", type="primary", use_container_width=True)
            
        if start_sim:
            # Start the execution timer
            start_time = time.time()
            
            with st.spinner(f"Simulating {n_sim} scenarios..."):
                sim_res = run_monte_carlo(active_list, n_sim, latest_stats, h2h_cache, model, scaler, imputer, ml_features)
                st.session_state['sim_data'] = sim_res
            
            # Stop the timer and calculate execution time
            end_time = time.time()
            exec_time = end_time - start_time
            
            # Store the execution time in session state so it persists across UI re-renders
            st.session_state['exec_time'] = exec_time 
            
            with msg_col:
                # Display the execution time inside the success message
                st.success(f"Completed in {exec_time:.2f} seconds!")

        if 'sim_data' in st.session_state:
            data = st.session_state['sim_data']
            data['Win Prob (%)'] = (data['Wins'] / n_sim) * 100
            data = data.sort_values('Win Prob (%)', ascending=False)
            
            # Display execution time in the results header if available
            if 'exec_time' in st.session_state:
                st.markdown(f"### Results: ")
            else:
                st.markdown("### Results")
                
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                # Top 1
                top_winner = data.iloc[0]
                st.metric(label="🥇 Tournament Winner:", value=top_winner['Player'], delta=f"{top_winner['Win Prob (%)']:.1f}% Chance")
                
                st.divider()
                
                # Top 2 
                runner_up = data.iloc[1]
                st.metric(label="🥈 Runner-Up:", value=runner_up['Player'], delta=f"{runner_up['Win Prob (%)']:.1f}% Chance", delta_color="normal")

            with res_col2:
                # Top 5 Chart
                fig = px.bar(data.head(5), x='Win Prob (%)', y='Player', orientation='h', color='Win Prob (%)', color_continuous_scale='Greens')
                fig.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    margin=dict(l=0, r=0, t=0, b=0), 
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: HEAD-TO-HEAD & DETAILED ANALYSIS ---
with tabs[3]:
    active_list = st.session_state['selected_players']
    if len(active_list) < 2:
        st.error("Please select players in the '1. Player Selection' tab first.")
    else:
        col_a, col_b = st.columns(2)
        
        # 1. SELECT PLAYERS (UI Input)
        with col_a:
            p1 = st.selectbox("Player 1: ", active_list, index=0, key="h2h_p1")
        with col_b:
            p2_opts = [p for p in active_list if p != p1]
            p2 = st.selectbox("Player 2: ", p2_opts, index=0, key="h2h_p2")

        # 2. PROBABILITY & LATEST STATS
        p1_stats = df[df['player_name'] == p1].iloc[-1]
        p2_stats = df[df['player_name'] == p2].iloc[-1]
        
        prob = predict_batch_proba([(p1, p2)], latest_stats, h2h_cache, model, scaler, imputer, ml_features)[0]

        # 3. COMPACT METRICS 
        with col_a:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Height", f"{int(p1_stats['player_ht'])}cm" if p1_stats['player_ht'] > 0 else "N/A")
            m2.metric("Rank", f"#{int(p1_stats['player_rank'])}")
            m3.metric("Country", get_flag(p1_stats['player_ioc']))
            m4.metric("Age", f"{int(p1_stats['player_age'])}")
            m5.metric("Win Prob", f"{prob:.2%}")

        with col_b:
            m6, m7, m8, m9, m10 = st.columns(5)
            m6.metric("Height", f"{int(p2_stats['player_ht'])}cm" if p2_stats['player_ht'] > 0 else "N/A")
            m7.metric("Rank", f"#{int(p2_stats['player_rank'])}")
            m8.metric("Country", get_flag(p2_stats['player_ioc']))
            m9.metric("Age", f"{int(p2_stats['player_age'])}")
            m10.metric("Win Prob", f"{(1-prob):.2%}")

        # 4. H2H CONTEXT EXTRACTION LOGIC 
        h2h_data_p1 = df[(df['player_name'] == p1) & (df['opponent_name'] == p2)].copy()

        if not h2h_data_p1.empty:
            # 1. Avg form stats H2H
            cols_to_avg = ['avg_1st_srv_won_last_10', 'avg_bp_save_rate_last_10', 'avg_ace_per_game_last_10']
            p1_avg_stats = h2h_data_p1[cols_to_avg].mean()
            
            h2h_data_p2 = df[(df['player_name'] == p2) & (df['opponent_name'] == p1)]
            p2_avg_stats = h2h_data_p2[cols_to_avg].mean()
            
            # 2. Total minutes context H2H
            shared_tourneys = h2h_data_p1['tourney_id'].unique()
            
            # All matches
            df_p1_shared = df[(df['player_name'] == p1) & (df['tourney_id'].isin(shared_tourneys))]
            df_p2_shared = df[(df['player_name'] == p2) & (df['tourney_id'].isin(shared_tourneys))]
            
            # Average of max fatigue per tournament across shared tournaments
            p1_avg_max_fatigue = df_p1_shared.groupby('tourney_id')['acc_played_minutes'].max().mean()
            p2_avg_max_fatigue = df_p2_shared.groupby('tourney_id')['acc_played_minutes'].max().mean()

            chart_subtitle = f"<br><sup><i>(Average across {len(h2h_data_p1)} shared tournaments)</i></sup>"
        else:
            # Fallback: if no H2H history, use overall latest stats as a proxy for form and fatigue
            p1_avg_stats = p1_stats
            p2_avg_stats = p2_stats
            p1_avg_max_fatigue = p1_stats['acc_played_minutes']
            p2_avg_max_fatigue = p2_stats['acc_played_minutes']
            chart_subtitle = "<br><sup><i>(Latest Overall - No H2H history)</i></sup>"
            
        color_map = {p1: "#1f77b4", p2: "#d62728"}  
        
        # ---  LAYOUT 2x2 (COMPACT HEIGHTS) ---
        st.markdown(" ")
        r1_col1, r1_col2 = st.columns(2) 
        r2_col1, r2_col2 = st.columns(2) 

        # --- row 1 x column 1: BAR CHART (MAX FATIGUE AVERAGE) ---
        with r1_col1:
            fatigue_df = pd.DataFrame({
                'Player': [p1, p2], 
                'Avg Max Minutes': [p1_avg_max_fatigue, p2_avg_max_fatigue]
            })
            
            fig_fatigue = px.bar(
                fatigue_df, 
                x='Player', 
                y='Avg Max Minutes', 
                color='Player', 
                color_discrete_map=color_map, 
                text_auto='.1f', 
                title=f"Avg Fatigue per Tournament {chart_subtitle}"
            )
            
            fig_fatigue.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=240, showlegend=False)
            st.plotly_chart(fig_fatigue, use_container_width=True)

        # --- row 1 x column 2: RADAR CHART (AVERAGE FORM) ---
        with r1_col2:
            ACE_BENCHMARK = 1.2 
            categories = ['1st Serve Won', 'Break Point Save Rate', 'Ace per Game']
            
            p1_ace_norm = min(p1_avg_stats['avg_ace_per_game_last_10'] / ACE_BENCHMARK, 1.0)
            p2_ace_norm = min(p2_avg_stats['avg_ace_per_game_last_10'] / ACE_BENCHMARK, 1.0)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=[p1_avg_stats['avg_1st_srv_won_last_10'], p1_avg_stats['avg_bp_save_rate_last_10'], p1_ace_norm], theta=categories, fill='toself', name=p1, line_color='#1f77b4'))
            fig_radar.add_trace(go.Scatterpolar(r=[p2_avg_stats['avg_1st_srv_won_last_10'], p2_avg_stats['avg_bp_save_rate_last_10'], p2_ace_norm], theta=categories, fill='toself', name=p2, line_color='#d62728'))

            fig_radar.update_layout(
                title=f"Technical Form {chart_subtitle}",
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                margin=dict(l=30, r=30, t=40, b=10), height=240, showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- H2H PROCESS (PIE & TABLE ) ---
        if not h2h_data_p1.empty:
            p1_wins = h2h_data_p1['result'].sum()
            p2_wins = len(h2h_data_p1) - p1_wins
            
            with r2_col1:
                fig_pie = px.pie(values=[p1_wins, p2_wins], names=[p1, p2], 
                                 color=[p1, p2], color_discrete_map=color_map, hole=0.3, title=f"Win Distribution (Total: {len(h2h_data_p1)})")
                fig_pie.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=220)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with r2_col2:
                st.markdown("**Historical Matches**")
                h2h_data_p1['Winner'] = np.where(h2h_data_p1['result'] == 1, p1, p2)
                h2h_data_p1['Tourney'] = h2h_data_p1['tourney_name'] + " '" + h2h_data_p1['tourney_date'].astype(str).str[2:4]
                h2h_data_p1['Score'] = h2h_data_p1['score']
                st.dataframe(
                    h2h_data_p1[['Tourney', 'Score', 'Winner']], 
                    use_container_width=True, 
                    hide_index=True,
                    height=180 
                )
        else:
            st.info(f"No previous Head-to-Head matches found between {p1} and {p2}.")