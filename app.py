import html
import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import features as F

# --- Page Configuration ---
st.set_page_config(page_title="US Open 2026 ML Predictor", page_icon="🎾", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'match_model.pkl')
MAX_PLAYERS = 32

IOC_FLAGS = {
    'ITA': '🇮🇹', 'ESP': '🇪🇸', 'GER': '🇩🇪', 'USA': '🇺🇸', 'GBR': '🇬🇧',
    'SRB': '🇷🇸', 'AUS': '🇦🇺', 'RUS': '🇷🇺', 'DEN': '🇩🇰', 'NOR': '🇳🇴',
    'CZE': '🇨🇿', 'FRA': '🇫🇷', 'KAZ': '🇰🇿', 'CAN': '🇨🇦', 'GRE': '🇬🇷',
    'ARG': '🇦🇷', 'NED': '🇳🇱', 'JPN': '🇯🇵', 'POL': '🇵🇱', 'CHN': '🇨🇳',
    'BUL': '🇧🇬', 'CRO': '🇭🇷', 'SUI': '🇨🇭', 'BRA': '🇧🇷', 'CHI': '🇨🇱',
    'POR': '🇵🇹', 'BEL': '🇧🇪', 'HUN': '🇭🇺', 'AUT': '🇦🇹', 'FIN': '🇫🇮',
}


def get_flag(ioc):
    return IOC_FLAGS.get(ioc, '🏳️')


P1_COLOR, P2_COLOR = "#1f77b4", "#d62728"     # blue / red, used for every Player 1 / Player 2 element


def fmt_int(value, prefix='', suffix='', min_value=None):
    """Format a number as an int, or 'N/A' if it is missing (avoids int(NaN) crashes)."""
    if pd.isna(value) or (min_value is not None and value < min_value):
        return 'N/A'
    return f'{prefix}{int(value)}{suffix}'


def player_card(name, s, prob, color):
    """A colored card with a player's key stats and win probability (HTML)."""
    ioc = s.get('player_ioc')
    ioc = ioc if isinstance(ioc, str) else ''
    stats = [('Rank', fmt_int(s['player_rank'], prefix='#')),
             ('Country', f"{get_flag(ioc)} {ioc}"),
             ('Height', fmt_int(s['player_ht'], suffix=' cm', min_value=1)),
             ('Age', fmt_int(s.get('player_age')))]
    cells = ''.join(
        f"<div><div style='font-size:0.75rem; opacity:0.7;'>{k}</div>"
        f"<div style='font-size:1.05rem; font-weight:600;'>{v}</div></div>" for k, v in stats)
    return f"""
<div style="border:1px solid {color}66; border-top:6px solid {color}; border-radius:10px;
            padding:12px 16px; background:{color}14; margin-top:4px;">
  <div style="color:{color}; font-size:1.3rem; font-weight:700;">{html.escape(name)}</div>
  <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:12px; margin-top:8px;">
    <div style="display:grid; grid-template-columns:repeat(4, auto); gap:6px 18px;">{cells}</div>
    <div style="text-align:right;">
      <div style="font-size:0.75rem; opacity:0.7;">Win probability</div>
      <div style="color:{color}; font-size:2rem; font-weight:800; line-height:1.1;">{prob:.1%}</div>
    </div>
  </div>
</div>"""


# --- Data & Model (cached) ---
@st.cache_resource
def initialize_system():
    try:
        long = F.prepare(DATA_DIR)              # same cleaning + features as the notebook
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading models: {e}. Run the notebook to create models/match_model.pkl.")
        st.stop()
    if bundle.get('features') != F.FEATURE_COLS:
        st.error("The saved models use a different feature list than features.py. Re-run the notebook.")
        st.stop()

    snapshot = F.player_snapshot(long)
    h2h = F.h2h_table(long)
    minutes_pool = F.slam_minutes(long)

    try:
        players_df = pd.read_csv(os.path.join(DATA_DIR, 'Players.csv'), header=None, names=['seed', 'name'])
        players_df['name'] = players_df['name'].astype(str).str.strip()
        players_df = players_df[players_df['name'].str.upper() != 'NAME']
        players_df['seed'] = pd.to_numeric(players_df['seed'], errors='coerce')
    except Exception as e:
        st.warning(f"Could not load Players.csv ({e}). Starting with an empty selection.")
        players_df = pd.DataFrame(columns=['seed', 'name'])

    return long, bundle, snapshot, h2h, minutes_pool, players_df


# --- Bracket helpers ---
def seeded_slot_order(size):
    """Standard bracket order for seeds 1..size, e.g. size 8 -> [1, 8, 4, 5, 2, 7, 3, 6]."""
    order = [1]
    while len(order) < size:
        n = len(order) * 2
        order = [s for x in order for s in (x, n + 1 - x)]
    return np.array(order) - 1                  # 0-based seed positions


def order_by_seed(names, players_df, snapshot):
    """Sort players by tournament seed, then by ATP rank for unseeded players."""
    seeds = dict(zip(players_df['name'], players_df['seed']))
    return sorted(names, key=lambda n: (seeds.get(n, np.inf) if pd.notna(seeds.get(n, np.nan)) else np.inf,
                                        snapshot.loc[n, 'player_rank'] if pd.notna(snapshot.loc[n, 'player_rank']) else np.inf))


# --- Monte Carlo (vectorized across all iterations) ---
def run_monte_carlo(names, n_iter, snapshot, h2h, model, minutes_pool, seeded, seed=None):
    """
    Simulates the full knockout n_iter times. Each round makes ONE batched model call
    for all iterations, so 1,000 tournaments take well under a second.
    Returns title and final probabilities per player.
    """
    rng = np.random.default_rng(seed)
    n = len(names)
    feats = snapshot.loc[names, F.PLAYER_FEATURES].to_numpy(float)
    h2h_rate, h2h_n = F.h2h_matrices(names, h2h)

    size = 1 << (n - 1).bit_length()           # next power of two; missing spots are byes
    slot_pos = seeded_slot_order(size)          # which seed position sits in each slot
    if seeded:
        pos_to_player = np.tile(np.arange(n), (n_iter, 1))   # names are already in seed order
    else:
        pos_to_player = rng.permuted(np.tile(np.arange(n), (n_iter, 1)), axis=1)
    slots = np.where(slot_pos < n, pos_to_player[:, np.minimum(slot_pos, n - 1)], -1)

    fatigue = np.zeros((n_iter, n))
    finals = np.zeros(n)
    while slots.shape[1] > 1:
        if slots.shape[1] == 2:
            finals += np.bincount(slots[slots >= 0], minlength=n)
        a, b = slots[:, 0::2], slots[:, 1::2]
        winners = np.where(a < 0, b, a)         # bye -> the other player advances
        it, m = np.nonzero((a >= 0) & (b >= 0))
        if len(it):
            ia, ib = a[it, m], b[it, m]
            A, B = feats[ia].copy(), feats[ib].copy()
            A[:, F.MINUTES_IDX] = fatigue[it, ia]
            B[:, F.MINUTES_IDX] = fatigue[it, ib]
            p = F.predict_proba_symmetric(model, A, B, h2h_rate[ia, ib], h2h_n[ia, ib])
            winners[it, m] = np.where(rng.random(len(p)) < p, ia, ib)
            # Fatigue: each match lasts a random Grand Slam match length
            mins = rng.choice(minutes_pool, size=len(p))
            fatigue[it, ia] += mins
            fatigue[it, ib] += mins
        slots = winners

    titles = np.bincount(slots[:, 0], minlength=n)
    return pd.DataFrame({
        'Player': names,
        'Title (%)': titles / n_iter * 100,
        'Final (%)': finals / n_iter * 100,
    }).sort_values('Title (%)', ascending=False).reset_index(drop=True)


def match_probability(p1, p2, snapshot, h2h, model):
    names = [p1, p2]
    feats = snapshot.loc[names, F.PLAYER_FEATURES].to_numpy(float)
    rate, count = F.h2h_matrices(names, h2h)
    return float(F.predict_proba_symmetric(model, feats[[0]], feats[[1]], [rate[0, 1]], [count[0, 1]])[0])


# --- APP FLOW ---
long, bundle, snapshot, h2h, minutes_pool, players_df = initialize_system()

# --- Model choice (both models tied on the test set, so the user can compare them) ---
model_names = list(bundle['models'])
default_index = model_names.index(bundle.get('default', model_names[0]))
MODEL_HELP = ("Both models scored the same on the test set (log loss 0.606). "
              "Switch to see how their predictions differ.")


def model_picker(key):
    """Model dropdown. Each tab has its own, so both models can be compared side by side."""
    name = st.selectbox("Model", model_names, index=default_index, key=key, help=MODEL_HELP)
    return name, bundle['models'][name]


trained = str(bundle.get('trained_until', ''))
st.title("🏆 US Open ML Predictor")
st.markdown(f"**Author**: Luke Vu | **Data:** {long['match_id'].nunique():,} matches  |  "
            f"**Models:** {' & '.join(model_names)}  |  **Trained until:** {trained[:4]}-{trained[4:6]}")

all_db_players = sorted(snapshot.index.tolist())
if 'selected_players' not in st.session_state:
    st.session_state['selected_players'] = [n for n in players_df['name'] if n in snapshot.index][:MAX_PLAYERS]

tabs = st.tabs(["📖 Readme", "👥 1. Player Selection", "📊 2. Tournament Prediction", "⚔️ 3. Head to Head Prediction"])

# --- TAB 0: README ---
with tabs[0]:
    st.header("Welcome!")
    st.markdown("This app uses Machine Learning (Logistic Regression and XGBoost) and Monte Carlo simulation to "
                "predict tennis match outcomes. For every match the model compares the two players on 4 pillars "
                "(each feature is the **difference** between Player 1 and Player 2):")
    st.markdown("""
    * **Ranking & Physical:** ATP rank (log scale), rank points and height.
    * **Tournament Fatigue:** Minutes already played in the current tournament (`acc_played_minutes`).
    * **Form (last 10 matches):** 1st serve in %, 1st serve won %, break points saved %, aces per service game and win rate.
    * **Head-to-Head (H2H):** Number of previous meetings and Player 1's H2H win rate.""")
    metrics = bundle.get('metrics')
    if metrics:
        st.markdown("#### Model performance (held-out test set)")
        st.dataframe(pd.DataFrame(metrics).T.round(3), width='stretch')

    st.markdown("#### Navigation")
    st.markdown("""
    | Tab | Function | Explanation |
    | :--- | :--- | :--- |
    | **👥 1. Player Selection** | Build your tournament pool. | Select up to 32 players. |
    | **📊 2. Tournament Prediction** | Predict the champion. | Simulates the whole draw 100–5,000 times to capture upsets. |
    | **⚔️ 3. Head to Head** | Quick 1v1 prediction. | Win probability between two selected players. |
    """)

# --- TAB 1: PLAYER SELECTION ---
with tabs[1]:
    selected = st.multiselect(
        "Select Players for the Simulation:",
        options=all_db_players,
        default=[p for p in st.session_state['selected_players'] if p in all_db_players],
        max_selections=MAX_PLAYERS,
    )
    st.session_state['selected_players'] = selected
    st.info(f"Selected: {len(selected)} / {MAX_PLAYERS} players.")
    if selected:
        view = snapshot.loc[selected, ['player_rank', 'player_ioc', 'win_rate_last_10', 'matches_in_data']].copy()
        view['win_rate_last_10'] = (view['win_rate_last_10'] * 100).round(0)
        view.columns = ['Rank', 'Country', 'Win % (last 10)', 'Matches in data']
        st.dataframe(view.sort_values('Rank'), width='stretch')

# --- TAB 2: TOURNAMENT SIMULATION ---
with tabs[2]:
    active_list = st.session_state['selected_players']
    if len(active_list) < 2:
        st.error("Please select at least 2 players in the 'Player Selection' tab.")
    else:
        st.markdown("**How it works**: the model plays out the entire tournament N times. "
                    "A player's title probability is the share of simulations they win "
                    "(e.g. 250 titles in 1,000 runs = 25%).")
        with st.expander("How is each match decided?"):
            st.markdown(
                    "1. The model gives **p**, Player 1's chance to win, from ranking, form, fatigue and head-to-head.\n"
                    "2. A random number **r** between 0 and 1 is drawn: **Player 1 wins if r < p**, "
                    "so a 70% favorite wins about 7 times out of 10.\n"
                    "3. Match length is drawn from real Grand Slam matches and added to both players' fatigue.\n"
                    "4. Winners advance until one champion remains.\n\n"
                    "**Why random?** About a third of real matches are upsets. If the favorite always advanced, "
                    "the top seed would win 100% of the time."
                )

        col0, col1, col2, btn_col = st.columns([2, 2, 2, 1])
        with col0:
            model_name, model = model_picker('model_tournament')
        with col1:
            n_sim = st.select_slider("Iterations", options=[100, 500, 1000, 2000, 5000], value=1000)
        with col2:
            draw_mode = st.radio("Draw", ["Seeded bracket", "Random draw each run"], horizontal=True,
                                 help="Seeded: uses the seeds in Players.csv (unseeded players ordered by rank), "
                                      "so top seeds only meet late. Random: reshuffles the draw every run.")
        with btn_col:
            st.write("")
            start_sim = st.button("🚀 Start", type="primary", width='stretch')

        if start_sim:
            start_time = time.time()
            seeded = draw_mode == "Seeded bracket"
            names = order_by_seed(active_list, players_df, snapshot) if seeded else list(active_list)
            with st.spinner(f"Simulating {n_sim:,} tournaments..."):
                res = run_monte_carlo(names, n_sim, snapshot, h2h, model, minutes_pool, seeded)
            # Store the settings WITH the result, so moving the slider later doesn't change the numbers
            st.session_state['sim'] = {'data': res, 'n_sim': n_sim, 'draw': draw_mode, 'model': model_name,
                                       'players': list(active_list), 'secs': time.time() - start_time}

        sim = st.session_state.get('sim')
        if sim:
            data = sim['data']
            st.success(f"{sim['n_sim']:,} simulations ({sim['draw'].lower()}, {sim['model']}) "
                       f"completed in {sim['secs']:.2f} s.")
            if sorted(sim['players']) != sorted(active_list) or sim['model'] != model_name:
                st.warning("The player selection or model changed since this run. Press Start to update.")

            st.markdown("### Results")
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                fav = data.iloc[0]
                st.metric("🥇 Title favorite", fav['Player'], f"{fav['Title (%)']:.1f}% title chance")
                st.divider()
                second = data.iloc[1]
                st.metric("🥈 2nd favorite", second['Player'], f"{second['Title (%)']:.1f}% title chance")

            with res_col2:
                top = data.head(8)
                fig = px.bar(top, x='Title (%)', y='Player', orientation='h', color='Title (%)',
                             color_continuous_scale='Greens', hover_data={'Final (%)': ':.1f'})
                fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                                  margin=dict(l=0, r=0, t=0, b=0), height=350, coloraxis_showscale=False)
                st.plotly_chart(fig, width='stretch')

            with st.expander("Full table"):
                st.dataframe(data.round(1), width='stretch', hide_index=True)

# --- TAB 3: HEAD-TO-HEAD ---
with tabs[3]:
    active_list = st.session_state['selected_players']
    if len(active_list) < 2:
        st.error("Please select at least 2 players in the '1. Player Selection' tab first.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            p1 = st.selectbox("🔵 Player 1:", active_list, index=0, key="h2h_p1")
        with col_b:
            p2 = st.selectbox("🔴 Player 2:", [p for p in active_list if p != p1], index=0, key="h2h_p2")

        h2h_model_name, h2h_model = model_picker('model_h2h')
        s1, s2 = snapshot.loc[p1], snapshot.loc[p2]
        prob = match_probability(p1, p2, snapshot, h2h, h2h_model)

        # Player cards in each player's color
        for col, name, s, pr, color in [(col_a, p1, s1, prob, P1_COLOR), (col_b, p2, s2, 1 - prob, P2_COLOR)]:
            with col:
                st.markdown(player_card(name, s, pr, color), unsafe_allow_html=True)

        # Win probability bar: blue share vs red share
        st.markdown(f"""
<div style="display:flex; height:30px; border-radius:8px; overflow:hidden; margin:14px 0 4px;
            font-weight:700; color:white; font-size:0.9rem;">
  <div style="width:{prob * 100:.1f}%; background:{P1_COLOR}; display:flex; align-items:center;
              padding-left:10px; white-space:nowrap;">{prob:.0%}</div>
  <div style="width:{(1 - prob) * 100:.1f}%; background:{P2_COLOR}; display:flex; align-items:center;
              justify-content:flex-end; padding-right:10px; white-space:nowrap;">{1 - prob:.0%}</div>
</div>""", unsafe_allow_html=True)
        st.caption(f"Win probability from {h2h_model_name}.")

        h2h_p1 = long[(long['player_name'] == p1) & (long['opponent_name'] == p2)].copy()
        color_map = {p1: P1_COLOR, p2: P2_COLOR}

        st.markdown(" ")
        r1_col1, r1_col2 = st.columns(2)
        r2_col1, r2_col2 = st.columns(2)

        # Fatigue: average minutes already played before their last match, in tournaments both played
        with r1_col1:
            if not h2h_p1.empty:
                shared = h2h_p1['tourney_id'].unique()
                subtitle = f"<br><sup><i>(Average across {len(shared)} shared tournaments)</i></sup>"
                vals = [long[(long['player_name'] == p) & long['tourney_id'].isin(shared)]
                        .groupby('tourney_id')['acc_played_minutes'].max().mean() for p in (p1, p2)]
            else:
                subtitle = "<br><sup><i>(All tournaments - no H2H history)</i></sup>"
                vals = [long[long['player_name'] == p].groupby('tourney_id')['acc_played_minutes'].max().mean()
                        for p in (p1, p2)]
            fatigue_df = pd.DataFrame({'Player': [p1, p2], 'Avg Minutes Before Last Match': vals})
            fig_f = px.bar(fatigue_df, x='Player', y='Avg Minutes Before Last Match', color='Player',
                           color_discrete_map=color_map, text_auto='.0f',
                           title=f"Avg Fatigue per Tournament {subtitle}")
            fig_f.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=260, showlegend=False)
            st.plotly_chart(fig_f, width='stretch')

        # Current form radar (last 10 matches)
        with r1_col2:
            ACE_BENCHMARK = 1.2
            cats = ['1st Serve Won', 'Break Point Save Rate', 'Ace per Game', 'Win Rate']

            def radar_values(s):
                ace = s['avg_ace_per_game_last_10']
                return [np.nan_to_num(s['avg_1st_srv_won_last_10']),
                        np.nan_to_num(s['avg_bp_save_rate_last_10']),
                        min(np.nan_to_num(ace) / ACE_BENCHMARK, 1.0),
                        np.nan_to_num(s['win_rate_last_10'])]

            fig_r = go.Figure()
            for name, s in [(p1, s1), (p2, s2)]:
                fig_r.add_trace(go.Scatterpolar(r=radar_values(s), theta=cats, fill='toself',
                                                name=name, line_color=color_map[name]))
            fig_r.update_layout(title="Current Form<br><sup><i>(Last 10 matches)</i></sup>",
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                margin=dict(l=30, r=30, t=50, b=10), height=260, showlegend=False)
            st.plotly_chart(fig_r, width='stretch')

        if not h2h_p1.empty:
            p1_wins = int(h2h_p1['result'].sum())
            p2_wins = len(h2h_p1) - p1_wins
            with r2_col1:
                fig_pie = px.pie(values=[p1_wins, p2_wins], names=[p1, p2], color=[p1, p2],
                                 color_discrete_map=color_map, hole=0.3,
                                 title=f"H2H Wins (Total: {len(h2h_p1)} matches)")
                fig_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=240)
                st.plotly_chart(fig_pie, width='stretch')
            with r2_col2:
                st.markdown("**Historical Matches**")
                h2h_p1 = h2h_p1.sort_values(['tourney_date', 'match_num'], ascending=False)
                table = pd.DataFrame({
                    'Tourney': h2h_p1['tourney_name'] + " '" + h2h_p1['tourney_date'].astype(str).str[2:4],
                    'Score': h2h_p1['score'],
                    'Winner': np.where(h2h_p1['result'] == 1, p1, p2),
                })
                st.dataframe(table, width='stretch', hide_index=True, height=200)
        else:
            st.info(f"No previous Head-to-Head matches found between {p1} and {p2}.")