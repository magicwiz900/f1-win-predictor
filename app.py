import streamlit as st
import pandas as pd
import requests
import time

from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="F1 Race Win Predictor", page_icon="🏎️", layout="centered")

BASE_URL = "https://api.jolpi.ca/ergast/f1"
HEADERS = {"User-Agent": "F1PredictionPortfolioProject/1.0"}
DATA_FILE = "f1_results.csv"


# ---------- Load the bundled historical dataset (no live API call needed) ----------
@st.cache_data(show_spinner="Loading race history...")
def load_local_data():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["position"])
    return df


# ---------- Optional: fetch the newest season's results live ----------
def fetch_season_results(year):
    """Fetch every race result for one F1 season, handling pagination."""
    results = []
    offset = 0
    limit = 100
    session = requests.Session()
    while True:
        url = f"{BASE_URL}/{year}/results.json?limit={limit}&offset={offset}"
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()["MRData"]
        races = data["RaceTable"]["Races"]
        for race in races:
            for res in race["Results"]:
                results.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "date": race["date"],
                    "driver_id": res["Driver"]["driverId"],
                    "driver_code": res["Driver"].get("code", ""),
                    "constructor_id": res["Constructor"]["constructorId"],
                    "grid": int(res["grid"]),
                    "position": int(res["position"]) if res["position"].isdigit() else None,
                    "points": float(res["points"]),
                    "status": res["status"],
                })
        total = int(data["total"])
        offset += limit
        if offset >= total:
            break
        time.sleep(0.3)
    return results


@st.cache_data(show_spinner="Building features...")
def build_features(df):
    df["win"] = (df["position"] == 1).astype(int)

    df = df.sort_values(["driver_id", "season", "round"])
    df["recent_form"] = (
        df.groupby("driver_id")["position"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    df = df.sort_values(["constructor_id", "season", "round"])
    df["constructor_form"] = (
        df.groupby("constructor_id")["points"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["recent_form"] = df["recent_form"].fillna(df["recent_form"].median())
    df["constructor_form"] = df["constructor_form"].fillna(df["constructor_form"].median())
    return df


@st.cache_resource(show_spinner="Training the model...")
def train_model(df, train_end_year):
    features = ["grid", "recent_form", "constructor_form"]
    train = df[df["season"] < train_end_year]
    X_train, y_train = train[features], train["win"]
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    return model, features


# ---------- Load base data ----------
if "df" not in st.session_state:
    st.session_state.df = load_local_data()

# ---------- Sidebar: optional live refresh ----------
with st.sidebar:
    st.markdown("### Data")
    max_season = int(st.session_state.df["season"].max())
    st.caption(f"Bundled data covers up to {max_season}.")
    if st.button("🔄 Try fetching a newer season"):
        try:
            new_rows = fetch_season_results(max_season + 1)
            if new_rows:
                new_df = pd.DataFrame(new_rows).dropna(subset=["position"])
                st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
                st.success(f"Added {max_season + 1} season data.")
            else:
                st.info(f"No results found yet for {max_season + 1}.")
        except Exception as e:
            st.warning(
                "Couldn't reach the live API right now (it's a small free service and "
                "sometimes rate-limits or is briefly unavailable). Using the saved data instead."
            )

df = build_features(st.session_state.df.copy())
model, features = train_model(df, int(df["season"].max()))

# ---------- UI ----------
st.title("🏎️ F1 Race Win Predictor")
st.write(
    "A Random Forest model trained on grid position and recent form. "
    "Pick a race below to see predicted win probabilities."
)

seasons = sorted(df["season"].unique(), reverse=True)
season = st.selectbox("Season", seasons)

season_df = df[df["season"] == season]
races = season_df[["round", "race_name"]].drop_duplicates().sort_values("round")
race_options = [f"{row.race_name} (Round {row.round})" for row in races.itertuples()]
race_label = st.selectbox("Race", race_options)
selected_round = int(race_label.split("Round ")[1].replace(")", ""))

race_data = season_df[season_df["round"] == selected_round].copy()
race_data["predicted_win_prob"] = model.predict_proba(race_data[features])[:, 1]
race_data = race_data.sort_values("predicted_win_prob", ascending=False)

st.subheader(f"Predicted win probabilities — {race_data['race_name'].iloc[0]}")
st.bar_chart(race_data.set_index("driver_id")["predicted_win_prob"])

display = race_data[["driver_id", "grid", "predicted_win_prob", "win"]].rename(
    columns={"driver_id": "Driver", "grid": "Grid", "predicted_win_prob": "Predicted win %", "win": "Actually won"}
)
display["Predicted win %"] = (display["Predicted win %"] * 100).round(1)
display["Actually won"] = display["Actually won"].map({1: "🏆", 0: ""})
st.dataframe(display, hide_index=True, use_container_width=True)

winner_row = race_data[race_data["win"] == 1]
if not winner_row.empty:
    st.caption(f"Actual race winner: **{winner_row['driver_id'].iloc[0]}**")
