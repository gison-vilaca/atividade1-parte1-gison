from pathlib import Path
import time
import pandas as pd

from nba_api.stats.endpoints import (
    leaguegamefinder,
    commonteamroster,
    leaguedashplayerstats,
)


TEAM_ID = 1610612742  # Dallas Mavericks
SEASON = "2024-25"

OUTPUT_DIR = (Path(__file__).resolve().parents[1] / "data").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_games(team_id: int, season: str) -> pd.DataFrame:
    gf = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
    )
    df = gf.get_data_frames()[0]
    if "GAME_DATE" in df.columns:
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


def fetch_roster(team_id: int, season: str) -> pd.DataFrame:
    season_year = season.split("-")[0]
    cr = commonteamroster.CommonTeamRoster(team_id=team_id, season=season_year)
    df = cr.get_data_frames()[0]  # Roster
    return df


def fetch_player_season_stats(team_id: int, season: str, per_mode: str = "PerGame") -> pd.DataFrame:

    lps = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        team_id_nullable=team_id,
        per_mode_detailed=per_mode,
    )
    df = lps.get_data_frames()[0]
    try:
        if "TEAM_ID" in df.columns:
            df = df[df["TEAM_ID"].astype("Int64") == int(team_id)]
        elif "TEAM_ABBREVIATION" in df.columns:
            df = df[df["TEAM_ABBREVIATION"].astype(str).str.upper() == "DAL"]
    except Exception:
        pass
    return df


def main() -> None:
    print("Coletando dados do Dallas Mavericks – temporada 2024-25")

    print("- Buscando lista de jogos...")
    games_df = fetch_games(TEAM_ID, SEASON)
    games_csv = OUTPUT_DIR / "dal_games_2024_25.csv"
    games_df.to_csv(games_csv, index=False)
    print(f"  -> {len(games_df)} jogos salvos em {games_csv}")
    time.sleep(0.5)

    print("- Buscando elenco (roster)...")
    roster_df = fetch_roster(TEAM_ID, SEASON)
    roster_csv = OUTPUT_DIR / "dal_roster_2024_25.csv"
    roster_df.to_csv(roster_csv, index=False)
    print(f"  -> {len(roster_df)} jogadores salvos em {roster_csv}")
    time.sleep(0.5)

    print("- Buscando estatísticas de temporada por jogador (PerGame = médias)...")
    player_stats_pergame_df = fetch_player_season_stats(TEAM_ID, SEASON, per_mode="PerGame")
    player_stats_pergame_csv = OUTPUT_DIR / "dal_players_season_stats_media_2024_25.csv"
    player_stats_pergame_df.to_csv(player_stats_pergame_csv, index=False)
    print(f"  -> {len(player_stats_pergame_df)} linhas salvas em {player_stats_pergame_csv}")
    time.sleep(0.4)

    print("- Buscando estatísticas de temporada por jogador (Totals = acumulados)...")
    player_stats_totals_df = fetch_player_season_stats(TEAM_ID, SEASON, per_mode="Totals")
    player_stats_totals_csv = OUTPUT_DIR / "dal_players_season_stats_acumulado_2024_25.csv"
    player_stats_totals_df.to_csv(player_stats_totals_csv, index=False)
    print(f"  -> {len(player_stats_totals_df)} linhas salvas em {player_stats_totals_csv}")

    print("\nConcluído: dados extraídos na pasta data/.")


if __name__ == "__main__":
    main()
