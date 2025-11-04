from pathlib import Path
import time
import pandas as pd

from nba_api.stats.endpoints import (
    leaguegamefinder,
    commonteamroster,
    leaguedashplayerstats,
)


TEAM_ID = 1610612742  # Dallas Mavericks
SEASONS = ["2024-25", "2025-26"]

OUTPUT_DIR = (Path(__file__).resolve().parents[2] / "data" / "original").resolve()
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
    print("Coletando dados do Dallas Mavericks – temporadas 2024-25 e 2025-26")

    all_games_df = []
    all_roster_df = []
    all_player_stats_pergame_df = []
    all_player_stats_totals_df = []

    for season in SEASONS:
        print(f"\n--- Coletando dados da temporada {season} ---")
        
        print("- Buscando lista de jogos...")
        games_df = fetch_games(TEAM_ID, season)
        all_games_df.append(games_df)
        print(f"  -> {len(games_df)} jogos encontrados para {season}")
        time.sleep(0.5)

        print("- Buscando elenco (roster)...")
        roster_df = fetch_roster(TEAM_ID, season)
        all_roster_df.append(roster_df)
        print(f"  -> {len(roster_df)} jogadores encontrados para {season}")
        time.sleep(0.5)

        print("- Buscando estatísticas de temporada por jogador (PerGame = médias)...")
        player_stats_pergame_df = fetch_player_season_stats(TEAM_ID, season, per_mode="PerGame")
        all_player_stats_pergame_df.append(player_stats_pergame_df)
        print(f"  -> {len(player_stats_pergame_df)} linhas encontradas para {season}")
        time.sleep(0.4)

        print("- Buscando estatísticas de temporada por jogador (Totals = acumulados)...")
        player_stats_totals_df = fetch_player_season_stats(TEAM_ID, season, per_mode="Totals")
        all_player_stats_totals_df.append(player_stats_totals_df)
        print(f"  -> {len(player_stats_totals_df)} linhas encontradas para {season}")
        time.sleep(0.4)

    print("\n--- Concatenando dados de todas as temporadas ---")
    
    combined_games_df = pd.concat(all_games_df, ignore_index=True)
    games_csv = OUTPUT_DIR / "dal_games_2024_25_26.csv"
    combined_games_df.to_csv(games_csv, index=False)
    print(f"  -> {len(combined_games_df)} jogos salvos em {games_csv}")

    combined_roster_df = pd.concat(all_roster_df, ignore_index=True)
    roster_csv = OUTPUT_DIR / "dal_roster_2024_25_26.csv"
    combined_roster_df.to_csv(roster_csv, index=False)
    print(f"  -> {len(combined_roster_df)} jogadores salvos em {roster_csv}")

    combined_player_stats_pergame_df = pd.concat(all_player_stats_pergame_df, ignore_index=True)
    player_stats_pergame_csv = OUTPUT_DIR / "dal_players_season_stats_media_2024_25_26.csv"
    combined_player_stats_pergame_df.to_csv(player_stats_pergame_csv, index=False)
    print(f"  -> {len(combined_player_stats_pergame_df)} linhas salvas em {player_stats_pergame_csv}")

    combined_player_stats_totals_df = pd.concat(all_player_stats_totals_df, ignore_index=True)
    player_stats_totals_csv = OUTPUT_DIR / "dal_players_season_stats_acumulado_2024_25_26.csv"
    combined_player_stats_totals_df.to_csv(player_stats_totals_csv, index=False)
    print(f"  -> {len(combined_player_stats_totals_df)} linhas salvas em {player_stats_totals_csv}")

    print("\nConcluído: dados de ambas as temporadas extraídos e concatenados na pasta data/.")


if __name__ == "__main__":
    main()
