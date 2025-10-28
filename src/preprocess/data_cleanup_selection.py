import pandas as pd
import re
from pathlib import Path

# === Caminhos principais ===
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# === Arquivos de entrada ===
GAMES_CSV = INTERIM_DIR / "dal-games-traduzido.csv"
ROSTER_CSV = INTERIM_DIR / "dal-roster-traduzido.csv"
PLAYERS_MEDIA_CSV = INTERIM_DIR / "dal-players-media-traduzido.csv"
PLAYERS_TOTAL_CSV = INTERIM_DIR / "dal-players-total-traduzido.csv"

print("Iniciando limpeza e integração dos dados do Dallas Mavericks 2024–25\n")

# === 1 - Jogos ===
print("\nProcessando jogos...")
games_df = pd.read_csv(GAMES_CSV)

# === Ajustes e transformações iniciais ===

# Converter "confronto" em mando de quadra (1 = casa, 0 = fora)
games_df["mando-de-jogo"] = games_df["confronto"].apply(lambda x: 1 if "vs." in str(x) else 0)

# Converter resultado em numérico (W=1, L=0)
games_df["resultado"] = games_df["resultado"].map({"W": 1, "L": 0, 1: 1, 0: 0})

# Remover hífens das datas
games_df["data-jogo"] = games_df["data-jogo"].astype(str).str.replace("-", "")

# Substituir valores ausentes por 0
games_df.fillna(0, inplace=True)

# === Selecionar colunas finais (mantendo consistência) ===
colunas_games = [
    "id-jogo", "data-jogo", "resultado", "mando-de-jogo",
    "pontos", "saldo-pontos", "arremessos-convertidos", "arremessos-tentados",
    "porcentagem-arremessos", "triplos-convertidos", "triplos-tentados", "porcentagem-triplos",
    "lances-livres-convertidos", "lances-livres-tentados", "porcentagem-lances-livres",
    "rebotes-ofensivos", "rebotes-defensivos", "rebotes-totais",
    "assistencias", "roubos", "tocos", "erros", "faltas"
]
colunas_games = [c for c in colunas_games if c in games_df.columns]

games_clean = games_df[colunas_games].copy()

# === Salvar ===
games_clean.to_csv(PROCESSED_DIR / "dallas_games_2024-25.csv", index=False)
print(f"Jogos limpos: {len(games_clean)} linhas")


# === 2 - Roster ===
print("Processando roster...")
roster_df = pd.read_csv(ROSTER_CSV)

# Funções auxiliares
def convert_height_to_cm(height_str):
    """Converte altura de formato '6-7' (pés-polegadas) para cm."""
    match = re.match(r"(\d+)-(\d+)", str(height_str))
    if match:
        feet, inches = int(match.group(1)), int(match.group(2))
        return round(feet * 30.48 + inches * 2.54, 1)
    return None

def convert_weight_to_kg(weight_str):
    """Converte peso de libras para kg."""
    try:
        return round(float(weight_str) * 0.453592, 1)
    except:
        return None

# Converter unidades
roster_df["altura-cm"] = roster_df["altura"].apply(convert_height_to_cm)
roster_df["peso-kg"] = roster_df["peso"].apply(convert_weight_to_kg)

# Mapeamento de posições
position_map = {"G": 1, "F": 2, "F-C": 3, "C-F": 4, "C": 5}
roster_df["posicao-g-f-fc-cf-c"] = roster_df["posicao"].map(position_map).fillna(0)

# Selecionar colunas relevantes
colunas_roster = [
    "id-jogador", "sigla-time", "posicao-g-f-fc-cf-c", "altura-cm", "peso-kg", "idade"
]
colunas_roster = [c for c in colunas_roster if c in roster_df.columns]

roster_clean = roster_df[colunas_roster].copy()
roster_clean.fillna("NA", inplace=True)
roster_clean.to_csv(INTERIM_DIR / "dal-roster-clean.csv", index=False)
print(f"Roster limpo: {len(roster_clean)} linhas")


# === 3 - Estatísticas dos Jogadores - Médias ===
print("\nProcessando estatísticas (médias)...")
players_media_df = pd.read_csv(PLAYERS_MEDIA_CSV)

colunas_players = [
    "id-jogador", "jogos-disputados", "minutos", "arremessos-convertidos", "arremessos-tentados",
    "porcentagem-arremessos", "triplos-convertidos", "triplos-tentados",
    "porcentagem-triplos", "lances-livres-convertidos", "lances-livres-tentados",
    "porcentagem-lances-livres", "rebotes-ofensivos", "rebotes-defensivos",
    "rebotes-totais", "assistencias", "erros", "roubos", "tocos",
    "faltas", "pontos"
]
colunas_players = [c for c in colunas_players if c in players_media_df.columns]

players_media_clean = players_media_df[colunas_players].copy()
players_media_clean.fillna(0, inplace=True)
players_media_clean.to_csv(INTERIM_DIR / "dal-players-media-clean.csv", index=False)
print(f"Estatísticas médias limpas: {len(players_media_clean)} linhas")


# === 4 - Estatísticas dos Jogadores - Acumuladas ===
print("\nProcessando estatísticas (totais)...")
players_total_df = pd.read_csv(PLAYERS_TOTAL_CSV)

colunas_total = [c for c in colunas_players if c in players_total_df.columns]
players_total_clean = players_total_df[colunas_total].copy()
players_total_clean.fillna(0, inplace=True)
players_total_clean.to_csv(INTERIM_DIR / "dal-players-total-clean.csv", index=False)
print(f"Estatísticas totais limpas: {len(players_total_clean)} linhas")


# === 5 - Merge das estatísticas dos Jogadores (Roster + Médias) ===
print("\nCriando merge  das estatísticas dos jogadores (roster + médias)...")

if "id-jogador" in roster_clean.columns and "id-jogador" in players_media_clean.columns:
    merged = pd.merge(roster_clean, players_media_clean, on="id-jogador", how="left")
    merged.fillna(0, inplace=True)

    # Eficiência simples: pontos por tentativa de arremesso
    merged["eficiencia-simples"] = (
        merged["pontos"] / merged["arremessos-tentados"]
    ).replace([float("inf"), -float("inf")], 0)

    # Já está "processado" e finalizado:
    merged.to_csv(PROCESSED_DIR / "dallas_players_2024-25.csv", index=False)
    print(f"Merge salvo: dal-players-merged.csv ({len(merged)} linhas)")
else:
    print("Coluna 'id-jogador' ausente em um dos arquivos. Merge ignorado.")

print("\nProcessamento completo! Todos os arquivos estão em data/processed/")
