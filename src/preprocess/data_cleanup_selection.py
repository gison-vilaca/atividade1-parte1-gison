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
PLAYERS_TOTAL_CSV = INTERIM_DIR / "dal-players-total-traduzido.csv"

print("Iniciando limpeza e integração dos dados do Dallas Mavericks 2024–26\n")

# === 1 - Jogos ===
print("\nProcessando jogos...")
games_df = pd.read_csv(GAMES_CSV)

# Converter "confronto" em mando de quadra (1 = casa, 0 = fora)
games_df["mando-de-jogo"] = games_df["confronto"].apply(lambda x: 1 if "vs." in str(x) else 0)

# Converter resultado em numérico (W=1, L=0)
games_df["resultado"] = games_df["resultado"].map({"W": 1, "L": 0, 1: 1, 0: 0})

# Remover hífens das datas
games_df["data-jogo"] = games_df["data-jogo"].astype(str).str.replace("-", "")

# Substituir valores ausentes por 0
games_df.fillna(0, inplace=True)

# Selecionar colunas finais
colunas_games = [
    "data-jogo", "resultado", "mando-de-jogo",
    "pontos", "saldo-pontos", "arremessos-convertidos", "arremessos-tentados",
    "porcentagem-arremessos", "triplos-convertidos", "triplos-tentados", "porcentagem-triplos",
    "lances-livres-convertidos", "lances-livres-tentados", "porcentagem-lances-livres",
    "rebotes-ofensivos", "rebotes-defensivos", "rebotes-totais",
    "assistencias", "roubos", "tocos", "erros", "faltas"
]
colunas_games = [c for c in colunas_games if c in games_df.columns]
games_clean = games_df[colunas_games].copy()

games_clean.to_csv(PROCESSED_DIR / "dallas_games_2024-26.csv", index=False)
print(f"Jogos limpos: {len(games_clean)} linhas")


# === 2 - Roster ===
print("\nProcessando roster...")
roster_df = pd.read_csv(ROSTER_CSV)

# Funções auxiliares
def convert_height_to_cm(height_str):
    match = re.match(r"(\d+)-(\d+)", str(height_str))
    if match:
        feet, inches = int(match.group(1)), int(match.group(2))
        return round(feet * 30.48 + inches * 2.54, 1)
    return None

def convert_weight_to_kg(weight_str):
    try:
        return round(float(weight_str) * 0.453592, 1)
    except:
        return None

# Converter unidades
roster_df["altura-cm"] = roster_df["altura"].apply(convert_height_to_cm)
roster_df["peso-kg"] = roster_df["peso"].apply(convert_weight_to_kg)

# Mapear posições (categorias → numérico)
position_map = {"G": 1, "F": 2, "F-C": 3, "C-F": 4, "C": 5}
roster_df["posicao-g-f-fc-cf-c"] = roster_df["posicao"].map(position_map).fillna(0)

colunas_roster = ["nome-jogador", "posicao-g-f-fc-cf-c", "altura-cm", "peso-kg", "idade"]
colunas_roster = [c for c in colunas_roster if c in roster_df.columns]
roster_clean = roster_df[colunas_roster].copy()
roster_clean.fillna("NA", inplace=True)
roster_clean.to_csv(INTERIM_DIR / "dal-roster-clean.csv", index=False)
print(f"Roster limpo: {len(roster_clean)} linhas")


# === 3 - Estatísticas Totais (somando 24–25 e 25–26) ===
print("\nProcessando e somando estatísticas totais por jogador...")
players_total_df = pd.read_csv(PLAYERS_TOTAL_CSV)

# Colunas de soma direta
colunas_somar = [
    "jogos-disputados", "minutos", "arremessos-convertidos", "arremessos-tentados",
    "triplos-convertidos", "triplos-tentados",
    "lances-livres-convertidos", "lances-livres-tentados",
    "rebotes-ofensivos", "rebotes-defensivos", "rebotes-totais",
    "assistencias", "erros", "roubos", "tocos", "faltas", "pontos"
]
colunas_somar = [c for c in colunas_somar if c in players_total_df.columns]

# Somar por jogador
players_total_sum = (
    players_total_df.groupby("nome-jogador")[colunas_somar]
    .sum()
    .reset_index()
)

# === Recalcular porcentagens ===
players_total_sum["porcentagem-arremessos"] = (
    players_total_sum["arremessos-convertidos"] / players_total_sum["arremessos-tentados"]
).fillna(0)

players_total_sum["porcentagem-triplos"] = (
    players_total_sum["triplos-convertidos"] / players_total_sum["triplos-tentados"]
).fillna(0)

players_total_sum["porcentagem-lances-livres"] = (
    players_total_sum["lances-livres-convertidos"] / players_total_sum["lances-livres-tentados"]
).fillna(0)

players_total_sum.fillna(0, inplace=True)
players_total_sum.to_csv(INTERIM_DIR / "dal-players-total-somado.csv", index=False)
print(f"Totais somados por jogador: {len(players_total_sum)} linhas")


# === 4 - Merge Final (Roster + Totais) ===
print("\nMesclando roster com estatísticas totais recalculadas...")

roster_clean = roster_clean.drop_duplicates(subset=["nome-jogador"], keep="first")

merged = pd.merge(roster_clean, players_total_sum, on="nome-jogador", how="left")
merged.fillna(0, inplace=True)

# Eficiência simples: pontos por tentativa de arremesso
merged["eficiencia-simples"] = (
    merged["pontos"] / merged["arremessos-tentados"]
).replace([float("inf"), -float("inf")], 0)

# Converter porcentagens de fração para percentual
merged["porcentagem-arremessos"] *= 100
merged["porcentagem-triplos"] *= 100
merged["porcentagem-lances-livres"] *= 100

merged.to_csv(PROCESSED_DIR / "dallas_players_2024-26.csv", index=False)
print(f"Merge final salvo: dallas_players_2024-26.csv ({len(merged)} linhas)")

print("\n✅ Processamento completo! Arquivos finais estão em data/processed/")
