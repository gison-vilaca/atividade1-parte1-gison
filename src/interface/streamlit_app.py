import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="Dallas Mavericks 2024-25 - Análise Exploratória",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        border: 1px solid #d1ecf1;
    }
    .stMetric {
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        border: 1px solid #gray !important;
        margin: 0.25rem !important;
    }
    .stMetric > div {
        background-color: #gray !important;
    }
    .stMetric [data-testid="metric-container"] {
        background-color: #e8f4fd !important;
        border: 1px solid #bee5eb !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        ROOT_DIR = Path(__file__).resolve().parents[2]
        processed_dir = ROOT_DIR / "data" / "processed"
        original_dir = ROOT_DIR / "data" / "original"
        
        players_df = pd.read_csv(processed_dir / "dallas_players_2024-25.csv")
        games_df = pd.read_csv(processed_dir / "dallas_games_2024-25.csv")
        
        original_players_df = pd.read_csv(original_dir / "dal_players_season_stats_media_2024_25.csv")
        
        players_df = add_player_names(players_df, original_players_df)
        
        games_df['data-jogo'] = pd.to_datetime(games_df['data-jogo'], format='%Y%m%d')
        
        return players_df, games_df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

def add_player_names(processed_df, original_df):
    """Adiciona os nomes dos jogadores aos dados processados"""
    try:
        processed_with_names = processed_df.copy()
        processed_with_names['nome-jogador'] = ""
        
        for idx, row in processed_df.iterrows():
            matching_player = original_df[
                (abs(original_df['AGE'] - row['idade']) <= 1) &
                (abs(original_df['GP'] - row['jogos-disputados']) <= 2) &
                (abs(original_df['MIN'] - row['minutos']) <= 2.0) &
                (abs(original_df['PTS'] - row['pontos']) <= 1.0)
            ]
            
            if len(matching_player) == 1:
                processed_with_names.at[idx, 'nome-jogador'] = matching_player.iloc[0]['PLAYER_NAME']
            elif len(matching_player) > 1:
                best_match = matching_player.iloc[0]
                processed_with_names.at[idx, 'nome-jogador'] = best_match['PLAYER_NAME']
            else:
                processed_with_names.at[idx, 'nome-jogador'] = f"Jogador #{idx+1}"
        
        return processed_with_names
    except Exception as e:
        st.warning(f"Não foi possível adicionar nomes dos jogadores: {e}")
        processed_df['nome-jogador'] = [f"Jogador #{i+1}" for i in range(len(processed_df))]
        return processed_df

def create_summary_metrics(players_df, games_df):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_players = len(players_df)
        st.metric("Total de Jogadores", total_players)
    
    with col2:
        total_games = len(games_df)
        st.metric("Jogos Disputados", total_games)
    
    with col3:
        wins = games_df['resultado'].sum()
        win_pct = (wins / total_games * 100) if total_games > 0 else 0
        st.metric("Vitórias", f"{wins} ({win_pct:.1f}%)")
    
    with col4:
        avg_points = games_df['pontos'].mean()
        st.metric("Média de Pontos", f"{avg_points:.1f}")
    
    with col5:
        avg_assists = games_df['assistencias'].mean()
        st.metric("Média de Assistências", f"{avg_assists:.1f}")

def player_analysis(players_df):
    st.header("📊 Análise dos Jogadores")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Pontuadores")
        top_scorers = players_df.nlargest(10, 'pontos')[['nome-jogador', 'posicao-g-f-fc-cf-c', 'pontos', 'jogos-disputados']]
        top_scorers['pontos_por_jogo'] = top_scorers['pontos'] / top_scorers['jogos-disputados']
        
        top_scorers_chart = top_scorers.reset_index()
        
        fig = px.bar(
            top_scorers_chart, 
            x='pontos', 
            y='nome-jogador',
            orientation='h',
            title="Pontos por Jogador",
            labels={'pontos': 'Pontos Totais', 'nome-jogador': 'Jogadores'},
            color='pontos',
            color_continuous_scale='blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Eficiência de Arremessos")
        efficiency_df = players_df[players_df['arremessos-tentados'] >= 5].copy()
        efficiency_df['eficiencia_arremesso'] = efficiency_df['porcentagem-arremessos'] * 100
        
        fig = px.scatter(
            efficiency_df,
            x='arremessos-tentados',
            y='eficiencia_arremesso',
            size='pontos',
            color='porcentagem-triplos',
            hover_name='nome-jogador' if 'nome-jogador' in efficiency_df.columns else None,
            title="Eficiência vs Volume de Arremessos",
            labels={
                'arremessos-tentados': 'Arremessos Tentados',
                'eficiencia_arremesso': 'Eficiência (%)',
                'porcentagem-triplos': '% Triplos'
            },
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("👥 Distribuição por Posição")
    col1, col2 = st.columns(2)
    
    with col1:
        pos_counts = players_df['posicao-g-f-fc-cf-c'].value_counts()
        position_names = {1: '1 - Guard', 2: '2 - Forward', 3: '3 - Center', 4: '4 - Forward-Center', 5: '5 - Center-Forward'}
        
        pos_labels = [position_names.get(pos, f"Posição {pos}") for pos in pos_counts.index]
        
        fig = px.pie(
            values=pos_counts.values,
            names=pos_labels,
            title="Distribuição de Jogadores por Posição",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        pos_stats = players_df.groupby('posicao-g-f-fc-cf-c').agg({
            'pontos': 'mean',
            'rebotes-totais': 'mean',
            'assistencias': 'mean',
            'porcentagem-arremessos': 'mean'
        }).round(2)
        
        st.write("**Médias por Posição:**")
        st.dataframe(pos_stats, use_container_width=True)

def game_analysis(games_df):
    """Análise detalhada dos jogos"""
    st.header("🏀 Análise dos Jogos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance ao Longo da Temporada")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=games_df['data-jogo'],
            y=games_df['pontos'],
            mode='lines+markers',
            name='Pontos',
            line=dict(color='blue', width=2),
            marker=dict(
                color=games_df['resultado'].map({1: 'green', 0: 'red'}),
                size=8,
                line=dict(color='white', width=1)
            )
        ))
        
        fig.update_layout(
            title="Pontos por Jogo (Verde=Vitória, Vermelho=Derrota)",
            xaxis_title="Data do Jogo",
            yaxis_title="Pontos",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏠 Performance Casa vs Fora")
        
        home_away = games_df.groupby('mando-de-jogo').agg({
            'pontos': 'mean',
            'porcentagem-arremessos': 'mean',
            'assistencias': 'mean',
            'resultado': 'mean'
        }).round(3)
        
        home_away.index = ['Fora de Casa', 'Em Casa']
        
        fig = go.Figure()
        
        metrics = ['pontos', 'porcentagem-arremessos', 'assistencias']
        metric_names = ['Pontos Médios', '% Arremessos', 'Assistências']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            fig.add_trace(go.Bar(
                name=name,
                x=['Fora de Casa', 'Em Casa'],
                y=home_away[metric],
                yaxis=f'y{i+1}' if i > 0 else 'y'
            ))
        
        fig.update_layout(
            title="Comparação Casa vs Fora",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔗 Correlações entre Estatísticas")
    
    numeric_cols = [
        'pontos', 'arremessos-convertidos', 'porcentagem-arremessos',
        'triplos-convertidos', 'porcentagem-triplos', 'rebotes-totais',
        'assistencias', 'roubos', 'tocos', 'resultado'
    ]
    
    correlation_matrix = games_df[numeric_cols].corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Matriz de Correlação - Estatísticas dos Jogos",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def advanced_analysis(players_df, games_df):
    """Análises mais avançadas"""
    st.header("🧠 Análise Avançada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Eficiência vs Uso")
        
        players_df['uso_estimado'] = (
            players_df['arremessos-tentados'] + 
            players_df['lances-livres-tentados'] * 0.44 + 
            players_df['erros']
        ) / players_df['minutos']
        
        players_df['eficiencia_verdadeira'] = (
            players_df['pontos'] / 
            (2 * (players_df['arremessos-tentados'] + 0.44 * players_df['lances-livres-tentados']))
        )
        
        regular_players = players_df[players_df['minutos'] >= 10]
        
        fig = px.scatter(
            regular_players,
            x='uso_estimado',
            y='eficiencia_verdadeira',
            size='minutos',
            color='pontos',
            hover_name='nome-jogador' if 'nome-jogador' in regular_players.columns else None,
            title="Eficiência Verdadeira vs Taxa de Uso",
            labels={
                'uso_estimado': 'Taxa de Uso Estimada',
                'eficiencia_verdadeira': 'Eficiência Verdadeira',
                'minutos': 'Minutos',
                'pontos': 'Pontos'
            },
            color_continuous_scale='plasma'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Análise de Clutch Time")
        
        games_df['jogo_apertado'] = abs(games_df['saldo-pontos']) <= 10
        clutch_performance = games_df.groupby('jogo_apertado').agg({
            'pontos': 'mean',
            'porcentagem-arremessos': 'mean',
            'erros': 'mean',
            'resultado': 'mean'
        }).round(3)
        
        clutch_performance.index = ['Jogos Folgados', 'Jogos Apertados']
        
        categories = ['Pontos Médios', '% Arremessos', 'Erros (inv)', '% Vitórias']
        
        fig = go.Figure()
        
        for idx, game_type in enumerate(clutch_performance.index):
            values = [
                clutch_performance.loc[game_type, 'pontos'],
                clutch_performance.loc[game_type, 'porcentagem-arremessos'] * 100,
                (1 - clutch_performance.loc[game_type, 'erros'] / 20) * 100, 
                clutch_performance.loc[game_type, 'resultado'] * 100
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=game_type
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 120]
                )
            ),
            showlegend=True,
            title="Performance: Jogos Apertados vs Folgados",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def interactive_analysis(players_df, games_df):
    """Análise interativa com filtros"""
    st.header("🔍 Análise Interativa")
    
    st.sidebar.header("🎛️ Filtros")
    
    min_minutes = st.sidebar.slider(
        "Minutos mínimos por jogo",
        min_value=0,
        max_value=int(players_df['minutos'].max()),
        value=10,
        step=5
    )
    
    positions = players_df['posicao-g-f-fc-cf-c'].unique()
    position_names = {1: 'G', 2: 'F', 3: 'C', 4: 'FC', 5: 'CF'}
    selected_positions = st.sidebar.multiselect(
        "Posições",
        options=positions,
        default=positions,
        format_func=lambda x: position_names.get(x, f"Posição {x}")
    )
    
    filtered_players = players_df[
        (players_df['minutos'] >= min_minutes) &
        (players_df['posicao-g-f-fc-cf-c'].isin(selected_positions))
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        stat_options = {
            'pontos': 'Pontos',
            'rebotes-totais': 'Rebotes',
            'assistencias': 'Assistências',
            'porcentagem-arremessos': '% Arremessos',
            'porcentagem-triplos': '% Triplos',
            'roubos': 'Roubos',
            'tocos': 'Tocos'
        }
        
        x_stat = st.selectbox("Estatística X", options=list(stat_options.keys()), 
                             format_func=lambda x: stat_options[x])
    
    with col2:
        y_stat = st.selectbox("Estatística Y", options=list(stat_options.keys()), 
                             format_func=lambda x: stat_options[x], index=1)
    
    if not filtered_players.empty:
        fig = px.scatter(
            filtered_players,
            x=x_stat,
            y=y_stat,
            size='minutos',
            color='posicao-g-f-fc-cf-c',
            hover_name='nome-jogador' if 'nome-jogador' in filtered_players.columns else None,
            title=f"{stat_options[y_stat]} vs {stat_options[x_stat]}",
            labels={x_stat: stat_options[x_stat], y_stat: stat_options[y_stat]},
            color_discrete_map={1: 'blue', 2: 'green', 3: 'red', 4: 'purple', 5: 'orange'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Jogadores Selecionados")
        display_cols = ['nome-jogador', 'posicao-g-f-fc-cf-c', 'idade', 'jogos-disputados', 'minutos', 
                       'pontos', 'rebotes-totais', 'assistencias', 'porcentagem-arremessos']
        
        available_cols = [col for col in display_cols if col in filtered_players.columns]
        st.dataframe(filtered_players[available_cols].round(2), use_container_width=True)
    else:
        st.warning("Nenhum jogador atende aos critérios selecionados.")

def regression_analysis(players_df, games_df):
    """Análise de Regressão Linear e Logística"""
    st.header("📈 Análise Preditiva - Regressão Linear e Logística")
    
    if not SKLEARN_AVAILABLE:
        st.error("⚠️ Scikit-learn não está instalado. Instale com: pip install scikit-learn")
        return
    
    analysis_type = st.selectbox(
        "Tipo de Análise",
        ["Regressão Linear", "Regressão Logística"],
        help="Escolha o tipo de análise preditiva"
    )
    
    st.markdown("---")
    
    if analysis_type == "Regressão Linear":
        linear_regression_analysis(players_df, games_df)
    else:
        logistic_regression_analysis(players_df, games_df)

def linear_regression_analysis(players_df, games_df):
    """Análise de Regressão Linear"""
    st.subheader("🔢 Regressão Linear")
    st.markdown("**Predição de valores numéricos (pontos, rebotes, assistências)**")
    
    df = players_df.copy()
    
    df = df.dropna()
    
    if len(df) < 10:
        st.warning("Dados insuficientes para análise de regressão.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Configuração da Análise")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_options = {
            'pontos': 'Pontos do Jogador',
            'rebotes-totais': 'Rebotes do Jogador', 
            'assistencias': 'Assistências do Jogador',
            'porcentagem-arremessos': 'Porcentagem de Arremessos',
            'minutos': 'Minutos Jogados'
        }
        
        target_var = st.selectbox(
            "Variável Dependente (Y) - O que queremos prever:",
            options=[col for col in target_options.keys() if col in numeric_cols],
            format_func=lambda x: target_options.get(x, x),
            help="Esta é a variável que queremos prever"
        )
        
        feature_options = {
            'jogos-disputados': 'Jogos Disputados',
            'minutos': 'Minutos por Jogo',
            'arremessos-tentados': 'Arremessos Tentados',
            'arremessos-convertidos': 'Arremessos Convertidos',
            'porcentagem-arremessos': 'Porcentagem de Arremessos',
            'triplos-tentados': 'Triplos Tentados',
            'triplos-convertidos': 'Triplos Convertidos',
            'porcentagem-triplos': 'Porcentagem de Triplos',
            'lances-livres-tentados': 'Lances Livres Tentados',
            'lances-livres-convertidos': 'Lances Livres Convertidos',
            'rebotes-ofensivos': 'Rebotes Ofensivos',
            'rebotes-defensivos': 'Rebotes Defensivos',
            'idade': 'Idade',
            'altura-cm': 'Altura (cm)',
            'peso-kg': 'Peso (kg)'
        }
        
        available_features = [col for col in feature_options.keys() if col in numeric_cols and col != target_var]
        
        selected_features = st.multiselect(
            "Variáveis Independentes (X) - Fatores que influenciam:",
            options=available_features,
            default=available_features[:3] if len(available_features) >= 3 else available_features,
            format_func=lambda x: feature_options.get(x, x),
            help="Estas são as variáveis que podem influenciar nossa predição"
        )
        
        test_size = st.slider(
            "Porcentagem para teste (%)",
            min_value=10,
            max_value=40,
            value=20,
            help="Porcentagem dos dados reservada para testar o modelo"
        )
    
    with col2:
        st.subheader("📊 Informações dos Dados")
        
        if target_var and selected_features:
            X = df[selected_features]
            y = df[target_var]
            
            st.write("**Estatísticas da Variável Dependente:**")
            stats_df = pd.DataFrame({
                'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo'],
                'Valor': [y.mean(), y.median(), y.std(), y.min(), y.max()]
            })
            st.dataframe(stats_df.round(2), use_container_width=True)
            
            st.write(f"**Tamanho do dataset:** {len(df)} registros")
            st.write(f"**Variáveis independentes:** {len(selected_features)}")
    
    if target_var and selected_features and len(selected_features) > 0:
        st.markdown("---")
        
        X = df[selected_features]
        y = df[target_var]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Métricas do Modelo")
            st.metric("R² (Treino)", f"{train_r2:.3f}")
            st.metric("R² (Teste)", f"{test_r2:.3f}")
            st.metric("RMSE (Treino)", f"{train_rmse:.2f}")
            st.metric("RMSE (Teste)", f"{test_rmse:.2f}")
        
        with col2:
            st.subheader("⚙️ Coeficientes")
            coef_df = pd.DataFrame({
                'Variável': selected_features,
                'Coeficiente': model.coef_,
                'Impacto': ['Alto' if abs(c) > np.std(model.coef_) else 'Baixo' for c in model.coef_]
            })
            st.dataframe(coef_df.round(4), use_container_width=True)
            
            st.write(f"**Intercepto (β₀):** {model.intercept_:.4f}")
        
        with col3:
            st.subheader("🔮 Fazer Predição")
            st.write("Insira valores para fazer uma predição:")
            
            prediction_values = {}
            for feature in selected_features:
                mean_val = X[feature].mean()
                min_val = float(X[feature].min())
                max_val = float(X[feature].max())
                
                prediction_values[feature] = st.number_input(
                    feature_options.get(feature, feature),
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"pred_{feature}"
                )
            
            if st.button("🎯 Fazer Predição"):
                pred_input = np.array([list(prediction_values.values())])
                prediction = model.predict(pred_input)[0]
                st.success(f"**Predição:** {prediction:.2f}")
        
        st.markdown("---")
        st.subheader("📊 Visualizações")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Dispersão com Regressão", 
            "Predição vs Realidade", 
            "Resíduos", 
            "Importância das Variáveis"
        ])
        
        with tab1:
            # Diagrama de dispersão com linha de regressão (usando primeira variável)
            if len(selected_features) > 0:
                first_feature = selected_features[0]
                
                fig = px.scatter(
                    df, 
                    x=first_feature, 
                    y=target_var,
                    title=f"{target_options[target_var]} vs {feature_options.get(first_feature, first_feature)}",
                    trendline="ols",
                    trendline_color_override="red"
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Gráfico Predição vs Realidade
            pred_real_df = pd.DataFrame({
                'Real': np.concatenate([y_train, y_test]),
                'Predito': np.concatenate([y_pred_train, y_pred_test]),
                'Tipo': ['Treino'] * len(y_train) + ['Teste'] * len(y_test)
            })
            
            fig = px.scatter(
                pred_real_df,
                x='Real',
                y='Predito',
                color='Tipo',
                title="Valores Preditos vs Valores Reais",
                color_discrete_map={'Treino': 'blue', 'Teste': 'red'}
            )
            
            # Linha diagonal perfeita
            min_val = min(pred_real_df['Real'].min(), pred_real_df['Predito'].min())
            max_val = max(pred_real_df['Real'].max(), pred_real_df['Predito'].max())
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Predição Perfeita',
                line=dict(dash='dash', color='green')
            ))
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Análise de resíduos
            residuals_train = y_train - y_pred_train
            residuals_test = y_test - y_pred_test
            
            residuals_df = pd.DataFrame({
                'Predito': np.concatenate([y_pred_train, y_pred_test]),
                'Resíduo': np.concatenate([residuals_train, residuals_test]),
                'Tipo': ['Treino'] * len(y_train) + ['Teste'] * len(y_test)
            })
            
            fig = px.scatter(
                residuals_df,
                x='Predito',
                y='Resíduo',
                color='Tipo',
                title="Análise de Resíduos",
                color_discrete_map={'Treino': 'blue', 'Teste': 'red'}
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="green")
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Importância das variáveis (baseada nos coeficientes absolutos)
            importance_df = pd.DataFrame({
                'Variável': [feature_options.get(f, f) for f in selected_features],
                'Importância': np.abs(model.coef_)
            }).sort_values('Importância', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importância',
                y='Variável',
                orientation='h',
                title="Importância das Variáveis (Valor Absoluto dos Coeficientes)",
                color='Importância',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def logistic_regression_analysis(players_df, games_df):
    """Análise de Regressão Logística"""
    st.subheader("🎲 Regressão Logística")
    st.markdown("**Predição de categorias (Será que o jogador fará mais de X pontos?)**")
    
    df = games_df.copy()
    df = df.dropna()
    
    if len(df) < 10:
        st.warning("Dados insuficientes para análise de regressão logística.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Configuração da Análise")
        
        classification_options = {
            'vitoria': 'O time vencerá o jogo?',
            'pontos_altos': 'O time fará mais de 110 pontos?',
            'assistencias_altas': 'O time fará mais de 25 assistências?',
            'arremesso_eficiente': 'O time terá mais de 45% nos arremessos?'
        }
        
        target_type = st.selectbox(
            "Tipo de Predição:",
            options=list(classification_options.keys()),
            format_func=lambda x: classification_options[x]
        )
        
        if target_type == 'vitoria':
            df['target'] = df['resultado']
        elif target_type == 'pontos_altos':
            threshold = st.slider("Limite de pontos:", 100, 130, 110)
            df['target'] = (df['pontos'] > threshold).astype(int)
        elif target_type == 'assistencias_altas':
            threshold = st.slider("Limite de assistências:", 15, 35, 25)
            df['target'] = (df['assistencias'] > threshold).astype(int)
        elif target_type == 'arremesso_eficiente':
            threshold = st.slider("Limite de eficiência (%):", 35, 55, 45)
            df['target'] = (df['porcentagem-arremessos'] > threshold/100).astype(int)
        
        feature_options = {
            'arremessos-tentados': 'Arremessos Tentados',
            'arremessos-convertidos': 'Arremessos Convertidos',
            'triplos-tentados': 'Triplos Tentados',
            'triplos-convertidos': 'Triplos Convertidos',
            'lances-livres-tentados': 'Lances Livres Tentados',
            'rebotes-totais': 'Rebotes Totais',
            'assistencias': 'Assistências',
            'roubos': 'Roubos',
            'tocos': 'Tocos',
            'mando-de-jogo': 'Mando de Jogo'
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [col for col in feature_options.keys() if col in numeric_cols]
        
        selected_features = st.multiselect(
            "Variáveis Independentes (X):",
            options=available_features,
            default=available_features[:4] if len(available_features) >= 4 else available_features,
            format_func=lambda x: feature_options.get(x, x)
        )
        
        test_size = st.slider(
            "Porcentagem para teste (%):",
            min_value=10,
            max_value=40,
            value=20
        )
    
    with col2:
        st.subheader("📊 Distribuição da Variável Target")
        
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            
            fig = px.pie(
                values=target_counts.values,
                names=['Não', 'Sim'],
                title=f"Distribuição: {classification_options[target_type]}",
                color_discrete_map={0: 'lightcoral', 1: 'lightblue'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Total de registros:** {len(df)}")
            st.write(f"**Classe positiva:** {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
            st.write(f"**Classe negativa:** {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    
    if 'target' in df.columns and selected_features and len(selected_features) > 0:
        st.markdown("---")
        
        X = df[selected_features]
        y = df['target']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size/100, random_state=42
        )
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Métricas do Modelo")
            st.metric("Acurácia (Treino)", f"{train_accuracy:.3f}")
            st.metric("Acurácia (Teste)", f"{test_accuracy:.3f}")
            
            cm = confusion_matrix(y_test, y_pred_test)
            st.write("**Matriz de Confusão:**")
            st.write(pd.DataFrame(cm, 
                                index=['Real: Não', 'Real: Sim'],
                                columns=['Pred: Não', 'Pred: Sim']))
        
        with col2:
            st.subheader("⚙️ Coeficientes")
            coef_df = pd.DataFrame({
                'Variável': [feature_options.get(f, f) for f in selected_features],
                'Coeficiente': model.coef_[0],
                'Odds Ratio': np.exp(model.coef_[0])
            })
            st.dataframe(coef_df.round(4), use_container_width=True)
            
            st.write(f"**Intercepto:** {model.intercept_[0]:.4f}")
        
        with col3:
            st.subheader("🔮 Fazer Predição")
            st.write("Insira valores para fazer uma predição:")
            
            prediction_values = {}
            original_features = df[selected_features]
            
            for i, feature in enumerate(selected_features):
                mean_val = original_features[feature].mean()
                min_val = float(original_features[feature].min())
                max_val = float(original_features[feature].max())
                
                prediction_values[feature] = st.number_input(
                    feature_options.get(feature, feature),
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"log_pred_{feature}"
                )
            
            if st.button("🎯 Fazer Predição", key="logistic_predict"):
                pred_input = np.array([list(prediction_values.values())])
                pred_input_scaled = scaler.transform(pred_input)
                
                prediction = model.predict(pred_input_scaled)[0]
                probability = model.predict_proba(pred_input_scaled)[0, 1]
                
                result = "SIM" if prediction == 1 else "NÃO"
                st.success(f"**Predição:** {result}")
                st.info(f"**Probabilidade:** {probability:.1%}")
        
        st.markdown("---")
        st.subheader("📊 Visualizações")
        
        tab1, tab2, tab3 = st.tabs([
            "Matriz de Confusão", 
            "Probabilidades", 
            "Importância das Variáveis"
        ])
        
        with tab1:
            cm = confusion_matrix(y_test, y_pred_test)
            
            fig = px.imshow(
                cm,
                title="Matriz de Confusão",
                labels=dict(x="Predito", y="Real"),
                x=['Não', 'Sim'],
                y=['Não', 'Sim'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            prob_df = pd.DataFrame({
                'Probabilidade': y_pred_proba_test,
                'Real': y_test
            })
            
            fig = px.histogram(
                prob_df,
                x='Probabilidade',
                color='Real',
                title="Distribuição das Probabilidades Preditas",
                nbins=20,
                opacity=0.7
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            importance_df = pd.DataFrame({
                'Variável': [feature_options.get(f, f) for f in selected_features],
                'Importância': np.abs(model.coef_[0])
            }).sort_values('Importância', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importância',
                y='Variável',
                orientation='h',
                title="Importância das Variáveis (Valor Absoluto dos Coeficientes)",
                color='Importância',
                color_continuous_scale='plasma'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def prediction_interface(players_df, games_df):
    """Interface para predições específicas"""
    st.header("🎯 Predições Específicas")
    st.markdown("**Faça perguntas específicas sobre desempenho de jogadores e do time**")
    
    if not SKLEARN_AVAILABLE:
        st.error("⚠️ Scikit-learn não está instalado. Instale com: pip install scikit-learn")
        return
    
    prediction_type = st.selectbox(
        "Tipo de Predição",
        ["Desempenho do Jogador", "Desempenho do Time"],
        help="Escolha se quer prever algo sobre um jogador específico ou sobre o time"
    )
    
    if prediction_type == "Desempenho do Jogador":
        player_specific_predictions(players_df)
    else:
        team_specific_predictions(games_df)

def player_specific_predictions(players_df):
    """Predições específicas para jogadores"""
    st.subheader("👤 Predições de Jogadores")
    
    active_players = players_df[players_df['jogos-disputados'] >= 5].copy()
    
    if len(active_players) == 0:
        st.warning("Não há jogadores com dados suficientes para predição.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        player_options = {}
        for idx, row in active_players.iterrows():
            pos_name = {1: 'Guard', 2: 'Forward', 3: 'Center', 4: 'Forward-Center', 5: 'Center-Forward'}
            position = pos_name.get(row['posicao-g-f-fc-cf-c'], f"Pos-{row['posicao-g-f-fc-cf-c']}")
            
            if 'nome-jogador' in row and pd.notna(row['nome-jogador']) and row['nome-jogador'].strip():
                player_name = f"{row['nome-jogador']} ({position})"
            else:
                player_name = f"{position} #{idx} ({row['idade']} anos)"
            
            player_options[player_name] = idx
        
        selected_player_name = st.selectbox(
            "Selecione o Jogador:",
            options=list(player_options.keys()),
            help="Escolha o jogador para fazer a predição"
        )
        
        selected_player_idx = player_options[selected_player_name]
        player_data = active_players.loc[selected_player_idx]
        
        stat_type = st.selectbox(
            "O que queremos prever?",
            ["Pontos", "Rebotes", "Assistências"],
            help="Escolha a estatística que quer prever"
        )
        
        if stat_type == "Pontos":
            current_avg = player_data['pontos'] / player_data['jogos-disputados']
            target_value = st.number_input(
                f"Quantos {stat_type.lower()} o jogador fará?",
                min_value=0,
                max_value=100,
                value=int(current_avg),
                help=f"Média atual: {current_avg:.1f} por jogo"
            )
            stat_column = 'pontos'
        elif stat_type == "Rebotes":
            current_avg = player_data['rebotes-totais'] / player_data['jogos-disputados']
            target_value = st.number_input(
                f"Quantos {stat_type.lower()} o jogador fará?",
                min_value=0,
                max_value=30,
                value=int(current_avg),
                help=f"Média atual: {current_avg:.1f} por jogo"
            )
            stat_column = 'rebotes-totais'
        else:  
            current_avg = player_data['assistencias'] / player_data['jogos-disputados']
            target_value = st.number_input(
                f"Quantas {stat_type.lower()} o jogador fará?",
                min_value=0,
                max_value=20,
                value=int(current_avg),
                help=f"Média atual: {current_avg:.1f} por jogo"
            )
            stat_column = 'assistencias'
    
    with col2:
        st.subheader("📊 Dados do Jogador Selecionado")
        
        stats_to_show = {
            'Posição': {1: 'Guard', 2: 'Forward', 3: 'Center', 4: 'Forward-Center', 5: 'Center-Forward'}.get(player_data['posicao-g-f-fc-cf-c'], 'N/A'),
            'Idade': f"{player_data['idade']} anos",
            'Jogos': player_data['jogos-disputados'],
            'Minutos/Jogo': f"{player_data['minutos'] / player_data['jogos-disputados']:.1f}",
            'Pontos/Jogo': f"{player_data['pontos'] / player_data['jogos-disputados']:.1f}",
            'Rebotes/Jogo': f"{player_data['rebotes-totais'] / player_data['jogos-disputados']:.1f}",
            'Assistências/Jogo': f"{player_data['assistencias'] / player_data['jogos-disputados']:.1f}",
            '% Arremessos': f"{player_data['porcentagem-arremessos']*100:.1f}%"
        }
        
        for stat_name, stat_value in stats_to_show.items():
            st.metric(stat_name, stat_value)
    
    if st.button("🔮 Fazer Predição", type="primary"):
        make_player_prediction(active_players, selected_player_idx, stat_column, target_value, stat_type)

def make_player_prediction(players_df, player_idx, stat_column, target_value, stat_type):
    """Faz a predição para um jogador específico"""
    
    feature_columns = [
        'idade', 'jogos-disputados', 'minutos', 'arremessos-tentados',
        'porcentagem-arremessos', 'rebotes-totais', 'assistencias'
    ]
    
    available_features = [col for col in feature_columns if col in players_df.columns and col != stat_column]
    
    if len(available_features) < 3:
        st.error("Dados insuficientes para fazer a predição.")
        return
    
    X = players_df[available_features].fillna(0)
    y = players_df[stat_column].fillna(0)
    
    model = LinearRegression()
    model.fit(X, y)
    
    player_features = players_df.loc[player_idx, available_features].values.reshape(1, -1)
    
    predicted_total = model.predict(player_features)[0]
    games_played = players_df.loc[player_idx, 'jogos-disputados']
    predicted_per_game = predicted_total / games_played if games_played > 0 else 0
    
    similar_players = players_df[
        (abs(players_df['idade'] - players_df.loc[player_idx, 'idade']) <= 3) &
        (abs(players_df['posicao-g-f-fc-cf-c'] - players_df.loc[player_idx, 'posicao-g-f-fc-cf-c']) <= 1)
    ]
    
    if len(similar_players) > 3:
        similar_avg = similar_players[stat_column].mean() / similar_players['jogos-disputados'].mean()
        target_per_game = target_value
        
        diff_from_avg = abs(target_per_game - similar_avg)
        probability = max(0, min(100, 100 - (diff_from_avg * 10)))
    else:
        probability = 50 

    st.markdown("---")
    st.subheader("🎯 Resultado da Predição")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"Predição Atual do Modelo",
            f"{predicted_per_game:.1f} {stat_type.lower()}/jogo",
            help="Baseado no desempenho histórico do jogador"
        )
    
    with col2:
        comparison = "⬆️" if target_value > predicted_per_game else "⬇️" if target_value < predicted_per_game else "➡️"
        st.metric(
            f"Meta Desejada",
            f"{target_value} {stat_type.lower()}/jogo",
            delta=f"{comparison} {abs(target_value - predicted_per_game):.1f}"
        )
    
    with col3:
        probability_color = "🟢" if probability > 70 else "🟡" if probability > 40 else "🔴"
        st.metric(
            "Probabilidade de Sucesso",
            f"{probability_color} {probability:.0f}%",
            help="Baseado em jogadores similares"
        )
    
    if probability > 70:
        interpretation = "🎉 **Alta probabilidade!** O jogador tem boas chances de atingir essa meta."
    elif probability > 40:
        interpretation = "⚠️ **Probabilidade moderada.** A meta é desafiadora mas possível."
    else:
        interpretation = "🚨 **Baixa probabilidade.** A meta é muito ambiciosa para o perfil atual do jogador."
    
    st.markdown(f"**Interpretação:** {interpretation}")

def team_specific_predictions(games_df):
    """Predições específicas para o time"""
    st.subheader("🏀 Predições do Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team_stat = st.selectbox(
            "O que queremos prever para o time?",
            ["Pontos", "Rebotes", "Assistências"],
            help="Escolha a estatística do time que quer prever"
        )
        
        if team_stat == "Pontos":
            current_avg = games_df['pontos'].mean()
            target_value = st.number_input(
                f"Quantos {team_stat.lower()} o time fará no próximo jogo?",
                min_value=60,
                max_value=150,
                value=int(current_avg),
                help=f"Média atual: {current_avg:.1f} por jogo"
            )
            stat_column = 'pontos'
        elif team_stat == "Rebotes":
            current_avg = games_df['rebotes-totais'].mean()
            target_value = st.number_input(
                f"Quantos {team_stat.lower()} o time fará no próximo jogo?",
                min_value=20,
                max_value=80,
                value=int(current_avg),
                help=f"Média atual: {current_avg:.1f} por jogo"
            )
            stat_column = 'rebotes-totais'
        else: 
            current_avg = games_df['assistencias'].mean()
            target_value = st.number_input(
                f"Quantas {team_stat.lower()} o time fará no próximo jogo?",
                min_value=10,
                max_value=40,
                value=int(current_avg),
                help=f"Média atual: {current_avg:.1f} por jogo"
            )
            stat_column = 'assistencias'
        
        game_context = st.selectbox(
            "Contexto do jogo:",
            ["Casa", "Fora"],
            help="O time joga em casa ou fora?"
        )
    
    with col2:
        st.subheader("📊 Estatísticas Atuais do Time")
        
        team_stats = {
            'Jogos Disputados': len(games_df),
            'Vitórias': f"{games_df['resultado'].sum()} ({games_df['resultado'].mean()*100:.1f}%)",
            'Pontos/Jogo': f"{games_df['pontos'].mean():.1f}",
            'Rebotes/Jogo': f"{games_df['rebotes-totais'].mean():.1f}",
            'Assistências/Jogo': f"{games_df['assistencias'].mean():.1f}",
            '% Arremessos': f"{games_df['porcentagem-arremessos'].mean()*100:.1f}%",
            'Em Casa': f"{games_df[games_df['mando-de-jogo']==1]['resultado'].mean()*100:.1f}% vitórias",
            'Fora': f"{games_df[games_df['mando-de-jogo']==0]['resultado'].mean()*100:.1f}% vitórias"
        }
        
        for stat_name, stat_value in team_stats.items():
            st.metric(stat_name, stat_value)
    
    if st.button("🔮 Fazer Predição do Time", type="primary"):
        make_team_prediction(games_df, stat_column, target_value, team_stat, game_context)

def make_team_prediction(games_df, stat_column, target_value, stat_type, game_context):
    """Faz a predição para o time"""
    
    context_value = 1 if game_context == "Casa" else 0
    context_games = games_df[games_df['mando-de-jogo'] == context_value]
    
    if len(context_games) < 3:
        context_games = games_df  
    
    context_avg = context_games[stat_column].mean()
    context_std = context_games[stat_column].std()
    
    if context_std > 0:
        z_score = abs(target_value - context_avg) / context_std
        if z_score <= 1:
            probability = 68
        elif z_score <= 2:
            probability = 32
        else:
            probability = 5
    else:
        probability = 50
    
    recent_games = games_df.tail(5)
    recent_avg = recent_games[stat_column].mean()
    
    if abs(target_value - recent_avg) < abs(target_value - context_avg):
        probability += 10  # Bonus se está mais próximo da tendência recente
    
    probability = min(95, max(5, probability))  # Limitar entre 5% e 95%
    
    st.markdown("---")
    st.subheader("🎯 Resultado da Predição do Time")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            f"Média {game_context}",
            f"{context_avg:.1f} {stat_type.lower()}",
            help=f"Desempenho médio do time jogando {game_context.lower()}"
        )
    
    with col2:
        comparison = "⬆️" if target_value > context_avg else "⬇️" if target_value < context_avg else "➡️"
        st.metric(
            f"Meta Desejada",
            f"{target_value} {stat_type.lower()}",
            delta=f"{comparison} {abs(target_value - context_avg):.1f}"
        )
    
    with col3:
        probability_color = "🟢" if probability > 70 else "🟡" if probability > 40 else "🔴"
        st.metric(
            "Probabilidade de Sucesso",
            f"{probability_color} {probability:.0f}%",
            help="Baseado no histórico do time"
        )
    
    if probability > 70:
        interpretation = "🎉 **Alta probabilidade!** O time tem boas chances de atingir essa marca."
    elif probability > 40:
        interpretation = "⚠️ **Probabilidade moderada.** A meta está dentro da variação normal do time."
    else:
        interpretation = "🚨 **Baixa probabilidade.** A meta está fora do padrão histórico do time."
    
    st.markdown(f"**Interpretação:** {interpretation}")
    
    st.subheader("💡 Fatores que Podem Influenciar")
    
    factors_col1, factors_col2 = st.columns(2)
    
    with factors_col1:
        st.markdown("**Fatores Positivos:**")
        if game_context == "Casa":
            st.markdown("- 🏠 Vantagem de jogar em casa")
        st.markdown("- 📈 Tendência recente do time")
        st.markdown("- 🎯 Motivação da equipe")
        
    with factors_col2:
        st.markdown("**Fatores de Risco:**")
        if game_context == "Fora":
            st.markdown("- ✈️ Desgaste de viagem")
        st.markdown("- 🏥 Possíveis lesões")
        st.markdown("- 🛡️ Qualidade da defesa adversária")

def main():
    """Função principal da aplicação"""
    st.title("🏀 Dallas Mavericks 2024-25")
    st.markdown("### Análise Exploratória de Dados - Temporada 2024-25")
    
    players_df, games_df = load_data()
    
    if players_df is None or games_df is None:
        st.error("Não foi possível carregar os dados. Verifique se os arquivos estão no local correto.")
        return
    
    create_summary_metrics(players_df, games_df)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 Jogadores", "🏀 Jogos", "🧠 Análise Avançada", "🔍 Interativa", "🎯 Predições Específicas"])
    
    with tab1:
        player_analysis(players_df)
    
    with tab2:
        game_analysis(games_df)
    
    with tab3:
        advanced_analysis(players_df, games_df)
    
    with tab4:
        interactive_analysis(players_df, games_df)
    
    with tab5:
        prediction_interface(players_df, games_df)
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            <p>📊 Dashboard desenvolvido para análise exploratória dos dados dos Dallas Mavericks</p>
            <p>Temporada 2024-25 • Dados processados e limpos automaticamente</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
