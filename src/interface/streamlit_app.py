import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Análise de Regressão Linear - Dallas Mavericks",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏀 Análise de Regressão Linear - Dallas Mavericks 2024-25")
st.markdown("---")

# Mapeamento de posições
POSITION_MAP = {
    1: "G (Guard - Armador)",
    2: "F (Forward - Ala)", 
    3: "F-C (Forward-Center - Ala-Pivô)",
    4: "C-F (Center-Forward - Pivô-Ala)",
    5: "C (Center - Pivô)"
}

@st.cache_data
def load_data():
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        players_path = os.path.join(base_path, 'data', 'processed', 'dallas_players_2024-25.csv')
        games_path = os.path.join(base_path, 'data', 'processed', 'dallas_games_2024-25.csv')
        
        players_df = pd.read_csv(players_path)
        games_df = pd.read_csv(games_path)
        return players_df, games_df
    except FileNotFoundError:
        st.error("Arquivos de dados não encontrados. Verifique se os dados estão na pasta 'data/processed/'")
        return None, None

# Função para executar regressão linear
def run_linear_regression(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'mse_train': mean_squared_error(y_train, y_pred_train),
        'mse_test': mean_squared_error(y_test, y_pred_test),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'coef': model.coef_,
        'intercept': model.intercept_
    }
    
    return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, metrics

# Função para formatar dados para exibição
def format_data_for_display(df):
    df_display = df.copy()
    
    if 'posicao-g-f-fc-cf-c' in df_display.columns:
        df_display['posicao-descricao'] = df_display['posicao-g-f-fc-cf-c'].map(
            lambda x: f"{x} - {POSITION_MAP.get(x, 'Desconhecida')}"
        )
        cols = df_display.columns.tolist()
        pos_idx = cols.index('posicao-g-f-fc-cf-c')
        cols.insert(pos_idx + 1, cols.pop(cols.index('posicao-descricao')))
        df_display = df_display[cols]
    
    return df_display

# Carregar dados
players_df, games_df = load_data()

if players_df is not None:
    if 'posicao-g-f-fc-cf-c' in players_df.columns:
        with st.expander("🏀 Mapeamento de Posições", expanded=False):
            st.write("**Legenda das Posições:**")
            col1, col2 = st.columns(2)
            
            with col1:
                for pos_num, pos_desc in list(POSITION_MAP.items())[:3]:
                    st.write(f"**{pos_num}** - {pos_desc}")
            
            with col2:
                for pos_num, pos_desc in list(POSITION_MAP.items())[3:]:
                    st.write(f"**{pos_num}** - {pos_desc}")
            
            if not players_df.empty:
                position_counts = players_df['posicao-g-f-fc-cf-c'].value_counts().sort_index()
                position_labels = [f"{pos} - {POSITION_MAP.get(pos, 'Desconhecida')}" for pos in position_counts.index]
                
                fig_positions = px.bar(
                    x=position_labels,
                    y=position_counts.values,
                    title="Distribuição de Jogadores por Posição",
                    labels={'x': 'Posição', 'y': 'Quantidade de Jogadores'}
                )
                st.plotly_chart(fig_positions, use_container_width=True)
    
    st.sidebar.header("⚙️ Configurações da Análise")
    
    dataset_choice = st.sidebar.selectbox(
        "Selecione o dataset:",
        ["Dados dos Jogadores", "Dados dos Jogos"]
    )
    
    if dataset_choice == "Dados dos Jogadores":
        df = players_df.copy()
        st.subheader("📊 Dataset: Estatísticas dos Jogadores")
        
        if 'posicao-g-f-fc-cf-c' in df.columns:
            st.sidebar.subheader("🏀 Filtro por Posição")
            
            available_positions = sorted(df['posicao-g-f-fc-cf-c'].unique())
            position_options = ["Todas as Posições"] + [f"{pos} - {POSITION_MAP.get(pos, 'Desconhecida')}" for pos in available_positions]
            
            selected_position = st.sidebar.selectbox(
                "Selecionar Posição:",
                position_options
            )
            
            if selected_position != "Todas as Posições":
                pos_num = int(selected_position.split(" - ")[0])
                df = df[df['posicao-g-f-fc-cf-c'] == pos_num]
                st.info(f"Dados filtrados para posição: {selected_position}")
    else:
        df = games_df.copy()
        st.subheader("📊 Dataset: Estatísticas dos Jogos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Registros", len(df))
    with col2:
        st.metric("Colunas Numéricas", len(df.select_dtypes(include=[np.number]).columns))
    with col3:
        st.metric("Valores Nulos", df.isnull().sum().sum())
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.error("Não há colunas numéricas suficientes para análise de regressão.")
    else:
        st.sidebar.subheader("📈 Seleção de Variáveis")
        
        target_variable = st.sidebar.selectbox(
            "Variável Dependente (Y):",
            numeric_columns,
            index=0
        )
        
        available_features = [col for col in numeric_columns if col != target_variable]
        selected_features = st.sidebar.multiselect(
            "Variáveis Independentes (X):",
            available_features,
            default=available_features[:3] if len(available_features) >= 3 else available_features
        )
        
        if not selected_features:
            st.warning("Selecione pelo menos uma variável independente.")
        else:
            st.sidebar.subheader("🔧 Parâmetros do Modelo")
            test_size = st.sidebar.slider("Tamanho do conjunto de teste (%)", 10, 50, 20) / 100
            
            if st.sidebar.button("🚀 Executar Regressão Linear", type="primary"):
                
                X = df[selected_features].dropna()
                y = df.loc[X.index, target_variable]
                
                if len(X) == 0:
                    st.error("Não há dados válidos após remoção de valores nulos.")
                else:
                    model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, metrics = run_linear_regression(X, y, test_size)
                    
                    st.markdown("---")
                    st.header("📊 Resultados da Regressão Linear")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R² (Treino)", f"{metrics['r2_train']:.4f}")
                    with col2:
                        st.metric("R² (Teste)", f"{metrics['r2_test']:.4f}")
                    with col3:
                        st.metric("MSE (Teste)", f"{metrics['mse_test']:.4f}")
                    with col4:
                        st.metric("MAE (Teste)", f"{metrics['mae_test']:.4f}")
                    
                    st.subheader("🔍 Coeficientes do Modelo")
                    coef_df = pd.DataFrame({
                        'Variável': selected_features,
                        'Coeficiente': metrics['coef'],
                        'Coeficiente (Abs)': np.abs(metrics['coef'])
                    }).sort_values('Coeficiente (Abs)', ascending=False)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.dataframe(coef_df, use_container_width=True)
                    with col2:
                        st.write(f"**Intercepto:** {metrics['intercept']:.4f}")
                        
                        st.write("**Equação da Regressão:**")
                        equation = f"Y = {metrics['intercept']:.4f}"
                        for i, feature in enumerate(selected_features):
                            coef = metrics['coef'][i]
                            sign = "+" if coef >= 0 else ""
                            equation += f" {sign} {coef:.4f}*{feature}"
                        st.code(equation)
                    
                    st.subheader("📈 Visualizações")
                    
                    if len(selected_features) == 1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.scatter(
                                x=X[selected_features[0]], 
                                y=y,
                                title=f"Regressão: {target_variable} vs {selected_features[0]}",
                                labels={
                                    'x': selected_features[0],
                                    'y': target_variable
                                }
                            )
                            
                            x_range = np.linspace(X[selected_features[0]].min(), X[selected_features[0]].max(), 100)
                            y_pred_line = model.predict(x_range.reshape(-1, 1))
                            fig.add_trace(go.Scatter(
                                x=x_range, 
                                y=y_pred_line,
                                mode='lines',
                                name='Linha de Regressão',
                                line=dict(color='red', width=2)
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            residuals = y_test - y_pred_test
                            fig_residuals = px.scatter(
                                x=y_pred_test, 
                                y=residuals,
                                title="Gráfico de Resíduos",
                                labels={
                                    'x': 'Valores Preditos',
                                    'y': 'Resíduos'
                                }
                            )
                            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                            st.plotly_chart(fig_residuals, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pred = px.scatter(
                            x=y_test, 
                            y=y_pred_test,
                            title="Valores Reais vs Preditos (Teste)",
                            labels={
                                'x': f'{target_variable} (Real)',
                                'y': f'{target_variable} (Predito)'
                            }
                        )
                        
                        min_val = min(y_test.min(), y_pred_test.min())
                        max_val = max(y_test.max(), y_pred_test.max())
                        fig_pred.add_trace(go.Scatter(
                            x=[min_val, max_val], 
                            y=[min_val, max_val],
                            mode='lines',
                            name='Predição Perfeita',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    with col2:
                        importance_df = pd.DataFrame({
                            'Variável': selected_features,
                            'Importância': np.abs(metrics['coef'])
                        }).sort_values('Importância', ascending=True)
                        
                        fig_importance = px.bar(
                            importance_df,
                            x='Importância',
                            y='Variável',
                            orientation='h',
                            title="Importância das Variáveis (|Coeficiente|)"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    
                    st.subheader("🎯 Fazer Predições Personalizadas")
                    
                    prediction_inputs = {}
                    cols = st.columns(len(selected_features))
                    
                    for i, feature in enumerate(selected_features):
                        with cols[i]:
                            min_val = float(X[feature].min())
                            max_val = float(X[feature].max())
                            mean_val = float(X[feature].mean())
                            
                            prediction_inputs[feature] = st.number_input(
                                f"{feature}:",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                step=(max_val - min_val) / 100
                            )
                    
                    if st.button("🔮 Fazer Predição"):
                        input_array = np.array([list(prediction_inputs.values())])
                        prediction = model.predict(input_array)[0]
                        
                        st.success(f"**Predição para {target_variable}: {prediction:.4f}**")
                        
                        residual_std = np.std(y_test - y_pred_test)
                        confidence_interval = 1.96 * residual_std  # 95% CI aproximado
                        
                        st.info(f"Intervalo de confiança aproximado (95%): [{prediction - confidence_interval:.4f}, {prediction + confidence_interval:.4f}]")
    
    if st.checkbox("📋 Mostrar dados brutos"):
        st.subheader("Dados Brutos")
        
        df_display = format_data_for_display(df)
        st.dataframe(df_display, use_container_width=True)
        
        if st.checkbox("📊 Mostrar estatísticas descritivas"):
            st.subheader("Estatísticas Descritivas")
            st.dataframe(df.describe(), use_container_width=True)

else:
    st.error("Não foi possível carregar os dados. Verifique se os arquivos estão no local correto.")
    st.info("Os arquivos esperados são: 'data/processed/dallas_players_2024-25.csv' e 'data/processed/dallas_games_2024-25.csv'")