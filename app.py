"""
Ticker Scenario Analyzer
Una aplicación para obtener datos históricos de múltiples tickers
y generar matrices de precios para análisis de escenarios.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de página
st.set_page_config(
    page_title="Ticker Scenario Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado con estética financiera/terminal
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --accent-green: #00ff88;
        --accent-red: #ff4757;
        --accent-blue: #4facfe;
        --text-primary: #e8e8e8;
        --text-muted: #6b7280;
        --border-color: #1e1e2e;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-muted);
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    
    .ticker-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid var(--accent-green);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: var(--accent-green);
        font-size: 0.85rem;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .ticker-badge:hover {
        background: linear-gradient(135deg, #2a1a1e 0%, #3e1616 100%);
        border-color: var(--accent-red);
        color: var(--accent-red);
        box-shadow: 0 0 20px rgba(255, 71, 87, 0.2);
    }
    
    .ticker-badge:hover::after {
        content: ' ✕';
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.3);
    }
    
    div[data-testid="stDataFrame"] {
        background: var(--bg-secondary);
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    .status-success {
        color: var(--accent-green);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .status-error {
        color: var(--accent-red);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .ticker-hint {
        color: var(--text-muted);
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .scenario-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1a 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .scenario-max-return {
        border-left-color: var(--accent-green);
    }
    
    .scenario-min-vol {
        border-left-color: var(--accent-blue);
    }
    
    .scenario-optimal {
        border-left-color: #ffd700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# ALGORITMO GENÉTICO PARA OPTIMIZACIÓN
# ============================================

class PortfolioGeneticOptimizer:
    """Optimizador de portafolio usando algoritmo genético."""
    
    def __init__(self, returns: pd.DataFrame, population_size: int = 200, 
                 generations: int = 150, mutation_rate: float = 0.15,
                 elite_size: int = 10, risk_free_rate: float = 0.02):
        self.returns = returns
        self.n_assets = len(returns.columns)
        self.tickers = returns.columns.tolist()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.risk_free_rate = risk_free_rate / 252  # Diario
        
        # Pre-calcular estadísticas
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        
        # Almacenar todos los portafolios evaluados
        self.all_portfolios = []
    
    def _create_individual(self) -> np.ndarray:
        """Crea un individuo (portafolio) aleatorio que suma 100%."""
        weights = np.random.random(self.n_assets)
        return weights / weights.sum()
    
    def _create_population(self) -> np.ndarray:
        """Crea población inicial."""
        return np.array([self._create_individual() for _ in range(self.population_size)])
    
    def _portfolio_return(self, weights: np.ndarray) -> float:
        """Calcula el retorno esperado del portafolio."""
        return np.dot(weights, self.mean_returns) * 252  # Anualizado
    
    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calcula la volatilidad del portafolio."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
    
    def _portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Calcula el Sharpe ratio del portafolio."""
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        return (ret - self.risk_free_rate * 252) / vol if vol > 0 else 0
    
    def _diversity_bonus(self, weights: np.ndarray) -> float:
        """Bonus por diversificación (penaliza concentración extrema)."""
        # Entropía normalizada como medida de diversificación
        weights_nonzero = weights[weights > 0.01]
        if len(weights_nonzero) <= 1:
            return 0
        entropy = -np.sum(weights_nonzero * np.log(weights_nonzero + 1e-10))
        max_entropy = np.log(self.n_assets)
        return entropy / max_entropy
    
    def _fitness_max_return(self, weights: np.ndarray) -> float:
        """Fitness para maximizar retorno con bonus de diversificación."""
        ret = self._portfolio_return(weights)
        diversity = self._diversity_bonus(weights)
        return ret + 0.1 * diversity
    
    def _fitness_min_volatility(self, weights: np.ndarray) -> float:
        """Fitness para minimizar volatilidad con bonus de diversificación."""
        vol = self._portfolio_volatility(weights)
        diversity = self._diversity_bonus(weights)
        return -vol + 0.05 * diversity  # Negativo porque minimizamos
    
    def _fitness_max_sharpe(self, weights: np.ndarray) -> float:
        """Fitness para maximizar Sharpe con bonus de diversificación."""
        sharpe = self._portfolio_sharpe(weights)
        diversity = self._diversity_bonus(weights)
        return sharpe + 0.1 * diversity
    
    def _selection(self, population: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
        """Selección por torneo."""
        selected = []
        for _ in range(self.population_size - self.elite_size):
            # Torneo de 3
            idx = np.random.choice(len(population), 3, replace=False)
            winner_idx = idx[np.argmax(fitness_values[idx])]
            selected.append(population[winner_idx])
        return np.array(selected)
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Crossover aritmético."""
        alpha = np.random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child / child.sum()  # Normalizar
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutación gaussiana."""
        if np.random.random() < self.mutation_rate:
            mutation = np.random.normal(0, 0.1, self.n_assets)
            individual = individual + mutation
            individual = np.maximum(individual, 0)  # No negativos
            individual = individual / individual.sum()  # Normalizar
        return individual
    
    def _evolve(self, population: np.ndarray, fitness_func) -> tuple:
        """Una generación de evolución."""
        fitness_values = np.array([fitness_func(ind) for ind in population])
        
        # Guardar todos los portafolios
        for i, ind in enumerate(population):
            self.all_portfolios.append({
                'weights': ind.copy(),
                'return': self._portfolio_return(ind),
                'volatility': self._portfolio_volatility(ind),
                'sharpe': self._portfolio_sharpe(ind)
            })
        
        # Élite
        elite_idx = np.argsort(fitness_values)[-self.elite_size:]
        elite = population[elite_idx]
        
        # Selección
        selected = self._selection(population, fitness_values)
        
        # Crossover y mutación
        new_population = list(elite)
        for i in range(0, len(selected) - 1, 2):
            child1 = self._crossover(selected[i], selected[i + 1])
            child2 = self._crossover(selected[i + 1], selected[i])
            new_population.append(self._mutate(child1))
            new_population.append(self._mutate(child2))
        
        # Completar población si es necesario
        while len(new_population) < self.population_size:
            new_population.append(self._create_individual())
        
        return np.array(new_population[:self.population_size]), np.max(fitness_values)
    
    def optimize(self, objective: str = 'sharpe', progress_callback=None) -> dict:
        """
        Ejecuta la optimización.
        
        Args:
            objective: 'max_return', 'min_volatility', 'sharpe'
            progress_callback: función para actualizar progreso
        
        Returns:
            dict con los mejores pesos y métricas
        """
        fitness_funcs = {
            'max_return': self._fitness_max_return,
            'min_volatility': self._fitness_min_volatility,
            'sharpe': self._fitness_max_sharpe
        }
        
        fitness_func = fitness_funcs[objective]
        population = self._create_population()
        
        best_fitness_history = []
        
        for gen in range(self.generations):
            population, best_fitness = self._evolve(population, fitness_func)
            best_fitness_history.append(best_fitness)
            
            if progress_callback:
                progress_callback((gen + 1) / self.generations)
        
        # Encontrar el mejor individuo final
        fitness_values = np.array([fitness_func(ind) for ind in population])
        best_idx = np.argmax(fitness_values)
        best_weights = population[best_idx]
        
        return {
            'weights': best_weights,
            'tickers': self.tickers,
            'return': self._portfolio_return(best_weights) * 100,
            'volatility': self._portfolio_volatility(best_weights) * 100,
            'sharpe': self._portfolio_sharpe(best_weights),
            'fitness_history': best_fitness_history
        }
    
    def get_efficient_frontier_data(self) -> pd.DataFrame:
        """Retorna todos los portafolios evaluados."""
        return pd.DataFrame(self.all_portfolios)


# ============================================
# FUNCIONES DE UTILIDAD
# ============================================

def verify_ticker(symbol: str) -> bool:
    """Verifica si un ticker existe en Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get('regularMarketPrice') is not None or info.get('previousClose') is not None
    except Exception:
        return False

def add_ticker_callback():
    """Callback para agregar ticker cuando se presiona Enter."""
    input_value = st.session_state.ticker_input_widget.upper().strip()
    
    if not input_value:
        return
    
    if input_value in st.session_state.valid_tickers:
        st.session_state.last_error = f"'{input_value}' ya está en la lista"
        return
    
    with st.spinner(f"Verificando {input_value}..."):
        if verify_ticker(input_value):
            st.session_state.valid_tickers.append(input_value)
            st.session_state.last_error = None
        else:
            st.session_state.last_error = f"'{input_value}' no encontrado en Yahoo Finance"

def remove_ticker(ticker: str):
    """Elimina un ticker de la lista."""
    if ticker in st.session_state.valid_tickers:
        st.session_state.valid_tickers.remove(ticker)

def create_efficient_frontier_plot(all_portfolios: pd.DataFrame, 
                                    max_return_portfolio: dict,
                                    min_vol_portfolio: dict,
                                    optimal_portfolio: dict) -> go.Figure:
    """Crea el gráfico de frontera eficiente."""
    
    fig = go.Figure()
    
    # Todos los portafolios simulados
    fig.add_trace(go.Scatter(
        x=all_portfolios['volatility'] * 100,
        y=all_portfolios['return'] * 100,
        mode='markers',
        marker=dict(
            size=4,
            color=all_portfolios['sharpe'],
            colorscale='Viridis',
            colorbar=dict(title='Sharpe Ratio'),
            opacity=0.6
        ),
        name='Portafolios Simulados',
        hovertemplate='Volatilidad: %{x:.2f}%<br>Retorno: %{y:.2f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>'
    ))
    
    # Portafolio máximo retorno
    fig.add_trace(go.Scatter(
        x=[min_vol_portfolio['volatility']],
        y=[min_vol_portfolio['return']],
        mode='markers',
        marker=dict(size=20, color='#4facfe', symbol='star', line=dict(width=2, color='white')),
        name=f"Mín. Volatilidad ({min_vol_portfolio['volatility']:.2f}%)",
        hovertemplate='<b>Mínima Volatilidad</b><br>Volatilidad: %{x:.2f}%<br>Retorno: %{y:.2f}%<extra></extra>'
    ))
    
    # Portafolio mínima volatilidad
    fig.add_trace(go.Scatter(
        x=[max_return_portfolio['volatility']],
        y=[max_return_portfolio['return']],
        mode='markers',
        marker=dict(size=20, color='#00ff88', symbol='star', line=dict(width=2, color='white')),
        name=f"Máx. Retorno ({max_return_portfolio['return']:.2f}%)",
        hovertemplate='<b>Máximo Retorno</b><br>Volatilidad: %{x:.2f}%<br>Retorno: %{y:.2f}%<extra></extra>'
    ))
    
    # Portafolio óptimo (máximo Sharpe)
    fig.add_trace(go.Scatter(
        x=[optimal_portfolio['volatility']],
        y=[optimal_portfolio['return']],
        mode='markers',
        marker=dict(size=25, color='#ffd700', symbol='star', line=dict(width=3, color='white')),
        name=f"Óptimo - Sharpe ({optimal_portfolio['sharpe']:.3f})",
        hovertemplate='<b>Portafolio Óptimo</b><br>Volatilidad: %{x:.2f}%<br>Retorno: %{y:.2f}%<br>Sharpe: ' + f"{optimal_portfolio['sharpe']:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text='Frontera Eficiente - Optimización por Algoritmo Genético',
            font=dict(size=20, family='Space Grotesk')
        ),
        xaxis_title='Volatilidad Anualizada (%)',
        yaxis_title='Retorno Esperado Anualizado (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(10, 10, 15, 0.8)',
        plot_bgcolor='rgba(18, 18, 26, 0.8)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        height=600
    )
    
    return fig

def create_weights_comparison_chart(max_ret: dict, min_vol: dict, optimal: dict) -> go.Figure:
    """Crea gráfico comparativo de pesos."""
    
    tickers = max_ret['tickers']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Máx. Retorno',
        x=tickers,
        y=[w * 100 for w in max_ret['weights']],
        marker_color='#00ff88'
    ))
    
    fig.add_trace(go.Bar(
        name='Mín. Volatilidad',
        x=tickers,
        y=[w * 100 for w in min_vol['weights']],
        marker_color='#4facfe'
    ))
    
    fig.add_trace(go.Bar(
        name='Óptimo (Sharpe)',
        x=tickers,
        y=[w * 100 for w in optimal['weights']],
        marker_color='#ffd700'
    ))
    
    fig.update_layout(
        title='Comparación de Pesos por Escenario',
        xaxis_title='Ticker',
        yaxis_title='Peso (%)',
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(10, 10, 15, 0.8)',
        plot_bgcolor='rgba(18, 18, 26, 0.8)',
        height=400
    )
    
    return fig


# ============================================
# INTERFAZ PRINCIPAL
# ============================================

# Header
st.markdown('<h1 class="main-header">Ticker Scenario Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">[ Obtén datos históricos de múltiples tickers para análisis de escenarios ]</p>', unsafe_allow_html=True)

# Inicializar estado de sesión
if 'valid_tickers' not in st.session_state:
    st.session_state.valid_tickers = []
if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = ''
if 'last_error' not in st.session_state:
    st.session_state.last_error = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Sidebar para configuración
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    st.markdown("---")
    
    # Temporalidad
    st.markdown("#### 📅 Temporalidad")
    interval_options = {
        "1 minuto": "1m",
        "2 minutos": "2m",
        "5 minutos": "5m",
        "15 minutos": "15m",
        "30 minutos": "30m",
        "1 hora": "1h",
        "1 día": "1d",
        "5 días": "5d",
        "1 semana": "1wk",
        "1 mes": "1mo",
        "3 meses": "3mo"
    }
    selected_interval = st.selectbox(
        "Intervalo",
        options=list(interval_options.keys()),
        index=6,
        help="Nota: Intervalos menores a 1 día solo disponibles para últimos 7 días"
    )
    interval = interval_options[selected_interval]
    
    st.markdown("---")
    
    # Rango de fechas
    st.markdown("#### Rango de Fechas")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Fecha inicio",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "Fecha fin",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    if interval in ["1m", "2m", "5m", "15m", "30m", "1h"]:
        max_days = 7 if interval in ["1m", "2m", "5m", "15m", "30m"] else 60
        if (end_date - start_date).days > max_days:
            st.warning(f"⚠️ Intervalos intradía limitados a {max_days} días. Ajustando...")
            start_date = end_date - timedelta(days=max_days)
    
    st.markdown("---")
    
    # Parámetros del algoritmo genético
    st.markdown("#### 🧬 Algoritmo Genético")
    
    ga_population = st.slider("Población", 50, 500, 200, 50,
                               help="Número de portafolios por generación")
    ga_generations = st.slider("Generaciones", 50, 300, 150, 25,
                                help="Número de iteraciones evolutivas")
    ga_mutation = st.slider("Tasa de mutación", 0.05, 0.30, 0.15, 0.05,
                            help="Probabilidad de mutación")
    
    st.markdown("---")
    
    st.markdown("#### Información")
    st.caption("""
    **Datos vía Yahoo Finance**
    
    • Intervalos < 1d: últimos 7 días
    • Intervalo 1h: últimos 60 días
    • Otros: sin límite
    
    **Algoritmo Genético**
    
    • Optimiza pesos del portafolio
    • Siempre suma 100%
    • Premio por diversificación
    """)

# Área principal - Tickers
st.markdown("### 🏷️ Tickers")

st.text_input(
    "Agregar ticker",
    key="ticker_input_widget",
    placeholder="Escribe un símbolo y presiona Enter (ej: AAPL, MSFT, BTC-USD)",
    on_change=add_ticker_callback,
    label_visibility="collapsed"
)

st.markdown('<p class="ticker-hint">Presiona Enter para validar y agregar el ticker</p>', unsafe_allow_html=True)

if st.session_state.last_error:
    st.error(st.session_state.last_error)

if st.session_state.valid_tickers:
    st.markdown("**Tickers activos:** (click para eliminar)")
    
    cols = st.columns(min(len(st.session_state.valid_tickers), 8))
    
    for idx, ticker in enumerate(st.session_state.valid_tickers):
        col_idx = idx % 8
        with cols[col_idx]:
            if st.button(
                f"🏷️ {ticker}",
                key=f"remove_{ticker}",
                use_container_width=True,
                help=f"Click para eliminar {ticker}"
            ):
                remove_ticker(ticker)
                st.rerun()
else:
    st.markdown('<p class="ticker-hint">No hay tickers agregados. Escribe un símbolo arriba para comenzar.</p>', unsafe_allow_html=True)

st.markdown("---")

# Botón de análisis
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "🚀 Obtener Datos Históricos",
        use_container_width=True,
        type="primary",
        disabled=len(st.session_state.valid_tickers) == 0
    )

# Procesar datos
if analyze_button and st.session_state.valid_tickers:
    
    with st.spinner("Descargando datos de Yahoo Finance..."):
        
        results = {}
        errors = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, ticker in enumerate(st.session_state.valid_tickers):
            status_text.markdown(f'<p class="status-success">Descargando: {ticker}...</p>', unsafe_allow_html=True)
            
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )
                
                if not data.empty:
                    results[ticker] = data
                else:
                    errors.append(f"{ticker}: Sin datos disponibles")
                    
            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(st.session_state.valid_tickers))
        
        status_text.empty()
        progress_bar.empty()
    
    if errors:
        with st.expander("⚠️ Errores encontrados", expanded=True):
            for error in errors:
                st.markdown(f'<p class="status-error">• {error}</p>', unsafe_allow_html=True)
    
    if results:
        # Guardar resultados en session state
        close_prices = pd.DataFrame()
        for ticker, data in results.items():
            if 'Close' in data.columns:
                close_prices[ticker] = data['Close']
            elif ('Close', ticker) in data.columns:
                close_prices[ticker] = data[('Close', ticker)]
        
        returns = close_prices.pct_change().dropna()
        
        st.session_state.analysis_results = {
            'results': results,
            'close_prices': close_prices,
            'returns': returns,
            'selected_interval': selected_interval
        }

# Mostrar resultados si existen
if st.session_state.analysis_results:
    results = st.session_state.analysis_results['results']
    close_prices = st.session_state.analysis_results['close_prices']
    returns = st.session_state.analysis_results['returns']
    selected_interval = st.session_state.analysis_results['selected_interval']
    
    st.markdown("### 📈 Resultados")
    
    volatility_per_ticker = returns.std(ddof=1) * 100
    avg_volatility = volatility_per_ticker.mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tickers", len(results))
    with col2:
        st.metric("Registros", len(close_prices))
    with col3:
        st.metric("Intervalo", selected_interval)
    with col4:
        st.metric("Volatilidad Prom.", f"{avg_volatility:.2f}%")
    
    st.markdown("---")
    
    # Tabs para diferentes vistas
    tab1, tab2, tab3 = st.tabs(["📊 Precios de Cierre", "📈 Rendimientos", "🎯 Escenarios"])
    
    with tab1:
        st.dataframe(
            close_prices.style.format("{:.2f}"),
            use_container_width=True,
            height=400
        )
        
        st.markdown("##### Evolución de Precios")
        st.line_chart(close_prices, use_container_width=True)
    
    with tab2:
        st.markdown("##### Rendimientos (%)")
        st.dataframe(
            (returns * 100).style.format("{:.4f}%"),
            use_container_width=True,
            height=400
        )
        
        st.markdown("##### Estadísticas de Rendimientos")
        stats = pd.DataFrame({
            'Promedio (%)': returns.mean() * 100,
            'Volatilidad (%)': returns.std(ddof=1) * 100,
            'Sharpe (aprox)': (returns.mean() / returns.std()) * (252 ** 0.5)
        })
        st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)
        
        if len(results) > 1:
            correlation = returns.corr()
            
            st.markdown("##### Matriz de Correlación entre precios de Tickers")
            st.dataframe(
                correlation.style.format("{:.4f}").background_gradient(cmap='RdYlGn', vmin=-1, vmax=1),
                use_container_width=True
            )
            
            st.markdown("##### Matriz de Covarianza entre los precios de cierre de Tickers")
            covariance = returns.cov(ddof=1)
            st.dataframe(
                covariance.style.format("{:.6f}").background_gradient(cmap='coolwarm'),
                use_container_width=True
            )
            st.caption("Covarianza muestral de los rendimientos (equivalente a COVARIANCE.S en Excel)")
        else:
            st.info("Se necesitan al menos 2 tickers para calcular correlaciones y covarianzas.")
    
    with tab3:
        st.markdown("##### 🎯 Optimización de Portafolio")
        
        if len(results) < 2:
            st.warning("Se necesitan al menos 2 tickers para generar escenarios de optimización.")
        else:
            st.markdown("""
            El algoritmo genético buscará tres escenarios óptimos:
            - **🟢 Máximo Retorno**: Maximiza el retorno esperado
            - **🔵 Mínima Volatilidad**: Minimiza el riesgo del portafolio  
            - **🟡 Óptimo Global**: Maximiza el ratio Sharpe (mejor relación retorno/riesgo)
            
            Todos los escenarios suman exactamente 100% y premian la diversificación.
            """)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                generate_scenarios = st.button(
                    "🧬 Generar Escenarios",
                    use_container_width=True,
                    type="primary"
                )
            
            if generate_scenarios:
                st.markdown("---")
                
                progress_container = st.empty()
                status_container = st.empty()
                
                # Optimización 1: Máximo Retorno
                status_container.markdown("**Optimizando: Máximo Retorno...**")
                progress_bar = progress_container.progress(0)
                
                optimizer = PortfolioGeneticOptimizer(
                    returns,
                    population_size=ga_population,
                    generations=ga_generations,
                    mutation_rate=ga_mutation
                )
                max_return_result = optimizer.optimize('max_return', lambda p: progress_bar.progress(p))
                
                # Optimización 2: Mínima Volatilidad
                status_container.markdown("**Optimizando: Mínima Volatilidad...**")
                progress_bar = progress_container.progress(0)
                
                optimizer2 = PortfolioGeneticOptimizer(
                    returns,
                    population_size=ga_population,
                    generations=ga_generations,
                    mutation_rate=ga_mutation
                )
                min_vol_result = optimizer2.optimize('min_volatility', lambda p: progress_bar.progress(p))
                
                # Optimización 3: Máximo Sharpe (Óptimo)
                status_container.markdown("**Optimizando: Portafolio Óptimo (Sharpe)...**")
                progress_bar = progress_container.progress(0)
                
                optimizer3 = PortfolioGeneticOptimizer(
                    returns,
                    population_size=ga_population,
                    generations=ga_generations,
                    mutation_rate=ga_mutation
                )
                optimal_result = optimizer3.optimize('sharpe', lambda p: progress_bar.progress(p))
                
                # Combinar todos los portafolios para el gráfico
                all_portfolios = pd.concat([
                    optimizer.get_efficient_frontier_data(),
                    optimizer2.get_efficient_frontier_data(),
                    optimizer3.get_efficient_frontier_data()
                ], ignore_index=True)

                # Eliminar duplicados basándose solo en las columnas numéricas
                all_portfolios = all_portfolios.drop_duplicates(
                    subset=['return', 'volatility', 'sharpe']
                )
                
                progress_container.empty()
                status_container.empty()
                
                st.success("✅ Optimización completada!")
                
                # Mostrar gráfico de frontera eficiente
                st.markdown("##### 📈 Frontera Eficiente")
                fig = create_efficient_frontier_plot(
                    all_portfolios,
                    max_return_result,
                    min_vol_result,
                    optimal_result
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar resultados de cada escenario
                st.markdown("##### 📋 Resultados de Escenarios")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**🟢 Máximo Retorno**")
                    st.metric("Retorno Anual", f"{max_return_result['return']:.2f}%")
                    st.metric("Volatilidad", f"{max_return_result['volatility']:.2f}%")
                    st.metric("Sharpe Ratio", f"{max_return_result['sharpe']:.3f}")
                    
                    st.markdown("**Pesos:**")
                    for t, w in zip(max_return_result['tickers'], max_return_result['weights']):
                        if w > 0.01:
                            st.write(f"• {t}: {w*100:.1f}%")
                
                with col2:
                    st.markdown("**🔵 Mínima Volatilidad**")
                    st.metric("Retorno Anual", f"{min_vol_result['return']:.2f}%")
                    st.metric("Volatilidad", f"{min_vol_result['volatility']:.2f}%")
                    st.metric("Sharpe Ratio", f"{min_vol_result['sharpe']:.3f}")
                    
                    st.markdown("**Pesos:**")
                    for t, w in zip(min_vol_result['tickers'], min_vol_result['weights']):
                        if w > 0.01:
                            st.write(f"• {t}: {w*100:.1f}%")
                
                with col3:
                    st.markdown("**🟡 Óptimo (Max Sharpe)**")
                    st.metric("Retorno Anual", f"{optimal_result['return']:.2f}%")
                    st.metric("Volatilidad", f"{optimal_result['volatility']:.2f}%")
                    st.metric("Sharpe Ratio", f"{optimal_result['sharpe']:.3f}")
                    
                    st.markdown("**Pesos:**")
                    for t, w in zip(optimal_result['tickers'], optimal_result['weights']):
                        if w > 0.01:
                            st.write(f"• {t}: {w*100:.1f}%")
                
                # Gráfico comparativo de pesos
                st.markdown("##### ⚖️ Comparación de Pesos")
                weights_fig = create_weights_comparison_chart(
                    max_return_result,
                    min_vol_result,
                    optimal_result
                )
                st.plotly_chart(weights_fig, use_container_width=True)
                
                # Tabla resumen
                st.markdown("##### 📊 Tabla Resumen")
                summary_df = pd.DataFrame({
                    'Escenario': ['Máx. Retorno', 'Mín. Volatilidad', 'Óptimo (Sharpe)'],
                    'Retorno (%)': [max_return_result['return'], min_vol_result['return'], optimal_result['return']],
                    'Volatilidad (%)': [max_return_result['volatility'], min_vol_result['volatility'], optimal_result['volatility']],
                    'Sharpe Ratio': [max_return_result['sharpe'], min_vol_result['sharpe'], optimal_result['sharpe']]
                })
                st.dataframe(summary_df.style.format({
                    'Retorno (%)': '{:.2f}',
                    'Volatilidad (%)': '{:.2f}',
                    'Sharpe Ratio': '{:.3f}'
                }), use_container_width=True)
                
                # Exportar pesos
                st.markdown("##### 💾 Exportar Pesos")
                
                weights_export = pd.DataFrame({
                    'Ticker': max_return_result['tickers'],
                    'Máx. Retorno (%)': [w * 100 for w in max_return_result['weights']],
                    'Mín. Volatilidad (%)': [w * 100 for w in min_vol_result['weights']],
                    'Óptimo (%)': [w * 100 for w in optimal_result['weights']]
                })
                
                csv_weights = weights_export.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Pesos (CSV)",
                    data=csv_weights,
                    file_name=f"pesos_portafolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # Botón de descarga (fuera de los tabs)
    st.markdown("---")
    st.markdown("#### Exportar Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = close_prices.to_csv()
        st.download_button(
            label="📥 Precios (CSV)",
            data=csv,
            file_name=f"precios_cierre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if len(results) > 1:
            corr_csv = returns.corr().to_csv()
            st.download_button(
                label="📥 Correlación (CSV)",
                data=corr_csv,
                file_name=f"correlacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        if len(results) > 1:
            cov_csv = returns.cov(ddof=1).to_csv()
            st.download_button(
                label="📥 Covarianza (CSV)",
                data=cov_csv,
                file_name=f"covarianza_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #6b7280; font-size: 0.8rem;">Desarrollado con Streamlit + yfinance | Optimización por Algoritmo Genético</p>',
    unsafe_allow_html=True
)
