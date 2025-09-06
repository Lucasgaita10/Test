
"""
Standalone Streamlit Portfolio Optimization App with Backtesting
Includes portfolio simulation, rebalancing strategies, and performance analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
import io
import warnings
import time
import base64
from PIL import Image

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Vantage Capital - Portfolio Optimization & Backtesting",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Color Schemes
COLORS = [
    '#4A90E2', '#888888', '#162d73', '#333A56', '#50C2A8', '#A7E8F1',
    '#FFA07A', '#9B59B6', '#F39C12', '#E74C3C', '#2ECC71', '#3498DB',
    '#F1C40F', '#E67E22', '#1ABC9C', '#95A5A6', '#34495E', '#8E44AD'
]

CHART_COLORS = [
    '#E74C3C', '#2ECC71', '#3498DB', '#F1C40F', '#E67E22', '#1ABC9C',
    '#95A5A6', '#34495E', '#8E44AD', '#D35400'
]

# ============================================================================
# EMBEDDED PORTFOLIO OPTIMIZER CLASS WITH BACKTESTING
# ============================================================================

class EnhancedPortfolioOptimizer:
    """
    Vantage Capital - Portfolio Optimization Tool with Backtesting Features
    """
   
    def __init__(self):
        # Common attributes
        self.mode = None  # 'backward_looking' or 'forward_looking'
        self.tickers = []
        self.asset_names = []
        self.expected_returns = None
        self.volatilities = None
        self.min_weights = None
        self.max_weights = None
        self.initial_weights = None
        self.lookback_years = None
        self.risk_free_rate = 0.02  # Default 2% annual risk-free rate
       
        # Data containers
        self.correlation_matrix = None
        self.covariance_matrix = None
        self.price_data = None
        self.returns_data = None
       
        # Results containers
        self.optimization_results = None
        self.summary_statistics_df = None
        self.portfolio_info_df = None
       
        # Backtesting containers
        self.backtest_results = {}
        self.rebalancing_dates = []
        self.transaction_costs = 0.001  # 0.1% default transaction cost
   
    def portfolio_stats(self, weights: np.ndarray):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        weights = np.array(weights).flatten()
        weights = weights / weights.sum()  # Normalize to ensure sum = 1
       
        # Calculate portfolio expected return
        portfolio_return = np.sum(weights * self.expected_returns.values)
       
        # Calculate portfolio volatility using covariance matrix
        try:
            portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            portfolio_vol = np.sqrt(max(portfolio_variance, 1e-10))
        except:
            portfolio_vol = np.sqrt(np.sum((weights * self.volatilities.values)**2))
       
        # Calculate Sharpe ratio
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_vol if portfolio_vol > 1e-10 else 0
       
        return portfolio_return, portfolio_vol, sharpe_ratio
   
    def calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown for a price series"""
        cumulative = price_series / price_series.iloc[0]  # Normalize to start at 1
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return max_drawdown
   
    def _regularize_correlation_matrix(self, corr_matrix: np.ndarray, min_eigenval: float = 1e-8) -> np.ndarray:
        """Regularize correlation matrix to ensure positive definiteness"""
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
       
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
       
        if np.min(eigenvals) < min_eigenval:
            eigenvals = np.maximum(eigenvals, min_eigenval)
            corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            scaling = np.sqrt(np.diag(corr_matrix))
            corr_matrix = corr_matrix / np.outer(scaling, scaling)
            np.fill_diagonal(corr_matrix, 1.0)
       
        return corr_matrix
   
    def _regularize_covariance_matrix(self, cov_matrix: np.ndarray, min_eigenval: float = 1e-8) -> np.ndarray:
        """Regularize covariance matrix to ensure positive definiteness"""
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
       
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
       
        if np.min(eigenvals) <= min_eigenval:
            eigenvals_clipped = np.maximum(eigenvals, min_eigenval)
            cov_matrix_reg = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T
            return cov_matrix_reg
       
        return cov_matrix
   
    def optimize_portfolio(self):
        """Optimize portfolio using different objectives"""
        if self.expected_returns is None:
            raise ValueError("Statistics not calculated. Run analysis first.")
       
        n_assets = len(self.asset_names)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple(zip(self.min_weights, self.max_weights))
        x0 = self.initial_weights
       
        results = {}
       
        # 1. Initial Portfolio Stats
        try:
            ret, vol, sharpe = self.portfolio_stats(self.initial_weights)
            results['initial'] = {
                'weights': self.initial_weights.copy(),
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            }
        except Exception as e:
            st.error(f"Error calculating initial portfolio: {e}")
       
        # 2. Minimum Variance Portfolio
        def min_variance_objective(weights):
            try:
                return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
            except:
                return np.sum((weights * self.volatilities.values)**2)
       
        try:
            result_min_var = minimize(
                min_variance_objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
           
            if result_min_var.success:
                ret, vol, sharpe = self.portfolio_stats(result_min_var.x)
                results['min_variance'] = {
                    'weights': result_min_var.x.copy(),
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe
                }
        except Exception as e:
            st.warning(f"Min variance optimization failed: {e}")
       
        # 3. Maximum Sharpe Ratio Portfolio
        def negative_sharpe_objective(weights):
            try:
                ret, vol, sharpe = self.portfolio_stats(weights)
                return -sharpe if vol > 0 else 1e6
            except:
                return 1e6
       
        try:
            # Try multiple starting points for better convergence
            best_sharpe_result = None
            best_sharpe_value = -np.inf
           
            starting_points = [
                x0,
                self.min_weights + 0.1 * (self.max_weights - self.min_weights),
                self.min_weights + 0.9 * (self.max_weights - self.min_weights),
            ]
           
            for start_point in starting_points:
                start_point = start_point / start_point.sum()
               
                result = minimize(
                    negative_sharpe_objective, start_point, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 2000, 'ftol': 1e-12}
                )
               
                if result.success:
                    ret, vol, sharpe = self.portfolio_stats(result.x)
                    if sharpe > best_sharpe_value:
                        best_sharpe_value = sharpe
                        best_sharpe_result = result
           
            if best_sharpe_result is not None:
                ret, vol, sharpe = self.portfolio_stats(best_sharpe_result.x)
                results['max_sharpe'] = {
                    'weights': best_sharpe_result.x.copy(),
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe
                }
        except Exception as e:
            st.warning(f"Max Sharpe optimization failed: {e}")
       
        # 4. Maximum Return Portfolio
        def negative_return_objective(weights):
            return -np.sum(weights * self.expected_returns.values)
       
        try:
            result_max_return = minimize(
                negative_return_objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000}
            )
           
            if result_max_return.success:
                ret, vol, sharpe = self.portfolio_stats(result_max_return.x)
                results['max_return'] = {
                    'weights': result_max_return.x.copy(),
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe
                }
        except Exception as e:
            st.warning(f"Max return optimization failed: {e}")
       
        self.optimization_results = results
        return results
   
    def efficient_frontier(self, n_portfolios: int = 50):
        """Generate efficient frontier (full range, including inefficient part)"""
        try:
            min_ret = np.min(self.expected_returns.values)
            max_ret = np.max(self.expected_returns.values)
            # Cover the full range from min to max
            target_returns = np.linspace(min_ret, max_ret, n_portfolios)
           
            efficient_volatilities = []
            efficient_returns = []
           
            for target_ret in target_returns:
                try:
                    constraints = [
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                        {'type': 'eq', 'fun': lambda x: np.sum(x * self.expected_returns.values) - target_ret}
                    ]
                   
                    bounds = tuple(zip(self.min_weights, self.max_weights))
                   
                    def min_variance_objective(weights):
                        return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
                   
                    result = minimize(
                        min_variance_objective, self.initial_weights,
                        method='SLSQP', bounds=bounds, constraints=constraints
                    )
                   
                    if result.success:
                        _, vol, _ = self.portfolio_stats(result.x)
                        efficient_volatilities.append(vol)
                        efficient_returns.append(target_ret)
                       
                except:
                    continue
           
            return np.array(efficient_returns), np.array(efficient_volatilities)
        except:
            return np.array([]), np.array([])
   
    # ============================================================================
    # NEW BACKTESTING FEATURES
    # ============================================================================
   
    def run_backtest(self, strategy_weights, rebalance_freq='quarterly', transaction_cost=0.001, initial_value=100000, reference_CAGR=0.08):
        """
        Run comprehensive backtesting for multiple strategies
       
        Parameters:
        - strategy_weights: dict of strategy names and their weights
        - rebalance_freq: frequency of rebalancing ('monthly', 'quarterly', 'annually', 'none')
        - transaction_cost: cost per rebalancing as decimal
        - initial_value: starting portfolio value
        - reference_CAGR: reference compound annual growth rate for comparison
        """
        if not hasattr(self, 'price_data') or self.price_data is None:
            raise ValueError("Price and returns data not available for backtesting")
       
        if not hasattr(self, 'returns_data') or self.returns_data is None:
            raise ValueError("Returns data not available for backtesting")
       
        # Get rebalancing dates
        rebalance_dates = self._get_rebalance_dates(rebalance_freq)
        self.rebalancing_dates = rebalance_dates
       
        # Run backtests for each strategy
        results = {}
        for strategy_name, weights in strategy_weights.items():
            # Run backtest for this strategy
            backtest_result = self._run_single_backtest(
                weights, rebalance_dates, transaction_cost, initial_value, strategy_name
            )
            results[strategy_name] = backtest_result
       
        # Add reference CAGR benchmark
        reference_result = self._create_reference_CAGR_benchmark(
            initial_value, reference_CAGR, "Reference CAGR"
        )
        results['reference_CAGR'] = reference_result
       
        self.backtest_results = results
        return results
   
    def _get_rebalance_dates(self, frequency):
        """Generate rebalancing dates based on frequency"""
        if frequency == 'none':
            return []
       
        if not hasattr(self, 'returns_data') or self.returns_data is None or len(self.returns_data) == 0:
            return []
       
        start_date = self.returns_data.index[0]
        end_date = self.returns_data.index[-1]
        dates = []
       
        if frequency == 'monthly':
            # First trading day of each month
            for date in pd.date_range(start_date, end_date, freq='MS'):
                # Find the first trading day on or after this date
                trading_date = self._find_next_trading_day(date)
                if trading_date is not None and trading_date <= end_date:
                    dates.append(trading_date)
       
        elif frequency == 'quarterly':
            # First trading day of each quarter
            for date in pd.date_range(start_date, end_date, freq='QS'):
                trading_date = self._find_next_trading_day(date)
                if trading_date is not None and trading_date <= end_date:
                    dates.append(trading_date)
       
        elif frequency == 'annually':
            # First trading day of each year
            for date in pd.date_range(start_date, end_date, freq='YS'):
                trading_date = self._find_next_trading_day(date)
                if trading_date is not None and trading_date <= end_date:
                    dates.append(trading_date)
       
        # Remove the first date (start with initial allocation)
        final_dates = [d for d in dates if d > start_date]
       
        return final_dates
   
    def _find_next_trading_day(self, target_date):
        """Find the next available trading day on or after target_date"""
        available_dates = self.returns_data.index
        future_dates = available_dates[available_dates >= target_date]
        return future_dates[0] if len(future_dates) > 0 else None
   
    def _run_single_backtest(self, weights, rebalance_dates, transaction_cost, initial_value, strategy_name):
        """Run backtest for a single strategy"""
       
        # Initialize tracking variables
        portfolio_values = []
        portfolio_weights_over_time = []
        transaction_costs_incurred = []
        turnover_rates = []
       
        # Starting conditions
        current_weights = np.array(weights)
        portfolio_value = initial_value
        total_transaction_costs = 0
       
        returns_data = self.returns_data[self.asset_names].fillna(0)
       
        for i, date in enumerate(returns_data.index):
            # Check if this is a rebalancing date
            is_rebalance_date = date in rebalance_dates
           
            if i == 0:
                # Initial allocation
                portfolio_values.append(portfolio_value)
                portfolio_weights_over_time.append(current_weights.copy())
                transaction_costs_incurred.append(0)
                turnover_rates.append(0)
            else:
                # Calculate returns and update portfolio value
                daily_returns = returns_data.iloc[i].values
               
                # Update portfolio value based on asset returns
                portfolio_value *= (1 + np.sum(current_weights * daily_returns))
               
                # Update weights due to price movements (drift)
                current_weights *= (1 + daily_returns)
                current_weights /= current_weights.sum()  # Renormalize
               
                # Rebalance if it's a rebalancing date
                if is_rebalance_date:
                    target_weights = np.array(weights)
                   
                    # Calculate turnover and transaction costs
                    turnover = np.sum(np.abs(target_weights - current_weights))
                    transaction_cost_dollar = portfolio_value * turnover * transaction_cost
                   
                    # Apply transaction costs
                    portfolio_value -= transaction_cost_dollar
                    total_transaction_costs += transaction_cost_dollar
                   
                    # Update weights to target
                    current_weights = target_weights.copy()
                   
                    transaction_costs_incurred.append(transaction_cost_dollar)
                    turnover_rates.append(turnover)
                else:
                    transaction_costs_incurred.append(0)
                    turnover_rates.append(0)
               
                portfolio_values.append(portfolio_value)
                portfolio_weights_over_time.append(current_weights.copy())
       
        # Create results DataFrame
        portfolio_df = pd.DataFrame({
            'date': returns_data.index,
            'portfolio_value': portfolio_values,
            'transaction_costs': transaction_costs_incurred,
            'turnover': turnover_rates
        }).set_index('date')
       
        # Add individual asset weights over time
        for j, asset_name in enumerate(self.asset_names):
            portfolio_df[f'weight_{asset_name}'] = [w[j] for w in portfolio_weights_over_time]
       
        # Calculate performance metrics
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().fillna(0)
        cumulative_returns = portfolio_df['portfolio_value'] / initial_value
       
        # Performance calculations
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / initial_value - 1) * 100
       
        # Calculate actual time period for CAGR
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        days_between = (end_date - start_date).days
        years_between = days_between / 365.25
       
        # Calculate CAGR (Compound Annual Growth Rate) - this is what we want to show
        if years_between > 0:
            cagr = ((portfolio_df['portfolio_value'].iloc[-1] / initial_value) ** (1 / years_between) - 1)
        else:
            cagr = total_return / 100  # Convert percentage to decimal
       
        # For volatility calculation, we still need annualized daily returns
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (cagr * 100 - self.risk_free_rate * 100) / volatility if volatility > 0 else 0
       
        # Drawdown calculation
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
       
        # Additional metrics
        total_transaction_costs_pct = (total_transaction_costs / initial_value) * 100
        avg_turnover = np.mean([t for t in turnover_rates if t > 0]) if any(t > 0 for t in turnover_rates) else 0
       
        # Winning periods
        positive_returns = portfolio_returns[portfolio_returns > 0]
        win_rate = len(positive_returns) / len(portfolio_returns) * 100 if len(portfolio_returns) > 0 else 0
       
        return {
            'portfolio_df': portfolio_df,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'drawdowns': drawdowns,
            'metrics': {
                'total_return': total_return,
                'annualized_return': cagr,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_transaction_costs_pct': total_transaction_costs_pct,
                'avg_turnover': avg_turnover,
                'final_value': portfolio_df['portfolio_value'].iloc[-1],
                'rebalance_count': len([d for d in rebalance_dates if d <= portfolio_df.index[-1]])
            }
        }
   
    def _create_reference_CAGR_benchmark(self, initial_value, reference_CAGR, strategy_name):
        """Create a reference CAGR benchmark for comparison"""
       
        # Get the same date range as the actual backtest
        returns_data = self.returns_data[self.asset_names].fillna(0)
        dates = returns_data.index
       
        # Calculate daily CAGR rate (convert annual to daily)
        daily_cagr = (1 + reference_CAGR) ** (1/252) - 1
       
        # Generate portfolio values based on CAGR
        portfolio_values = []
        current_value = initial_value
       
        for date in dates:
            portfolio_values.append(current_value)
            # Apply daily CAGR growth
            current_value *= (1 + daily_cagr)
       
        # Create portfolio dataframe
        portfolio_df = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'transaction_costs': [0] * len(dates),  # No transaction costs for reference
            'turnover': [0] * len(dates)  # No turnover for reference
        }, index=dates)
       
        # Calculate metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / initial_value) - 1
        annualized_return = reference_CAGR  # This is the target CAGR
        volatility = 0  # Reference CAGR has no volatility
        sharpe_ratio = float('inf') if volatility == 0 else annualized_return / volatility
        max_drawdown = 0  # Reference CAGR has no drawdowns
       
        # Calculate win rate (all periods are "wins" for CAGR)
        win_rate = 100.0
       
        # Calculate cumulative returns and drawdowns for plotting
        cumulative_returns = portfolio_df['portfolio_value'] / initial_value
        drawdowns = pd.Series([0] * len(dates), index=dates)  # No drawdowns for reference CAGR
       
        return {
            'portfolio_df': portfolio_df,
            'portfolio_returns': pd.Series([daily_cagr] * len(dates), index=dates),
            'cumulative_returns': cumulative_returns,
            'drawdowns': drawdowns,
            'metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_transaction_costs': 0,
                'total_transaction_costs_pct': 0,
                'avg_turnover': 0,
                'final_value': portfolio_df['portfolio_value'].iloc[-1],
                'rebalance_count': 0
            }
        }
   
    def calculate_strategy_comparison(self):
        """Calculate comparison metrics across all backtested strategies"""
        if not self.backtest_results:
            return None
       
        comparison_data = []
       
        for strategy_name, results in self.backtest_results.items():
            # Skip reference CAGR for strategy comparison
            if strategy_name == 'reference_CAGR':
                continue
               
            metrics = results['metrics']
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': f"{metrics['total_return']:.2f}",
                'Annualized Return (%)': f"{metrics['annualized_return']:.2f}",
                'Volatility (%)': f"{metrics['volatility']:.2f}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                'Max Drawdown (%)': f"{metrics['max_drawdown']:.2f}",
                'Win Rate (%)': f"{metrics['win_rate']:.1f}",
                'Transaction Costs (%)': f"{metrics['total_transaction_costs_pct']:.3f}",
                'Avg Turnover (%)': f"{metrics['avg_turnover']*100:.1f}",
                'Final Value ($)': f"{metrics['final_value']:,.0f}",
                'Rebalances': metrics['rebalance_count']
            })
       
        return pd.DataFrame(comparison_data)

# ============================================================================
# UTILITY FUNCTIONS FOR SAFE PLOTTING
# ============================================================================

def safe_update_axes(fig, x_format=None, y_format=None):
    """Safely update axis formatting to prevent errors"""
    try:
        if x_format:
            fig.update_xaxes(tickformat=x_format)
        if y_format:
            fig.update_yaxes(tickformat=y_format)
    except Exception as e:
        st.warning(f"Could not apply axis formatting: {e}")
    return fig

def safe_plotly_chart(fig, use_container_width=True, **kwargs):
    """Safely display plotly chart with error handling"""
    try:
        st.plotly_chart(fig, use_container_width=use_container_width, **kwargs)
    except Exception as e:
        st.error(f"Error displaying chart: {e}")
        st.write("Chart data available but display failed.")

# ============================================================================
# STREAMLIT APP INTERFACE
# ============================================================================

# Initialize session state
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = EnhancedPortfolioOptimizer()
    st.session_state.analysis_complete = False
    st.session_state.results = None
    st.session_state.backtest_complete = False
   
# Initialize backtesting parameters with default values
if 'rebalance_freq' not in st.session_state:
    st.session_state.rebalance_freq = 'quarterly'
if 'transaction_cost' not in st.session_state:
    st.session_state.transaction_cost = 0.001
if 'initial_value' not in st.session_state:
    st.session_state.initial_value = 100000
if 'reference_CAGR' not in st.session_state:
    st.session_state.reference_CAGR = 0.08

# Inject custom CSS for sidebar background color
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #162d73 !important;
    }
    [data-testid="stSidebar"] * {
        color: #fff !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] .st-emotion-cache-1v0mbdj {
        font-weight: bold !important;
        color: #fff !important;
    }
    /* Style radio button circles and slider bars as grey */
    [data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
        accent-color: #b0b0b0 !important;
    }
    [data-testid="stSidebar"] input[type="checkbox"] {
        accent-color: #b0b0b0 !important;
    }
    [data-testid="stSidebar"] .stSlider [role="slider"] {
        background: #b0b0b0 !important;
        border-color: #b0b0b0 !important;
    }
    [data-testid="stSidebar"] .stSlider .css-1gv0vcd [role="slider"] {
        background: #b0b0b0 !important;
        border-color: #b0b0b0 !important;
    }
    /* Remove previous slider bar background overrides */
    /* No background or label color changes for radio/slider labels */
    /* Primary button color override (robust for all Streamlit versions) */
    .stButton > button {
        background-color: #162d73 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #203a8c !important;
        color: #fff !important;
    }
    /* Add spacing between Streamlit tab labels */
    .stTabs [data-baseweb="tab-list"] button {
        margin-right: 2.5rem !important;
    }
    /* Make help icons white */
    [data-testid="stSidebar"] .stTooltipIcon {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stTooltipIcon svg {
        fill: #ffffff !important;
    }
    /* Ensure help text is white */
    [data-testid="stSidebar"] .stTooltipIcon + div {
        color: #ffffff !important;
    }
    /* Fix input field text color - make text black for readability */
    [data-testid="stSidebar"] input[type="text"],
    [data-testid="stSidebar"] input[type="number"],
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stTextArea textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    /* Fix selectbox text color - make text black for readability */
    [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] .stSelectbox option,
    [data-testid="stSidebar"] [data-baseweb="select"],
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] button,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [role="listbox"],
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [role="option"] {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
   
    /* Remove white background and shadow from expanders in sidebar */
    [data-testid="stSidebar"] .stExpander,
    [data-testid="stSidebar"] .stExpander > div,
    [data-testid="stSidebar"] .stExpander > div > div,
    [data-testid="stSidebar"] .stExpander > div > div > div,
    [data-testid="stSidebar"] .stExpander > div > div > div > div {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        background: none !important;
    }
   
    /* Target specific Streamlit expander classes */
    [data-testid="stSidebar"] [data-testid="stExpander"],
    [data-testid="stSidebar"] .stExpander,
    [data-testid="stSidebar"] .st-emotion-cache-1r6slb0,
    [data-testid="stSidebar"] .st-emotion-cache-1r6slb0 > div,
    [data-testid="stSidebar"] .st-emotion-cache-1r6slb0 > div > div {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        background: none !important;
    }
   
    /* Remove any remaining white backgrounds */
    [data-testid="stSidebar"] *[class*="stExpander"],
    [data-testid="stSidebar"] *[class*="st-emotion-cache"] {
        background-color: transparent !important;
        background: none !important;
    }
    /* Additional selectbox styling for dropdown options */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [role="listbox"] [role="option"] {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [role="listbox"] [role="option"]:hover {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
    /* Force black text on selectbox input elements only, not labels */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] button,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [role="listbox"] [role="option"] {
        color: #000000 !important;
    }
    /* Override any white text specifically */
    [data-testid="stSidebar"] .stSelectbox [style*="color: white"],
    [data-testid="stSidebar"] .stSelectbox [style*="color: #fff"],
    [data-testid="stSidebar"] .stSelectbox [style*="color: #ffffff"] {
        color: #000000 !important;
    }
    /* Ensure placeholder text is also visible */
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #666666 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Vantage Capital Header
from PIL import Image
import io

def simulate_ticker_search(query):
    """
    Search for tickers using yfinance library with real API calls.
    """
    import yfinance as yf
    import pandas as pd
   
    results = {'STOCK': [], 'ETF': [], 'MUTUAL FUND': []}
   
    try:
        # Use yfinance's search functionality
        search_results = yf.Tickers(query)
       
        # Get info for each found ticker
        for ticker_symbol in search_results.tickers:
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
               
                if info and 'longName' in info and info['longName']:
                    name = info['longName']
                   
                    # Determine category based on ticker info
                    quote_type = info.get('quoteType', '').lower()
                    if 'etf' in quote_type:
                        category = 'ETF'
                    elif 'mutualfund' in quote_type or 'mutual fund' in quote_type:
                        category = 'MUTUAL FUND'
                    else:
                        category = 'STOCK'
                   
                    results[category].append((ticker_symbol, name))
                   
                    # Limit results per category
                    if len(results[category]) >= 5:
                        break
                       
            except Exception as e:
                continue
               
    except Exception as e:
        # If search fails, try direct ticker lookup
        try:
            ticker = yf.Ticker(query)
            info = ticker.info
           
            if info and 'longName' in info and info['longName']:
                name = info['longName']
                quote_type = info.get('quoteType', '').lower()
               
                if 'etf' in quote_type:
                    category = 'ETF'
                elif 'mutualfund' in quote_type or 'mutual fund' in quote_type:
                    category = 'MUTUAL FUND'
                else:
                    category = 'STOCK'
               
                results[category].append((query, name))
               
        except Exception as e2:
            pass
   
    return results

def search_tickers_with_yfinance(query):
    """
    Search for tickers using yfinance library with real API calls.
    """
    import yfinance as yf
    import pandas as pd
   
    results = {'STOCK': [], 'ETF': [], 'MUTUAL FUND': []}
   
    try:
        # Use yfinance's search functionality
        search_results = yf.Tickers(query)
       
        # Get info for each found ticker
        for ticker_symbol in search_results.tickers:
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
               
                if info and 'longName' in info and info['longName']:
                    name = info['longName']
                   
                    # Determine category based on ticker info
                    quote_type = info.get('quoteType', '').lower()
                    if 'etf' in quote_type:
                        category = 'ETF'
                    elif 'mutualfund' in quote_type or 'mutual fund' in quote_type:
                        category = 'MUTUAL FUND'
                    else:
                        category = 'STOCK'
                   
                    results[category].append((ticker_symbol, name))
                   
                    # Limit results per category
                    if len(results[category]) >= 5:
                        break
                       
            except Exception as e:
                continue
               
    except Exception as e:
        # If search fails, try direct ticker lookup
        try:
            ticker = yf.Ticker(query)
            info = ticker.info
           
            if info and 'longName' in info and info['longName']:
                name = info['longName']
                quote_type = info.get('quoteType', '').lower()
               
                if 'etf' in quote_type:
                    category = 'ETF'
                elif 'mutualfund' in quote_type or 'mutual fund' in quote_type:
                    category = 'MUTUAL FUND'
                else:
                    category = 'STOCK'
               
                results[category].append((query, name))
               
        except Exception as e2:
            pass
   
    return results

def search_tickers_with_alphavantage(query):
    """
    Search for tickers using Alpha Vantage API.
    Requires API key: https://www.alphavantage.co/support/#api-key
    """
    import requests
   
    # You would need to get an API key from Alpha Vantage
    API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
   
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey={API_KEY}"
   
    try:
        response = requests.get(url)
        data = response.json()
       
        results = {'STOCK': [], 'ETF': [], 'MUTUAL FUND': []}
       
        if 'bestMatches' in data:
            for match in data['bestMatches'][:10]:  # Limit to 10 results
                symbol = match['1. symbol']
                name = match['2. name']
                type_ = match['3. type']
                region = match['4. region']
               
                # Categorize based on type
                if 'ETF' in type_:
                    category = 'ETF'
                elif 'Mutual Fund' in type_:
                    category = 'MUTUAL FUND'
                else:
                    category = 'STOCK'
               
                results[category].append((symbol, name))
       
        return results
       
    except Exception as e:
        st.error(f"Alpha Vantage API error: {e}")
        return {'STOCK': [], 'ETF': [], 'MUTUAL FUND': []}

def search_tickers_with_iex(query):
    """
    Search for tickers using IEX Cloud API.
    Requires API key: https://iexcloud.io/cloud-login#/register
    """
    import requests
   
    # You would need to get an API key from IEX Cloud
    API_KEY = "YOUR_IEX_API_KEY"
   
    url = f"https://cloud.iexapis.com/stable/search/{query}?token={API_KEY}"
   
    try:
        response = requests.get(url)
        data = response.json()
       
        results = {'STOCK': [], 'ETF': [], 'MUTUAL FUND': []}
       
        for item in data[:10]:  # Limit to 10 results
            symbol = item['symbol']
            name = item['securityName']
            type_ = item.get('securityType', '')
           
            # Categorize based on type
            if 'ETF' in type_:
                category = 'ETF'
            elif 'Mutual Fund' in type_:
                category = 'MUTUAL FUND'
            else:
                category = 'STOCK'
           
            results[category].append((symbol, name))
       
        return results
       
    except Exception as e:
        st.error(f"IEX Cloud API error: {e}")
        return {'STOCK': [], 'ETF': [], 'MUTUAL FUND': []}

# Load and encode the image as base64
with open("Vantage_Capital_Logo.png", "rb") as img_file:
    img_bytes = img_file.read()
    encoded = base64.b64encode(img_bytes).decode()

st.markdown(f"""
    <div style='width:100%; display:flex; flex-direction:column; justify-content:center; align-items:center; margin-bottom:2rem;'>
        <img src='data:image/png;base64,{encoded}' width='400' style='display:block;'/>
        <div style='margin-top: 1.2rem; color: #6c757d; font-size: 1.5rem; text-align: center;'>
            Proprietary Portfolio Optimization Platform
        </div>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("### 🛡️ Portfolio Configuration")
   
    # Available tickers with their vehicle names and sectors (moved to top for access by Forward-Looking Parameters)
    available_tickers = {
        'SPY': ('S&P 500 ETF', 'US Equity'),
        'QQQ': ('NASDAQ 100 ETF', 'Technology'),
        'EFA': ('International Developed Markets ETF', 'International'),
        'GLD': ('Gold ETF', 'Commodities'),
        'BND': ('Total Bond Market ETF', 'Fixed Income'),
        'VTI': ('Total Stock Market ETF', 'US Equity'),
        'VNQ': ('Real Estate ETF', 'Real Estate'),
        'VUG': ('Growth ETF', 'US Equity'),
        'EEM': ('Emerging Markets ETF', 'International'),
        'TLT': ('Long Treasury ETF', 'Fixed Income'),
        'IWM': ('Small Cap ETF', 'US Equity'),
        'VYM': ('High Dividend ETF', 'US Equity'),
        'SCHD': ('Dividend ETF', 'US Equity'),
        'VIG': ('Dividend Growth ETF', 'US Equity'),
        'XLK': ('Technology ETF', 'Technology'),
        'XLF': ('Financial ETF', 'Financials'),
        'XLE': ('Energy ETF', 'Energy'),
        'AAPL': ('Apple Inc.', 'Technology'),
        'MSFT': ('Microsoft Corp.', 'Technology'),
        'ARKK': ('Innovation ETF', 'Technology'),
        'VGT': ('Technology ETF', 'Technology'),
        'VOO': ('S&P 500 ETF (Vanguard)', 'US Equity'),
        'IVV': ('S&P 500 ETF (iShares)', 'US Equity'),
        'AGG': ('US Aggregate Bond ETF', 'Fixed Income'),
        'BNDX': ('International Bond ETF', 'Fixed Income'),
        'VEA': ('Developed Markets ETF', 'International'),
        'VWO': ('Emerging Markets ETF', 'International'),
        'VTV': ('Value ETF', 'US Equity'),
        'VBR': ('Small Cap Value ETF', 'US Equity'),
        'VXUS': ('Total International Stock ETF', 'International'),
        'XLV': ('Healthcare ETF', 'Healthcare'),
        'XLI': ('Industrial ETF', 'Industrials'),
        'XLP': ('Consumer Staples ETF', 'Consumer Staples'),
        'XLY': ('Consumer Discretionary ETF', 'Consumer Discretionary'),
        'XLU': ('Utilities ETF', 'Utilities'),
        'XLB': ('Materials ETF', 'Materials'),
        'XLC': ('Communication Services ETF', 'Communication Services'),
        'XLRE': ('Real Estate ETF', 'Real Estate'),
        'XBI': ('Biotechnology ETF', 'Healthcare'),
        'SOXX': ('Semiconductor ETF', 'Technology'),
        'SMH': ('Semiconductor ETF', 'Technology'),
        'JETS': ('Airlines ETF', 'Industrials'),
        'KRE': ('Regional Banking ETF', 'Financials'),
        'GDX': ('Gold Miners ETF', 'Materials'),
        'SLV': ('Silver ETF', 'Commodities'),
        'USO': ('US Oil Fund', 'Energy'),
        'TAN': ('Solar ETF', 'Energy'),
        'ICLN': ('Clean Energy ETF', 'Energy')
    }
   
    # Mode Selection
    st.subheader("1. Analysis Mode")
    mode_choice = st.radio(
        "Select approach:",
        ["Backward-Looking", "Forward-Looking"],
        help="Backward: Historical data analysis | Forward: Custom parameter inputs"
    )
   
    st.session_state.optimizer.mode = 'backward_looking' if mode_choice == "Backward-Looking" else 'forward_looking'
   
    # Forward-Looking Parameters (only show for forward-looking mode)
    if st.session_state.optimizer.mode == 'forward_looking':
        st.subheader("2. Forward-Looking Parameters")
       
        st.markdown("📈 **Custom Expected Returns & Volatilities**")
        st.write("Define annual expected returns and volatilities for each asset:")
       
        # Note: This will be populated after asset selection
        if 'selected_tickers_list' in st.session_state and len(st.session_state.selected_tickers_list) > 0:
            expected_returns = []
            volatilities = []
           
            for ticker in st.session_state.selected_tickers_list:
                # Get asset name from available_tickers
                asset_name = available_tickers.get(ticker, (ticker, 'Unknown'))[0]
               
                col1, col2 = st.columns(2)
               
                with col1:
                    # Default return based on asset type
                    if ticker in ['SPY', 'VTI']:
                        default_ret = 8.0
                    elif ticker == 'QQQ':
                        default_ret = 10.0
                    elif ticker in ['BND', 'TLT']:
                        default_ret = 3.0
                    elif ticker == 'GLD':
                        default_ret = 5.0
                    else:
                        default_ret = 7.0
                   
                    ret = st.number_input(
                        f"Return % - {asset_name}:",
                        min_value=-20.0,
                        max_value=50.0,
                        value=default_ret,
                        step=0.5,
                        key=f"ret_{ticker}"
                    )
                    expected_returns.append(ret / 100)
               
                with col2:
                    # Default volatility based on asset type
                    if ticker in ['SPY', 'VTI']:
                        default_vol = 15.0
                    elif ticker == 'QQQ':
                        default_vol = 20.0
                    elif ticker == 'BND':
                        default_vol = 8.0
                    elif ticker == 'GLD':
                        default_vol = 18.0
                    else:
                        default_vol = 16.0
                   
                    vol = st.number_input(
                        f"Vol % - {asset_name}:",
                        min_value=1.0,
                        max_value=100.0,
                        value=default_vol,
                        step=0.5,
                        key=f"vol_{ticker}"
                    )
                    volatilities.append(vol / 100)
           
            # Store forward-looking parameters
            if len(expected_returns) > 0:
                st.session_state.expected_returns = expected_returns
                st.session_state.volatilities = volatilities
                st.success("✅ Forward-looking parameters configured!")
        else:
            st.info("Please select assets in section 3 first to configure forward-looking parameters.")
            # Note: Forward-Looking Parameters section is integrated into section 1 when Forward-Looking mode is selected
   
    # Risk-free rate
    st.subheader("2. Risk Parameters")
    risk_free_rate = st.slider("Risk-Free Rate (%):", 0.0, 10.0, 2.0, 0.1)
    st.session_state.optimizer.risk_free_rate = risk_free_rate / 100
   
    # Years of data
    if st.session_state.optimizer.mode == 'backward_looking':
        years = st.slider("Years of Historical Data:", 1, 10, 5)
        st.info("Using historical market data for all calculations")
    else:
        years = st.slider("Years for Correlation Data:", 1, 10, 3)
        st.info("Using custom parameters + historical correlations")
   
    st.session_state.optimizer.lookback_years = years
   
    # Asset Selection
    st.subheader("3. Asset Selection")
   
    if st.session_state.optimizer.mode == 'forward_looking':
        st.info("💡 **Forward-Looking Mode:** After selecting assets, return to section 1 to configure expected returns and volatilities.")
   
    # Group tickers by sector
    sector_tickers = {}
    for ticker, (name, sector) in available_tickers.items():
        if sector not in sector_tickers:
            sector_tickers[sector] = []
        sector_tickers[sector].append((ticker, name))
   
    # Searchable ticker input with autocomplete
    st.write("**Search and Select Tickers:**")
   
    # Initialize session state for selected tickers
    if 'selected_tickers_list' not in st.session_state:
        st.session_state.selected_tickers_list = []
   
    # Initialize session state for search query
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
   
    # Search input
    search_query = st.text_input(
        "Search for tickers:",
        placeholder="Type ticker symbol or company name (e.g., TSLA, Apple, SPY)",
        help="Type a ticker or company name and press Enter to search for results"
    )
   
    # Update session state with current search query
    st.session_state.search_query = search_query
   
    # Search results container
    if search_query:
        with st.container():
            st.write("**Search Results:**")
           
            # Simulate search results (in a real app, this would call an API)
            search_results = simulate_ticker_search(search_query)
           
            if search_results:
                for category, items in search_results.items():
                    if items:  # Only show categories with results
                        st.write(f"**{category}:**")
                        for ticker, name in items:
                            # Create a button for each result
                            if st.button(f"➕ {ticker} - {name}", key=f"add_{ticker}", use_container_width=True):
                                if ticker not in [t[0] for t in st.session_state.selected_tickers_list]:
                                    st.session_state.selected_tickers_list.append((ticker, name, category))
                                    st.rerun()
                        st.write("")
            else:
                st.write("No results found. Try a different search term.")
   
    # Show selected tickers
    if st.session_state.selected_tickers_list:
        st.write("**Selected Tickers:**")
        for i, (ticker, name, category) in enumerate(st.session_state.selected_tickers_list):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {ticker} - {name} ({category})")
            with col2:
                if st.button(f"❌", key=f"remove_{i}"):
                    st.session_state.selected_tickers_list.pop(i)
                    st.rerun()
   
    # Extract ticker symbols for processing
    tickers = [ticker for ticker, name, category in st.session_state.selected_tickers_list]
   
    # If no tickers selected, use defaults
    if not tickers:
        tickers = ["SPY", "QQQ", "EFA", "GLD", "BND"]
        st.info("Using default tickers. Search above to add your own selections.")
   
    # Asset name mappings (for backward compatibility)
    asset_names_map = {ticker: name for ticker, name, category in st.session_state.selected_tickers_list}
    # Add defaults for backward compatibility
    for ticker, (name, sector) in available_tickers.items():
        if ticker not in asset_names_map:
            asset_names_map[ticker] = name
   
    # Asset name mappings (for backward compatibility)
    asset_names_map = {ticker: name for ticker, (name, sector) in available_tickers.items()}
   
    # ETF Reference Section
    with st.expander("📊 Browse ETF Reference by Category", expanded=False):
        st.write("**Browse and select from curated ETF categories:**")
       
        # Add CSS to improve tab layout
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            overflow-x: auto;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: nowrap;
            min-width: 80px;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
       
        # ETF categories with real ETF tickers
        etf_categories = {
            'Commodities': [
                ('GLD', 'SPDR Gold Shares', 'Commodities'),
                ('SLV', 'iShares Silver Trust', 'Commodities'),
                ('USO', 'United States Oil Fund LP', 'Commodities'),
                ('DJP', 'iPath Bloomberg Commodity Index Total Return ETN', 'Commodities'),
                ('GSG', 'iShares S&P GSCI Commodity-Indexed Trust', 'Commodities'),
                ('DBC', 'Invesco DB Commodity Index Tracking Fund', 'Commodities'),
                ('PDBC', 'Invesco Optimum Yield Diversified Commodity Strategy No K-1 ETF', 'Commodities'),
                ('COMT', 'iShares GSCI Commodity Dynamic Roll Strategy ETF', 'Commodities')
            ],
            'Currency Hedged': [
                ('HEFA', 'iShares Currency Hedged MSCI EAFE ETF', 'Currency Hedged'),
                ('HEDJ', 'WisdomTree Europe Hedged Equity Fund', 'Currency Hedged'),
                ('DBEF', 'Xtrackers MSCI EAFE Hedged Equity ETF', 'Currency Hedged'),
                ('IHDG', 'WisdomTree International Hedged Quality Dividend Growth Fund', 'Currency Hedged')
            ],
            'Dividend & Income': [
                ('VIG', 'Vanguard Dividend Appreciation ETF', 'Dividend & Income'),
                ('SCHD', 'Schwab U.S. Dividend Equity ETF', 'Dividend & Income'),
                ('DVY', 'iShares Select Dividend ETF', 'Dividend & Income'),
                ('SDY', 'SPDR S&P Dividend ETF', 'Dividend & Income'),
                ('NOBL', 'ProShares S&P 500 Dividend Aristocrats ETF', 'Dividend & Income'),
                ('FVD', 'First Trust Value Line Dividend Index Fund', 'Dividend & Income'),
                ('HDV', 'iShares High Dividend ETF', 'Dividend & Income'),
                ('DGRO', 'iShares Core Dividend Growth ETF', 'Dividend & Income')
            ],
            'Emerging Markets Debt': [
                ('EMB', 'iShares J.P. Morgan USD Emerging Markets Bond ETF', 'Emerging Markets Debt'),
                ('PCY', 'Invesco Emerging Markets Sovereign Debt ETF', 'Emerging Markets Debt'),
                ('LEMB', 'iShares J.P. Morgan EM Local Currency Bond ETF', 'Emerging Markets Debt'),
                ('EMLC', 'VanEck Vectors J.P. Morgan EM Local Currency Bond ETF', 'Emerging Markets Debt'),
                ('VWOB', 'Vanguard Emerging Markets Government Bond ETF', 'Emerging Markets Debt'),
                ('EMHY', 'iShares Emerging Markets High Yield Bond ETF', 'Emerging Markets Debt'),
                ('EMCB', 'WisdomTree Emerging Markets Corporate Bond Fund', 'Emerging Markets Debt'),
                ('EMAG', 'VanEck Vectors Emerging Markets Aggregate Bond ETF', 'Emerging Markets Debt')
            ],
            'Emerging Markets Equity': [
                ('EEM', 'iShares MSCI Emerging Markets ETF', 'Emerging Markets Equity'),
                ('VWO', 'Vanguard FTSE Emerging Markets ETF', 'Emerging Markets Equity'),
                ('IEMG', 'iShares Core MSCI Emerging Markets ETF', 'Emerging Markets Equity'),
                ('SCHE', 'Schwab Emerging Markets Equity ETF', 'Emerging Markets Equity'),
                ('FEM', 'First Trust Emerging Markets AlphaDEX Fund', 'Emerging Markets Equity'),
                ('DEM', 'WisdomTree Emerging Markets High Dividend Fund', 'Emerging Markets Equity'),
                ('DGRE', 'WisdomTree Emerging Markets Quality Dividend Growth Fund', 'Emerging Markets Equity'),
                ('FNDE', 'Schwab Fundamental Emerging Markets Large Company Index ETF', 'Emerging Markets Equity')
            ],
            'ESG Sustainable': [
                ('ESGU', 'iShares ESG Aware MSCI USA ETF', 'ESG Sustainable'),
                ('SUSA', 'iShares MSCI USA ESG Select ETF', 'ESG Sustainable'),
                ('VFTAX', 'Vanguard FTSE Social Index Fund', 'ESG Sustainable'),
                ('DSI', 'iShares MSCI KLD 400 Social ETF', 'ESG Sustainable'),
                ('SPYG', 'SPDR Portfolio S&P 500 Growth ETF', 'ESG Sustainable'),
                ('EAGG', 'iShares ESG Aware U.S. Aggregate Bond ETF', 'ESG Sustainable'),
                ('CRBN', 'iShares MSCI ACWI Low Carbon Target ETF', 'ESG Sustainable'),
                ('JUST', 'Goldman Sachs JUST U.S. Large Cap Equity ETF', 'ESG Sustainable')
            ],
            'Factor-Based': [
                ('MTUM', 'iShares MSCI USA Momentum Factor ETF', 'Factor-Based'),
                ('QUAL', 'iShares MSCI USA Quality Factor ETF', 'Factor-Based'),
                ('VLUE', 'iShares MSCI USA Value Factor ETF', 'Factor-Based'),
                ('USMV', 'iShares MSCI USA Min Vol Factor ETF', 'Factor-Based'),
                ('SIZE', 'iShares MSCI USA Size Factor ETF', 'Factor-Based')
            ],
            'Global Equity': [
                ('VT', 'Vanguard Total World Stock ETF', 'Global Equity'),
                ('ACWI', 'iShares MSCI ACWI ETF', 'Global Equity'),
                ('VTIAX', 'Vanguard Total International Stock Index Fund', 'Global Equity'),
                ('URTH', 'iShares MSCI World ETF', 'Global Equity'),
                ('IMI', 'iShares Core MSCI Total International Stock ETF', 'Global Equity'),
                ('SPDW', 'SPDR Portfolio Developed World ex-US ETF', 'Global Equity'),
                ('CWI', 'SPDR MSCI ACWI ex-US ETF', 'Global Equity')
            ],
            'Inflation-Protected Bonds': [
                ('TIP', 'iShares TIPS Bond ETF', 'Inflation-Protected Bonds'),
                ('VTIP', 'Vanguard Short-Term Inflation-Protected Securities ETF', 'Inflation-Protected Bonds'),
                ('SCHP', 'Schwab U.S. TIPS ETF', 'Inflation-Protected Bonds'),
                ('IPE', 'SPDR Bloomberg Barclays 1-10 Year TIPS ETF', 'Inflation-Protected Bonds'),
                ('LTPZ', 'PIMCO 15+ Year U.S. TIPS Index ETF', 'Inflation-Protected Bonds'),
                ('STIP', 'iShares 0-5 Year TIPS Bond ETF', 'Inflation-Protected Bonds'),
                ('SPIP', 'SPDR Portfolio TIPS ETF', 'Inflation-Protected Bonds')
            ],
            'International Developed': [
                ('EFA', 'iShares MSCI EAFE ETF', 'International Developed'),
                ('VEA', 'Vanguard FTSE Developed Markets ETF', 'International Developed'),
                ('IEFA', 'iShares Core MSCI EAFE ETF', 'International Developed'),
                ('SCHF', 'Schwab International Equity ETF', 'International Developed'),
                ('VXUS', 'Vanguard Total International Stock ETF', 'International Developed'),
                ('IXUS', 'iShares Core MSCI Total International Stock ETF', 'International Developed'),
                ('VEU', 'Vanguard FTSE All-World ex-US ETF', 'International Developed'),
                ('ACWX', 'iShares MSCI ACWI ex U.S. ETF', 'International Developed')
            ],
            'Managed Futures': [
                ('DBMF', 'iMGP DBi Managed Futures Strategy ETF', 'Managed Futures'),
                ('KMLM', 'KFA Mount Lucas Index Strategy ETF', 'Managed Futures'),
                ('CTA', 'Simplify Managed Futures Strategy ETF', 'Managed Futures'),
                ('PFIQ', 'Pacer Trendpilot 100 ETF', 'Managed Futures'),
                ('BTAL', 'AGFiQ U.S. Market Neutral Anti-Beta Fund', 'Managed Futures'),
                ('AMFAX', 'AQR Managed Futures Strategy Fund', 'Managed Futures')
            ],
            'Preferred Stocks': [
                ('PFF', 'iShares Preferred and Income Securities ETF', 'Preferred Stocks'),
                ('PGX', 'Invesco Preferred ETF', 'Preferred Stocks'),
                ('PREF', 'Principal Spectrum Preferred Securities Active ETF', 'Preferred Stocks'),
                ('PSK', 'SPDR ICE Preferred Securities ETF', 'Preferred Stocks'),
                ('FPE', 'First Trust Preferred Securities and Income ETF', 'Preferred Stocks'),
                ('VRP', 'Invesco Variable Rate Preferred ETF', 'Preferred Stocks'),
                ('PFXF', 'VanEck Vectors Preferred Securities ex Financials ETF', 'Preferred Stocks'),
                ('JPS', 'Nuveen Preferred & Income Opportunities Fund', 'Preferred Stocks')
            ],
            'Real Estate': [
                ('VNQ', 'Vanguard Real Estate ETF', 'Real Estate'),
                ('IYR', 'iShares U.S. Real Estate ETF', 'Real Estate'),
                ('SCHH', 'Schwab U.S. REIT ETF', 'Real Estate'),
                ('RWR', 'SPDR Dow Jones REIT ETF', 'Real Estate'),
                ('FREL', 'Fidelity MSCI Real Estate Index ETF', 'Real Estate'),
                ('USRT', 'iShares Core U.S. REIT ETF', 'Real Estate'),
                ('O', 'Realty Income Corporation', 'Real Estate'),
                ('AMT', 'American Tower Corporation', 'Real Estate')
            ],
            'Thematic Innovation': [
                ('ARKG', 'ARK Genomic Revolution ETF', 'Thematic Innovation'),
                ('BOTZ', 'Global X Robotics & Artificial Intelligence ETF', 'Thematic Innovation'),
                ('ROBO', 'Robo Global Robotics and Automation ETF', 'Thematic Innovation'),
                ('AIEQ', 'AI Powered Equity ETF', 'Thematic Innovation'),
                ('KOMP', 'SPDR S&P Kensho New Economies Composite ETF', 'Thematic Innovation'),
                ('DRIV', 'Global X Autonomous & Electric Vehicles ETF', 'Thematic Innovation'),
                ('THNQ', 'ROBO Global Artificial Intelligence ETF', 'Thematic Innovation'),
                ('FIVG', 'Defiance Next Gen Connectivity ETF', 'Thematic Innovation')
            ],
            'US High Yield': [
                ('HYG', 'iShares iBoxx $ High Yield Corporate Bond ETF', 'US High Yield'),
                ('JNK', 'SPDR Bloomberg High Yield Bond ETF', 'US High Yield'),
                ('VWEHX', 'Vanguard High-Yield Corporate Fund', 'US High Yield'),
                ('HYEM', 'VanEck Vectors Emerging Markets High Yield Bond ETF', 'US High Yield'),
                ('ANGL', 'VanEck Vectors Fallen Angel High Yield Bond ETF', 'US High Yield'),
                ('PHB', 'Invesco Fundamental High Yield Corporate Bond ETF', 'US High Yield'),
                ('HYLD', 'High Yield ETF', 'US High Yield'),
                ('HYLS', 'First Trust Tactical High Yield ETF', 'US High Yield')
            ],
            'US Investment Grade': [
                ('AGG', 'iShares Core U.S. Aggregate Bond ETF', 'US Investment Grade'),
                ('BND', 'Vanguard Total Bond Market ETF', 'US Investment Grade'),
                ('SCHZ', 'Schwab U.S. Aggregate Bond ETF', 'US Investment Grade'),
                ('VCIT', 'Vanguard Intermediate-Term Corporate Bond ETF', 'US Investment Grade'),
                ('LQD', 'iShares iBoxx $ Investment Grade Corporate Bond ETF', 'US Investment Grade'),
                ('VCSH', 'Vanguard Short-Term Corporate Bond ETF', 'US Investment Grade'),
                ('IGIB', 'iShares Intermediate-Term Corporate Bond ETF', 'US Investment Grade'),
                ('VIGI', 'Vanguard International Dividend Appreciation ETF', 'US Investment Grade')
            ],
            'US Large Cap': [
                ('SPY', 'SPDR S&P 500 ETF Trust', 'US Large Cap'),
                ('VOO', 'Vanguard S&P 500 ETF', 'US Large Cap'),
                ('IVV', 'iShares Core S&P 500 ETF', 'US Large Cap'),
                ('QQQ', 'Invesco QQQ Trust', 'US Large Cap'),
                ('VTI', 'Vanguard Total Stock Market ETF', 'US Large Cap'),
                ('ITOT', 'iShares Core S&P Total U.S. Stock Market ETF', 'US Large Cap'),
                ('SCHB', 'Schwab U.S. Broad Market ETF', 'US Large Cap'),
                ('DIA', 'SPDR Dow Jones Industrial Average ETF', 'US Large Cap')
            ],
            'US Small & Mid Cap': [
                ('IWM', 'iShares Russell 2000 ETF', 'US Small & Mid Cap'),
                ('IJH', 'iShares Core S&P Mid-Cap ETF', 'US Small & Mid Cap'),
                ('VB', 'Vanguard Small-Cap ETF', 'US Small & Mid Cap'),
                ('VO', 'Vanguard Mid-Cap ETF', 'US Small & Mid Cap'),
                ('SCHA', 'Schwab U.S. Small-Cap ETF', 'US Small & Mid Cap'),
                ('VIOO', 'Vanguard S&P Small-Cap 600 ETF', 'US Small & Mid Cap'),
                ('IJR', 'iShares Core S&P Small-Cap ETF', 'US Small & Mid Cap'),
                ('RWJ', 'Invesco S&P SmallCap 600 Revenue ETF', 'US Small & Mid Cap')
            ],
            'US Tech': [
                ('XLK', 'Technology Select Sector SPDR Fund', 'US Tech'),
                ('VGT', 'Vanguard Information Technology ETF', 'US Tech'),
                ('SMH', 'VanEck Vectors Semiconductor ETF', 'US Tech'),
                ('SOXX', 'iShares PHLX Semiconductor ETF', 'US Tech'),
                ('ARKK', 'ARK Innovation ETF', 'US Tech'),
                ('ARKW', 'ARK Next Generation Internet ETF', 'US Tech'),
                ('IGV', 'iShares Expanded Tech-Software Sector ETF', 'US Tech'),
                ('HACK', 'ETFMG Prime Cyber Security ETF', 'US Tech')
            ]
        }
       
        # Use dropdown instead of tabs
        category_options = list(etf_categories.keys())
        selected_category = st.selectbox(
            "Select ETF Category:",
            options=category_options,
            index=0,
            help="Choose an ETF category to browse available options"
        )
       
        # Display ETFs for selected category
        if selected_category in etf_categories:
            st.write(f"**{selected_category} ETFs:**")
            etfs = etf_categories[selected_category]
           
            for ticker, name, etf_category in etfs:
                if st.button(f"➕ {ticker} - {name}", key=f"etf_{selected_category}_{ticker}", use_container_width=True):
                    if ticker not in [t[0] for t in st.session_state.selected_tickers_list]:
                        st.session_state.selected_tickers_list.append((ticker, name, etf_category))
                        st.rerun()
   
    # Generate asset names
    asset_names = [asset_names_map.get(ticker, f'{ticker} Asset') for ticker in tickers]
   
    # Store in optimizer
    st.session_state.optimizer.tickers = tickers
    st.session_state.optimizer.asset_names = asset_names
   
    # Weight Constraints
    st.subheader("5. Weight Constraints")
    constraint_type = st.radio(
        "Constraint Type:",
        ["Default (0-40%)", "Custom", "Equal Weight Only"]
    )
   
    # Custom Weight Constraints (only show when Custom is selected)
    if constraint_type == "Custom":
        with st.expander("Custom Weight Constraints", expanded=True):
            min_weights = []
            max_weights = []
           
            for name in asset_names:
                col1, col2 = st.columns(2)
                with col1:
                    min_w = st.number_input(f"Min % - {name}:", 0.0, 50.0, 0.0, 1.0, key=f"min_{name}")
                    min_weights.append(min_w / 100)
                with col2:
                    max_w = st.number_input(f"Max % - {name}:", min_w, 100.0, 40.0, 1.0, key=f"max_{name}")
                    max_weights.append(max_w / 100)
           
            min_weights = np.array(min_weights)
            max_weights = np.array(max_weights)
   
    # Backtesting Configuration
    st.subheader("4. Backtesting Settings")
   
    rebalance_freq = st.selectbox(
        "Rebalancing Frequency:",
        ["Quarterly", "Monthly", "Annually", "None"],
        help="How often to rebalance the portfolio"
    )
   
    # Convert to lowercase for backtesting method
    rebalance_freq_lower = rebalance_freq.lower()
   
    # Store in session state for access in backtesting
    st.session_state.rebalance_freq = rebalance_freq_lower
   
    transaction_cost = st.slider(
        "Transaction Cost (%):",
        0.0, 1.0, 0.0, 0.05,
        help="Cost per rebalancing as percentage of portfolio value"
    ) / 100
   
    # Store in session state for access in backtesting
    st.session_state.transaction_cost = transaction_cost
   
    initial_value = st.number_input(
        "Initial Portfolio Value ($):",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000,
        help="Starting portfolio value for backtesting"
    )
   
    # Store in session state for access in backtesting
    st.session_state.initial_value = initial_value
   
    reference_CAGR = st.number_input(
        "Reference CAGR (%):",
        min_value=-50.0,
        max_value=100.0,
        value=8.0,
        step=0.5,
        help="Reference compound annual growth rate for comparison (e.g., S&P 500 historical average)"
    ) / 100
   
    # Store in session state for access in backtesting
    st.session_state.reference_CAGR = reference_CAGR
   
    # Setup constraints
    n_assets = len(tickers)
    if constraint_type == "Default (0-40%)":
        min_weights = np.zeros(n_assets)
        max_weights = np.full(n_assets, 0.4)
    elif constraint_type == "Equal Weight Only":
        equal_weight = 1.0 / n_assets
        min_weights = np.full(n_assets, equal_weight - 0.01)
        max_weights = np.full(n_assets, equal_weight + 0.01)
    else:  # Custom - constraints are already set in the expander above
        # min_weights and max_weights are already defined in the expander
        pass
   
    # Store constraints
    st.session_state.optimizer.min_weights = min_weights
    st.session_state.optimizer.max_weights = max_weights
    n_assets = len(asset_names)
    st.session_state.optimizer.initial_weights = np.full(n_assets, 1.0 / n_assets)
   
    # Configuration status
    st.markdown("---")
    st.markdown("**Current Configuration:**")
    st.write(f"• Mode: {mode_choice}")
    st.write(f"• Risk-free Rate: {risk_free_rate/100:.1%}")
    st.write(f"• Data Period: {years} years")
    st.write(f"• Assets: {len(tickers)}")
    st.write(f"• Rebalancing: {rebalance_freq}")
    st.write(f"• Transaction Cost: {transaction_cost:.2%}")
    st.write(f"• Reference CAGR: {reference_CAGR:.1%}")
   
    if st.session_state.optimizer.mode == 'forward_looking':
        if hasattr(st.session_state.optimizer, 'expected_returns') and st.session_state.optimizer.expected_returns is not None:
            avg_return = st.session_state.optimizer.expected_returns.mean()
            avg_vol = st.session_state.optimizer.volatilities.mean()
            st.write(f"• Avg Expected Return: {avg_return:.1%}")
            st.write(f"• Avg Volatility: {avg_vol:.1%}")



# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if len(tickers) >= 3:
    # Analysis Button
    if st.button("🚀 Run Complete Portfolio Optimization & Backtesting", type="primary", use_container_width=True):
       
        # Reset optimizer to ensure we have the latest version
        st.session_state.optimizer = EnhancedPortfolioOptimizer()
       
        # Reconfigure optimizer with current settings
        st.session_state.optimizer.mode = 'backward_looking' if mode_choice == "Backward-Looking" else 'forward_looking'
        st.session_state.optimizer.risk_free_rate = risk_free_rate / 100
       
        if st.session_state.optimizer.mode == 'backward_looking':
            st.session_state.optimizer.lookback_years = years
        else:
            # Forward-looking mode - parameters already set in sidebar
            pass
       
        # Set forward-looking parameters if needed
        if st.session_state.optimizer.mode == 'forward_looking':
            if 'expected_returns' in st.session_state and 'volatilities' in st.session_state:
                st.session_state.optimizer.expected_returns = pd.Series(st.session_state.expected_returns, index=asset_names)
                st.session_state.optimizer.volatilities = pd.Series(st.session_state.volatilities, index=asset_names)
            else:
                st.error("Forward-looking parameters not configured. Please configure expected returns and volatilities in the sidebar.")
                st.stop()
       
        # Set required optimizer attributes
        st.session_state.optimizer.asset_names = asset_names
        st.session_state.optimizer.min_weights = min_weights
        st.session_state.optimizer.max_weights = max_weights
        n_assets = len(asset_names)
        st.session_state.optimizer.initial_weights = np.full(n_assets, 1.0 / n_assets)
       
        with st.spinner("Running comprehensive portfolio analysis with backtesting..."):
           
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status = st.empty()
               
                # Step 1: Data Fetching
                status.info("📊 Fetching market data...")
                progress_bar.progress(0.15)
               
                # Fetch market data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years * 365)
               
                data = yf.download(tickers, start=start_date, end=end_date, progress=False, timeout=15)
               
                if data.empty:
                    st.error("Failed to fetch market data. Please check tickers and internet connection.")
                    st.stop()
               
                # Process price data
                if len(tickers) > 1:
                    if isinstance(data.columns, pd.MultiIndex):
                        price_data = pd.DataFrame()
                        for ticker in tickers:
                            if ('Adj Close', ticker) in data.columns:
                                price_data[ticker] = data[('Adj Close', ticker)]
                            elif ('Close', ticker) in data.columns:
                                price_data[ticker] = data[('Close', ticker)]
                    else:
                        price_data = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                else:
                    price_data = pd.DataFrame()
                    ticker = tickers[0]
                    price_data[ticker] = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
               
                # Clean data
                price_data = price_data.fillna(method='ffill').fillna(method='bfill').dropna()
                price_data.columns = asset_names
               
                st.session_state.optimizer.price_data = price_data
               
                # Step 2: Calculate Statistics
                status.info("📈 Calculating portfolio statistics...")
                progress_bar.progress(0.35)
               
                # Calculate returns
                returns_data = price_data.pct_change().dropna()
                returns_data = returns_data.replace([np.inf, -np.inf], np.nan).dropna()
                st.session_state.optimizer.returns_data = returns_data
               
                if st.session_state.optimizer.mode == 'backward_looking':
                    # Use historical data for all calculations
                    trading_days = 252
                    st.session_state.optimizer.expected_returns = returns_data.mean() * trading_days
                    st.session_state.optimizer.volatilities = returns_data.std() * np.sqrt(trading_days)
                else:
                    # Forward-looking mode - parameters already set in sidebar
                    pass
               
                # Calculate correlation and covariance matrices
                correlation_matrix = returns_data.corr().values
                st.session_state.optimizer.correlation_matrix = st.session_state.optimizer._regularize_correlation_matrix(correlation_matrix)
               
                if st.session_state.optimizer.mode == 'backward_looking':
                    cov_matrix = returns_data.cov().values * 252
                else:
                    # Create covariance matrix using forward-looking volatilities
                    if hasattr(st.session_state.optimizer, 'volatilities') and st.session_state.optimizer.volatilities is not None:
                        vol_matrix = np.outer(st.session_state.optimizer.volatilities.values, st.session_state.optimizer.volatilities.values)
                        cov_matrix = st.session_state.optimizer.correlation_matrix * vol_matrix
                    else:
                        st.error("Forward-looking volatilities not properly set. Please check your configuration.")
                        st.stop()
               
                st.session_state.optimizer.covariance_matrix = st.session_state.optimizer._regularize_covariance_matrix(cov_matrix)
               
                # Step 3: Portfolio Optimization
                status.info("🎯 Optimizing portfolios...")
                progress_bar.progress(0.55)
               
                # Ensure optimizer is properly initialized
                if not hasattr(st.session_state.optimizer, 'asset_names') or st.session_state.optimizer.asset_names is None:
                    st.error("Optimizer not properly initialized. Please check your asset selection.")
                    st.stop()
               
                if not hasattr(st.session_state.optimizer, 'expected_returns') or st.session_state.optimizer.expected_returns is None:
                    st.error("Expected returns not calculated. Please check your data.")
                    st.stop()
               
                results = st.session_state.optimizer.optimize_portfolio()
               
                # Check if results are valid
                if results is None or len(results) == 0:
                    st.error("Portfolio optimization failed. Please check your data and constraints.")
                    st.stop()
               
                # Step 4: Run Backtesting
                status.info("🔄 Running backtesting analysis...")
                progress_bar.progress(0.75)
               
                # Prepare strategy weights for backtesting
                strategy_weights = {}
                for strategy_name, strategy_data in results.items():
                    strategy_weights[strategy_name] = strategy_data['weights']
               
                # Run comprehensive backtest
                backtest_results = st.session_state.optimizer.run_backtest(
                    strategy_weights,
                    st.session_state.rebalance_freq,
                    st.session_state.transaction_cost,
                    st.session_state.initial_value,
                    st.session_state.reference_CAGR
                )
               
                # Step 5: Complete
                progress_bar.progress(1.0)
                status.success("✅ Analysis and backtesting complete!")
               
                st.session_state.results = results
                st.session_state.backtest_complete = True
                st.session_state.analysis_complete = True
               
                time.sleep(1)
                st.rerun()
               
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)

# ============================================================================
# RESULTS DISPLAY WITH BACKTESTING
# ============================================================================

if st.session_state.analysis_complete and st.session_state.results:
   
    st.markdown("---")
    st.markdown("## 📊 Portfolio Optimization & Backtesting Results")
   
    results = st.session_state.results
    optimizer = st.session_state.optimizer
   
    # Key Metrics
    if 'max_sharpe' in results:
        best = results['max_sharpe']
    else:
        best = results['initial']
   
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.metric(
            "🎯 Expected Return",
            f"{best['return']:.2%}",
            delta=f"{(best['return'] - optimizer.risk_free_rate):.2%} excess"
        )
   
    with col2:
        st.metric(
            "📊 Volatility",
            f"{best['volatility']:.2%}",
            help="Annual standard deviation"
        )
   
    with col3:
        st.metric(
            "⭐ Sharpe Ratio",
            f"{best['sharpe_ratio']:.3f}",
            help="Risk-adjusted return measure"
        )
   
    with col4:
        # Show backtesting metric if available
        if st.session_state.backtest_complete and 'max_sharpe' in optimizer.backtest_results:
            # Force recalculation of CAGR to ensure we get the updated value
            backtest_data = optimizer.backtest_results['max_sharpe']
            if 'portfolio_df' in backtest_data:
                # Recalculate CAGR using current method
                portfolio_df = backtest_data['portfolio_df']
                initial_value = 100000  # Default initial value
                start_date = portfolio_df.index[0]
                end_date = portfolio_df.index[-1]
                days_between = (end_date - start_date).days
                years_between = days_between / 365.25
               
                if years_between > 0:
                    backtest_return = ((portfolio_df['portfolio_value'].iloc[-1] / initial_value) ** (1 / years_between) - 1)
                else:
                    backtest_return = backtest_data['metrics']['annualized_return']
            else:
                backtest_return = backtest_data['metrics']['annualized_return']
               
            st.metric(
                "📈 Backtest Return",
                f"{backtest_return:.2%}",
                help="CAGR (Compound Annual Growth Rate) achieved during the backtesting period"
            )
        else:
            st.metric("📈 Backtest Return", "N/A")
   
    # Results Tabs with Backtesting
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Performance", "⚖️ Allocation", "🔄 Backtesting",
        "📊 Strategy Comparison", "📉 Analytics", "📥 Export"
    ])
   
    with tab1:
        st.subheader("Portfolio Performance Analysis")
       
        # Performance comparison table
        perf_data = []
        for portfolio_type, data in results.items():
            perf_data.append({
                'Portfolio': portfolio_type.replace('_', ' ').title(),
                'Return': f"{data['return']:.2%}",
                'Volatility': f"{data['volatility']:.2%}",
                'Sharpe Ratio': f"{data['sharpe_ratio']:.3f}",
                'Excess Return': f"{(data['return'] - optimizer.risk_free_rate):.2%}"
            })
       
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
       
        # Risk-return scatter plot with efficient frontier
        st.subheader("Risk-Return Profile with Efficient Frontier")
       
        fig = go.Figure()
       
        # Plot efficient frontier
        try:
            eff_ret, eff_vol = optimizer.efficient_frontier()
            if len(eff_ret) > 0:
                # Sort by volatility to ensure smooth line
                sorted_indices = np.argsort(eff_vol)
                eff_vol_sorted = eff_vol[sorted_indices]
                eff_ret_sorted = eff_ret[sorted_indices]
               
               
                fig.add_trace(go.Scatter(
                    x=eff_vol_sorted, y=eff_ret_sorted,
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='lightblue', width=3, dash='dash'),
                    hovertemplate='Efficient Frontier<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))
        except Exception as e:
            st.warning(f"Could not plot efficient frontier: {e}")
            pass
       
        # Plot individual assets
        for i, name in enumerate(asset_names):
            fig.add_trace(go.Scatter(
                x=[optimizer.volatilities.iloc[i]],
                y=[optimizer.expected_returns.iloc[i]],
                mode='markers',
                marker=dict(size=12, color=COLORS[i % len(COLORS)],
                           line=dict(width=2, color='white')),
                name=name,
                hovertemplate=f"<b>{name}</b><br>Return: %{{y:.2%}}<br>Volatility: %{{x:.2%}}<extra></extra>"
            ))
       
        # Plot portfolios
        portfolio_colors = {'initial': '#FF6B6B', 'min_variance': '#45B7D1',
                           'max_sharpe': '#2ECC71', 'max_return': '#F39C12'}
        portfolio_symbols = {'initial': 'circle', 'min_variance': 'triangle-up',
                            'max_sharpe': 'star', 'max_return': 'diamond'}
       
        for portfolio_type, data in results.items():
            fig.add_trace(go.Scatter(
                x=[data['volatility']],
                y=[data['return']],
                mode='markers',
                marker=dict(
                    size=16,
                    color=portfolio_colors.get(portfolio_type, '#95A5A6'),
                    symbol=portfolio_symbols.get(portfolio_type, 'circle'),
                    line=dict(width=2, color='white')
                ),
                name=f"{portfolio_type.replace('_', ' ').title()} Portfolio",
                hovertemplate=f"<b>{portfolio_type.replace('_', ' ').title()}</b><br>Return: %{{y:.2%}}<br>Volatility: %{{x:.2%}}<br>Sharpe: {data['sharpe_ratio']:.3f}<extra></extra>"
            ))
       
        fig.update_layout(
            xaxis_title="Volatility (Annual)",
            yaxis_title="Expected Return (Annual)",
            template="plotly_white",
            height=600,
            hovermode='closest',
            yaxis=dict(
                range=[0, None],
                zeroline=True,
                zerolinecolor='lightgray',
                zerolinewidth=1
            ),
            xaxis=dict(
                range=[0, None],
                zeroline=True,
                zerolinecolor='lightgray',
                zerolinewidth=1
            )
        )
       
        fig = safe_update_axes(fig, x_format='.1%', y_format='.1%')
        safe_plotly_chart(fig, use_container_width=True)
   
    with tab2:
        st.subheader("Portfolio Allocation Analysis")
       
        # Show optimal portfolio allocation
        if 'max_sharpe' in results:
            portfolio_data = results['max_sharpe']
            title = "🏆 Optimal Portfolio (Maximum Sharpe Ratio)"
        else:
            portfolio_data = results['initial']
            title = "📊 Portfolio Allocation"
       
        st.markdown(f"**{title}**")
       
        weights = portfolio_data['weights']
        # Filter out zero allocations
        filtered = [(name, w) for name, w in zip(asset_names, weights) if w > 0.001]
        if filtered:
            filtered_names, filtered_weights = zip(*filtered)
        else:
            filtered_names, filtered_weights = [], []
        # Allocation table
        alloc_data = []
        for i, (name, ticker) in enumerate(zip(asset_names, tickers)):
            alloc_data.append({
                'Asset': name,
                'Ticker': ticker,
                'Weight': f"{weights[i]:.1%}",
                'Expected Return': f"{optimizer.expected_returns.iloc[i]:.2%}",
                'Volatility': f"{optimizer.volatilities.iloc[i]:.2%}",
                'Sharpe Ratio': f"{(optimizer.expected_returns.iloc[i] - optimizer.risk_free_rate) / optimizer.volatilities.iloc[i]:.3f}"
            })
        alloc_df = pd.DataFrame(alloc_data)
        st.dataframe(alloc_df, use_container_width=True)
        # Allocation visualization
        col1, col2 = st.columns(2)
        with col1:
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=filtered_names,
                values=filtered_weights,
                hole=0.4,
                marker_colors=COLORS[:len(filtered_names)],
                textinfo='label+percent',
                textposition='outside'
            )])
            fig_pie.update_layout(
                title="Portfolio Allocation Breakdown",
                template="plotly_white",
                height=500,
                showlegend=False
            )
            safe_plotly_chart(fig_pie, use_container_width=True)
       
        with col2:
            # Horizontal bar chart
            fig_bar = go.Figure(go.Bar(
                x=weights,
                y=asset_names,
                orientation='h',
                marker_color=COLORS[:len(asset_names)],
                text=[f"{w:.1%}" for w in weights],
                textposition='inside'
            ))
           
            fig_bar.update_layout(
                title="Weight Distribution",
                xaxis_title="Portfolio Weight",
                template="plotly_white",
                height=500
            )
           
            fig_bar = safe_update_axes(fig_bar, x_format='.0%')
            safe_plotly_chart(fig_bar, use_container_width=True)
   
    with tab3:
        if st.session_state.backtest_complete:
            st.subheader("🔄 Backtesting Analysis")
           
            backtest_results = optimizer.backtest_results
           
            # Portfolio Value Over Time
            st.subheader("Portfolio Value Evolution")
           
            fig_value = go.Figure()
           
            for idx, (strategy_name, results_data) in enumerate(backtest_results.items()):
                portfolio_df = results_data['portfolio_df']
                fig_value.add_trace(go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name=strategy_name.replace('_', ' ').title(),
                    line=dict(width=2, color=COLORS[idx % len(COLORS)]),
                    hovertemplate=f"<b>{strategy_name.replace('_', ' ').title()}</b><br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>"
                ))
           
            fig_value.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
           
            safe_plotly_chart(fig_value, use_container_width=True)
           
            # Cumulative Returns Comparison
            st.subheader("Cumulative Returns Comparison")
           
            fig_returns = go.Figure()
           
            for idx, (strategy_name, results_data) in enumerate(backtest_results.items()):
                # Check if required keys exist
                if 'cumulative_returns' not in results_data:
                    st.warning(f"Missing cumulative_returns data for {strategy_name}")
                    continue
                   
                cumulative_returns = results_data['cumulative_returns']
                cumulative_returns_pct = (cumulative_returns - 1) * 100
               
                fig_returns.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns_pct,
                    mode='lines',
                    name=strategy_name.replace('_', ' ').title(),
                    line=dict(width=2, color=COLORS[idx % len(COLORS)]),
                    hovertemplate=f"<b>{strategy_name.replace('_', ' ').title()}</b><br>Date: %{{x}}<br>Return: %{{y:.1f}}%<extra></extra>"
                ))
           
            fig_returns.update_layout(
                title="Cumulative Returns Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
           
            fig_returns = safe_update_axes(fig_returns, y_format='.1f')
            safe_plotly_chart(fig_returns, use_container_width=True)
           
            # Drawdown Analysis
            st.subheader("Drawdown Analysis")
           
            fig_dd = go.Figure()
           
            for idx, (strategy_name, results_data) in enumerate(backtest_results.items()):
                # Skip reference CAGR for drawdown analysis
                if strategy_name == 'reference_CAGR':
                    continue
                   
                # Check if required keys exist
                if 'drawdowns' not in results_data:
                    st.warning(f"Missing drawdowns data for {strategy_name}")
                    continue
                   
                drawdowns = results_data['drawdowns']
               
                fig_dd.add_trace(go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns,
                    mode='lines',
                    name=strategy_name.replace('_', ' ').title(),
                    line=dict(width=2, color=COLORS[idx % len(COLORS)]),
                    hovertemplate=f"<b>{strategy_name.replace('_', ' ').title()}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>"
                ))
           
            fig_dd.update_layout(
                title="Portfolio Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
           
            fig_dd = safe_update_axes(fig_dd, y_format='.1f')
            safe_plotly_chart(fig_dd, use_container_width=True)
           
                    # Rolling performance analysis (if backtesting data available)
        if st.session_state.backtest_complete:
            st.subheader("Rolling Performance Analysis")
           
            # Calculate rolling metrics for backtested strategies
            window = 60  # 60-day rolling window
           
            fig_rolling = go.Figure()
           
            for idx, (strategy_name, results_data) in enumerate(optimizer.backtest_results.items()):
                # Skip reference CAGR for analytics
                if strategy_name == 'reference_CAGR':
                    continue
                   
                # Check if portfolio_returns exists
                if 'portfolio_returns' not in results_data:
                    st.warning(f"Missing portfolio_returns data for {strategy_name}")
                    continue
                   
                portfolio_returns = results_data['portfolio_returns']
                if len(portfolio_returns) > window:
                    # Compute rolling volatility (annualized)
                    rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252) * 100
                    fig_rolling.add_trace(go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol,
                        mode='lines',
                        name=f"{strategy_name.replace('_', ' ').title()}",
                        line=dict(width=2, color=COLORS[idx % len(COLORS)])
                    ))
           
            fig_rolling.update_layout(
                title=f"Rolling Volatility ({window}-Day Window)",
                xaxis_title="Date",
                yaxis_title="Rolling Volatility (%)",
                template="plotly_white",
                height=400
            )
           
            safe_plotly_chart(fig_rolling, use_container_width=True)
           

            # Transaction Costs and Turnover
            if optimizer.rebalancing_dates:
                st.subheader("Transaction Costs and Turnover")
               
                # Show rebalancing dates
                st.write(f"**Rebalancing Dates:** {len(optimizer.rebalancing_dates)} rebalances")
                st.write(f"**Rebalancing Frequency:** {st.session_state.get('rebalance_freq', 'quarterly')}")
               
                # Check if we have transaction costs data
                has_transaction_data = False
                transaction_data_summary = []
               
                for strategy_name, results_data in optimizer.backtest_results.items():
                    if strategy_name != 'reference_CAGR':
                        portfolio_df = results_data['portfolio_df']
                       
                        if 'transaction_costs' in portfolio_df.columns and 'turnover' in portfolio_df.columns:
                            costs_data = portfolio_df[portfolio_df['transaction_costs'] > 0]
                            turnover_data = portfolio_df[portfolio_df['turnover'] > 0]
                           
                            # Summary info
                            total_costs = costs_data['transaction_costs'].sum() if len(costs_data) > 0 else 0
                            total_turnover = turnover_data['turnover'].sum() if len(turnover_data) > 0 else 0
                            transaction_data_summary.append({
                                'Strategy': strategy_name,
                                'Cost Events': len(costs_data),
                                'Total Costs': f"${total_costs:.2f}",
                                'Turnover Events': len(turnover_data),
                                'Total Turnover': f"{total_turnover*100:.2f}%"
                            })
                           
                            if len(costs_data) > 0 or len(turnover_data) > 0:
                                has_transaction_data = True
               
                # Show transaction data summary
                if transaction_data_summary:
                    st.write("**Transaction Data Summary:**")
                    summary_df = pd.DataFrame(transaction_data_summary)
                    st.dataframe(summary_df, use_container_width=True)
               
                if has_transaction_data:
                    # Check if we have turnover data (depends on rebalancing frequency)
                    has_turnover_data = False
                    for strategy_name, results_data in optimizer.backtest_results.items():
                        if strategy_name != 'reference_CAGR':
                            portfolio_df = results_data['portfolio_df']
                            if 'turnover' in portfolio_df.columns:
                                turnover_data = portfolio_df[portfolio_df['turnover'] > 0]
                                if len(turnover_data) > 0:
                                    has_turnover_data = True
                                    break
                   
                    # Check if we have transaction costs data (depends on transaction cost setting)
                    has_costs_data = False
                    for strategy_name, results_data in optimizer.backtest_results.items():
                        if strategy_name != 'reference_CAGR':
                            portfolio_df = results_data['portfolio_df']
                            if 'transaction_costs' in portfolio_df.columns:
                                costs_data = portfolio_df[portfolio_df['transaction_costs'] > 0]
                                if len(costs_data) > 0:
                                    has_costs_data = True
                                    break
                   
                    if has_turnover_data:
                        # Determine subplot configuration based on available data
                        if has_costs_data:
                            # Both transaction costs and turnover
                            fig_costs = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=('Transaction Costs Over Time', 'Portfolio Turnover'),
                                vertical_spacing=0.15
                            )
                            subplot_rows = 2
                        else:
                            # Only turnover
                            fig_costs = make_subplots(
                                rows=1, cols=1,
                                subplot_titles=('Portfolio Turnover',),
                                vertical_spacing=0.15
                            )
                            subplot_rows = 1
                       
                        for idx, (strategy_name, results_data) in enumerate(optimizer.backtest_results.items()):
                            if strategy_name != 'reference_CAGR':  # Reference CAGR has no transaction costs
                                portfolio_df = results_data['portfolio_df']
                               
                                # Transaction costs (only if we have costs data)
                                if has_costs_data:
                                    costs_data = portfolio_df[portfolio_df['transaction_costs'] > 0]
                                    if len(costs_data) > 0:
                                        fig_costs.add_trace(
                                            go.Scatter(
                                                x=costs_data.index,
                                                y=costs_data['transaction_costs'],
                                                mode='markers+lines',
                                                name=f"{strategy_name.replace('_', ' ').title()} Costs",
                                                showlegend=True,
                                                line=dict(color=COLORS[idx % len(COLORS)])
                                            ),
                                            row=1, col=1
                                        )
                               
                                # Turnover (always show if available)
                                turnover_data = portfolio_df[portfolio_df['turnover'] > 0]
                                if len(turnover_data) > 0:
                                    fig_costs.add_trace(
                                        go.Scatter(
                                            x=turnover_data.index,
                                            y=turnover_data['turnover'] * 100,
                                            mode='markers+lines',
                                            name=f"{strategy_name.replace('_', ' ').title()} Turnover",
                                            showlegend=True,
                                            line=dict(color=COLORS[idx % len(COLORS)])
                                        ),
                                        row=subplot_rows, col=1
                                    )
                       
                        # Update axes based on subplot configuration
                        if has_costs_data:
                            fig_costs.update_xaxes(title_text="Date", row=2, col=1)
                            fig_costs.update_yaxes(title_text="Cost ($)", row=1, col=1)
                            fig_costs.update_yaxes(title_text="Turnover (%)", row=2, col=1)
                        else:
                            fig_costs.update_xaxes(title_text="Date", row=1, col=1)
                            fig_costs.update_yaxes(title_text="Turnover (%)", row=1, col=1)
                       
                        fig_costs.update_layout(
                            template="plotly_white",
                            height=400 if not has_costs_data else 600
                        )
                       
                        safe_plotly_chart(fig_costs, use_container_width=True)
                       
                        if not has_costs_data:
                            st.info("ℹ️ Transaction costs subplot is hidden because transaction cost is set to 0. Set a transaction cost > 0 to see transaction costs analysis.")
                    else:
                        st.info("ℹ️ No turnover data available. This may occur if rebalancing frequency is set to 'none' or there are no rebalancing events.")
                else:
                    st.warning("⚠️ No transaction costs or turnover data available. This may occur if:")
                    st.write("• No rebalancing events happened during the backtesting period")
                    st.write("• The rebalancing frequency is set to 'none'")
                    st.write("• The backtesting period is too short for rebalancing events")
                    st.write("• There's an issue with the transaction cost calculation")
            else:
                st.warning("⚠️ No rebalancing dates available. Transaction costs and turnover analysis requires rebalancing events.")
                st.write("**Possible causes:**")
                st.write("• Rebalancing frequency is set to 'none'")
                st.write("• The backtesting period is too short")
                st.write("• There's an issue with the rebalancing date generation")
       
        else:
            st.info("Run the complete analysis to see backtesting results.")
   
    with tab4:
        if st.session_state.backtest_complete:
            st.subheader("📊 Strategy Performance Comparison")
           
            # Comprehensive comparison table
            comparison_df = optimizer.calculate_strategy_comparison()
           
            if comparison_df is not None:
                st.dataframe(comparison_df, use_container_width=True)
               
                # Performance metrics visualization
                st.subheader("Performance Metrics Comparison")
               
                # Create comparison charts
                metrics_to_plot = ['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
               
                fig_comparison = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=metrics_to_plot,
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
               
                strategies = comparison_df['Strategy'].tolist()
                colors = COLORS[:len(strategies)]
               
                # Extract numeric values for plotting
                total_returns = [float(x.replace('%', '')) for x in comparison_df['Total Return (%)']]
                volatilities = [float(x.replace('%', '')) for x in comparison_df['Volatility (%)']]
                sharpe_ratios = [float(x) for x in comparison_df['Sharpe Ratio']]
                max_drawdowns = [abs(float(x.replace('%', ''))) for x in comparison_df['Max Drawdown (%)']]
               
                # Add traces
                fig_comparison.add_trace(
                    go.Bar(x=strategies, y=total_returns, marker_color=colors, name='Total Return', showlegend=False),
                    row=1, col=1
                )
               
                fig_comparison.add_trace(
                    go.Bar(x=strategies, y=volatilities, marker_color=colors, name='Volatility', showlegend=False),
                    row=1, col=2
                )
               
                fig_comparison.add_trace(
                    go.Bar(x=strategies, y=sharpe_ratios, marker_color=colors, name='Sharpe Ratio', showlegend=False),
                    row=2, col=1
                )
               
                fig_comparison.add_trace(
                    go.Bar(x=strategies, y=max_drawdowns, marker_color=colors, name='Max Drawdown', showlegend=False),
                    row=2, col=2
                )
               
                fig_comparison.update_layout(height=600, template="plotly_white")
                safe_plotly_chart(fig_comparison, use_container_width=True)
               
                # Risk-Return Scatter of Strategies
                st.subheader("Strategy Risk-Return Profile")
               
                fig_scatter = go.Figure()
               
                for i, strategy in enumerate(strategies):
                    fig_scatter.add_trace(go.Scatter(
                        x=[volatilities[i]],
                        y=[total_returns[i]],
                        mode='markers+text',
                        marker=dict(size=15, color=colors[i]),
                        text=[strategy],
                        textposition="top center",
                        name=strategy,
                        hovertemplate=f"<b>{strategy}</b><br>Volatility: {volatilities[i]:.1f}%<br>Return: {total_returns[i]:.1f}%<br>Sharpe: {sharpe_ratios[i]:.3f}<extra></extra>"
                    ))
               
                fig_scatter.update_layout(
                    title="Strategy Risk vs Return (Backtested)",
                    xaxis_title="Volatility (%)",
                    yaxis_title="Total Return (%)",
                    template="plotly_white",
                    height=500,
                    showlegend=False
                )
               
                safe_plotly_chart(fig_scatter, use_container_width=True)
       
        else:
            st.info("Run the complete analysis to see strategy comparison.")
   
    with tab5:
        st.subheader("Advanced Analytics")
       
        # Correlation heatmap
        st.subheader("Asset Correlation Matrix")
       
        corr_df = pd.DataFrame(
            optimizer.correlation_matrix,
            columns=asset_names,
            index=asset_names
        )
       
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_df.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
       
        fig_corr.update_layout(
            title="Asset Correlation Matrix",
            template="plotly_white",
            height=500
        )
       
        safe_plotly_chart(fig_corr, use_container_width=True)
       
        # Risk contribution analysis (if backtesting completed)
        if st.session_state.backtest_complete and 'max_sharpe' in results:
            st.subheader("Risk Contribution Analysis")
           
            optimal_weights = results['max_sharpe']['weights']
            portfolio_return = results['max_sharpe']['return']
           
            # Calculate return contribution using the correct formula
            # Return Contribution = (Weight in Portfolio × Return of Asset Class) / Portfolio Return
            return_contrib = (optimal_weights * optimizer.expected_returns.values) / portfolio_return
           
            # Calculate marginal risk contribution
            portfolio_vol = results['max_sharpe']['volatility']
            marginal_contrib = np.dot(optimizer.covariance_matrix, optimal_weights) / portfolio_vol
            risk_contrib = optimal_weights * marginal_contrib
            risk_contrib_pct = risk_contrib / np.sum(risk_contrib)
           
            risk_data = pd.DataFrame({
                'Asset': asset_names,
                'Weight': optimal_weights,
                'Return Contribution': return_contrib,
                'Risk Contribution': risk_contrib_pct,
                'Risk-to-Return Ratio': risk_contrib_pct / return_contrib
            })
           
            # Risk contribution chart
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Bar(
                x=asset_names,
                y=return_contrib,
                name='Return Contribution',
                marker_color='#162d73',
                opacity=1.0,
                offsetgroup='1',
                legendgroup='Return Contribution',
                showlegend=True
            ))
            fig_risk.add_trace(go.Bar(
                x=asset_names,
                y=risk_contrib_pct,
                name='Risk Contribution',  # Changed from 'Risk'
                marker_color='#888888',
                opacity=1.0,
                offsetgroup='2',
                legendgroup='Risk Contribution',
                showlegend=True
            ))
            fig_risk.update_layout(
                title="Return Contribution vs Risk Contribution",
                xaxis_title="Assets",
                template="plotly_white",
                height=400,
                barmode='group',
                yaxis=dict(title="Contribution", side="left"),
                yaxis2=dict(title="Contribution", side="right", overlaying="y")
            )
           
            try:
                fig_risk.update_yaxes(tickformat='.1%', secondary_y=False)
                fig_risk.update_yaxes(tickformat='.1%', secondary_y=True)
            except:
                pass
           
            safe_plotly_chart(fig_risk, use_container_width=True)
           
            # Risk contribution table
            # Format the dataframe to show percentages with 2 decimal places
            formatted_risk_data = risk_data.copy()
           
            # Format columns with special handling for zero weights
            formatted_risk_data['Weight'] = formatted_risk_data['Weight'].apply(lambda x: f"{x:.2%}")
            formatted_risk_data['Return Contribution'] = formatted_risk_data.apply(
                lambda row: "-" if abs(risk_data.loc[row.name, 'Weight']) < 0.001 else f"{row['Return Contribution']:.2%}", axis=1
            )
            formatted_risk_data['Risk Contribution'] = formatted_risk_data.apply(
                lambda row: "-" if abs(risk_data.loc[row.name, 'Weight']) < 0.001 else f"{row['Risk Contribution']:.2%}", axis=1
            )
            formatted_risk_data['Risk-to-Return Ratio'] = formatted_risk_data.apply(
                lambda row: "-" if abs(risk_data.loc[row.name, 'Weight']) < 0.001 else f"{row['Risk-to-Return Ratio']:.4f}", axis=1
            )
           
            st.dataframe(formatted_risk_data, use_container_width=True)
       

            # Performance statistics summary
            st.subheader("Detailed Performance Statistics")
           
            if optimizer.backtest_results:
                stats_data = []
               
                for strategy_name, results_data in optimizer.backtest_results.items():
                    # Skip reference CAGR for performance statistics
                    if strategy_name == 'reference_CAGR':
                        continue
                       
                    # Check if portfolio_returns exists
                    if 'portfolio_returns' not in results_data:
                        st.warning(f"Missing portfolio_returns data for {strategy_name}")
                        continue
                       
                    portfolio_returns = results_data['portfolio_returns']
                    metrics = results_data['metrics']
                   
                    # Calculate additional statistics
                    skewness = portfolio_returns.skew()
                    kurtosis = portfolio_returns.kurtosis()
                    var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252) * 100
                    var_99 = np.percentile(portfolio_returns, 1) * np.sqrt(252) * 100
                    # Calmar ratio
                    calmar_ratio = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
                   
                    # Sortino ratio (using negative returns only)
                    negative_returns = portfolio_returns[portfolio_returns < 0]
                    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
                    sortino_ratio = (metrics['annualized_return'] - optimizer.risk_free_rate * 100) / (downside_deviation * 100) if downside_deviation > 0 else 0
                   
                    stats_data.append({
                        'Strategy': strategy_name.replace('_', ' ').title(),
                        'Skewness': f"{skewness:.3f}",
                        'Kurtosis': f"{kurtosis:.3f}",
                        'VaR 95%': f"{var_95:.2f}%",
                        'VaR 99%': f"{var_99:.2f}%",
                        'Calmar Ratio': f"{calmar_ratio:.3f}",
                        'Sortino Ratio': f"{sortino_ratio:.3f}",
                        'Best Day': f"{portfolio_returns.max()*100:.2f}%",
                        'Worst Day': f"{portfolio_returns.min()*100:.2f}%"
                    })
               
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
   
    with tab6:
        st.subheader("Export Results")
       
        # Summary statistics
        st.subheader("Download Data")
       
        col1, col2 = st.columns(2)
       
        with col1:
            # Portfolio weights CSV
            weights_data = []
            for portfolio_type, data in results.items():
                row = {'Portfolio': portfolio_type.replace('_', ' ').title()}
                for i, name in enumerate(asset_names):
                    row[name] = data['weights'][i]
                weights_data.append(row)
           
            weights_df = pd.DataFrame(weights_data)
            csv_weights = weights_df.to_csv(index=False)
           
            st.download_button(
                label="📊 Download Portfolio Weights",
                data=csv_weights,
                file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
       
        with col2:
            # Performance summary CSV
            perf_df = pd.DataFrame(perf_data)
            csv_perf = perf_df.to_csv(index=False)
           
            st.download_button(
                label="📈 Download Performance Summary",
                data=csv_perf,
                file_name=f"portfolio_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
       
        # Backtesting results export
        if st.session_state.backtest_complete:
            st.subheader("Backtesting Data Export")
           
            col3, col4 = st.columns(2)
           
            with col3:
                # Strategy comparison CSV
                comparison_df = optimizer.calculate_strategy_comparison()
                if comparison_df is not None:
                    csv_comparison = comparison_df.to_csv(index=False)
                   
                    st.download_button(
                        label="🔄 Download Strategy Comparison",
                        data=csv_comparison,
                        file_name=f"strategy_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
           
            with col4:
                # Full backtesting data CSV
                if 'max_sharpe' in optimizer.backtest_results:
                    backtest_df = optimizer.backtest_results['max_sharpe']['portfolio_df']
                    csv_backtest = backtest_df.to_csv()
                   
                    st.download_button(
                        label="📊 Download Full Backtest Data",
                        data=csv_backtest,
                        file_name=f"backtest_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
       
        # Detailed report
        st.subheader("Comprehensive Analysis Report")
       
        # Generate comprehensive report with backtesting
        mode_title = "Backward-Looking Analysis" if optimizer.mode == 'backward_looking' else "Forward-Looking Analysis"
       
        report = f"""
VANTAGE CAPITAL PORTFOLIO OPTIMIZATION & BACKTESTING REPORT
{mode_title.upper()}
{'='*80}

Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}
Analysis Mode: {mode_title}
Risk-free Rate: {optimizer.risk_free_rate:.2%}
Data Period: {optimizer.lookback_years} years
Assets Analyzed: {len(optimizer.tickers)}

BACKTESTING CONFIGURATION:
{'-'*50}
Rebalancing Frequency: {st.session_state.get('rebalance_freq', 'quarterly')}
Transaction Cost: {optimizer.transaction_costs:.3%}
Initial Portfolio Value: ${st.session_state.get('initial_value', 100000):,}

ASSET SUMMARY:
{'-'*30}
"""
       
        for i, (name, ticker) in enumerate(zip(asset_names, tickers)):
            sharpe = (optimizer.expected_returns.iloc[i] - optimizer.risk_free_rate) / optimizer.volatilities.iloc[i]
            report += f"{name} ({ticker}):\n"
            report += f"  Expected Return: {optimizer.expected_returns.iloc[i]:.2%}\n"
            report += f"  Volatility: {optimizer.volatilities.iloc[i]:.2%}\n"
            report += f"  Sharpe Ratio: {sharpe:.3f}\n\n"
       
        report += "PORTFOLIO OPTIMIZATION RESULTS:\n"
        report += "-" * 40 + "\n"
       
        for portfolio_type, data in results.items():
            report += f"\n{portfolio_type.upper().replace('_', ' ')} PORTFOLIO:\n"
            report += f"Expected Return: {data['return']:.2%}\n"
            report += f"Volatility: {data['volatility']:.2%}\n"
            report += f"Sharpe Ratio: {data['sharpe_ratio']:.4f}\n"
            report += f"Excess Return: {(data['return'] - optimizer.risk_free_rate):.2%}\n"
           
            report += "\nAllocations:\n"
            weights_sorted = sorted(enumerate(data['weights']), key=lambda x: x[1], reverse=True)
            for i, (idx, weight) in enumerate(weights_sorted):
                if weight > 0.001:
                    report += f"  {asset_names[idx]}: {weight:.1%}\n"
            report += "\n"
       
        # Add backtesting results if available
        if st.session_state.backtest_complete:
            report += "BACKTESTING RESULTS:\n"
            report += "=" * 40 + "\n"
           
            comparison_df = optimizer.calculate_strategy_comparison()
            if comparison_df is not None:
                for _, row in comparison_df.iterrows():
                    strategy = row['Strategy']
                    report += f"\n{strategy.upper()}:\n"
                    report += f"  Total Return: {row['Total Return (%)']}\n"
                    report += f"  Annualized Return: {row['Annualized Return (%)']}\n"
                    report += f"  Volatility: {row['Volatility (%)']}\n"
                    report += f"  Sharpe Ratio: {row['Sharpe Ratio']}\n"
                    report += f"  Max Drawdown: {row['Max Drawdown (%)']}\n"
                    report += f"  Win Rate: {row['Win Rate (%)']}\n"
                    report += f"  Transaction Costs: {row['Transaction Costs (%)']}\n"
                    report += f"  Average Turnover: {row['Avg Turnover (%)']}\n"
                    report += f"  Final Portfolio Value: {row['Final Value ($)']}\n"
                    report += f"  Number of Rebalances: {row['Rebalances']}\n"
           
            if optimizer.rebalancing_dates:
                report += f"\nREBALANCING SCHEDULE:\n"
                report += f"Total Rebalances: {len(optimizer.rebalancing_dates)}\n"
                report += f"Rebalancing Dates:\n"
                for date in optimizer.rebalancing_dates[:10]:  # Show first 10 dates
                    report += f"  {date.strftime('%Y-%m-%d')}\n"
                if len(optimizer.rebalancing_dates) > 10:
                    report += f"  ... and {len(optimizer.rebalancing_dates) - 10} more\n"
       
        if 'max_sharpe' in results:
            best = results['max_sharpe']
            report += f"\nOPTIMAL PORTFOLIO SUMMARY:\n"
            report += f"{'='*40}\n"
            report += f"The optimal portfolio maximizes risk-adjusted returns with:\n"
            report += f"• Theoretical Sharpe ratio of {best['sharpe_ratio']:.3f}\n"
            report += f"• Expected annual return of {best['return']:.2%}\n"
            report += f"• Annual volatility of {best['volatility']:.2%}\n"
           
            if st.session_state.backtest_complete and 'max_sharpe' in optimizer.backtest_results:
                backtest_metrics = optimizer.backtest_results['max_sharpe']['metrics']
                report += f"\nBACKTESTED PERFORMANCE:\n"
                report += f"• Actual annualized return: {backtest_metrics['annualized_return']:.2%}\n"
                report += f"• Actual volatility: {backtest_metrics['volatility']:.2%}\n"
                report += f"• Actual Sharpe ratio: {backtest_metrics['sharpe_ratio']:.3f}\n"
                report += f"• Maximum drawdown: {backtest_metrics['max_drawdown']:.2%}\n"
                report += f"• Win rate: {backtest_metrics['win_rate']:.1f}%\n"
       
        report += f"\nMETHODOLOGY:\n"
        report += f"{'-'*20}\n"
        report += f"Expected Returns: Annualized using 252 trading days\n"
        report += f"Volatility: Annualized standard deviation (×√252)\n"
        report += f"Sharpe Ratio: (Return - Risk-free Rate) / Volatility\n"
        report += f"Optimization: Sequential Least Squares Programming (SLSQP)\n"
        report += f"Backtesting: Historical simulation with rebalancing\n"
        report += f"Transaction Costs: Applied at each rebalancing event\n"
       
        report += f"\nDISCLAIMER:\n"
        report += f"{'-'*20}\n"
        report += f"This analysis is for educational and informational purposes only.\n"
        report += f"Past performance does not guarantee future results.\n"
        report += f"Backtesting results may not reflect actual trading performance due to\n"
        report += f"market impact, liquidity constraints, and other real-world factors.\n"
        report += f"Please consult with a qualified financial advisor before making investment decisions.\n"
        report += f"\nGenerated by Vantage Capital Portfolio Optimization & Backtesting Platform\n"
       
        st.download_button(
            label="📋 Download Complete Analysis Report",
            data=report,
            file_name=f"portfolio_backtest_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
       
        # Show report preview
        with st.expander("📖 Report Preview"):
            st.text(report[:2500] + "..." if len(report) > 2500 else report)

else:
    # Welcome screen
    st.markdown("## 🛡️ Professional Portfolio Optimization & Backtesting")
   
    st.markdown("""
    Welcome to Vantage Capital's comprehensive portfolio optimization platform with advanced backtesting capabilities.
   
    ### ✨ New Backtesting Features:
   
    **🔄 Comprehensive Backtesting:**
    - **Multiple Strategies**: Compare optimized vs reference CAGR portfolios
    - **Realistic Simulation**: Account for transaction costs and rebalancing
    - **Performance Metrics**: Sharpe ratio, drawdown, win rate analysis
    - **Transaction Analysis**: Cost tracking and turnover monitoring
   
    **📊 Advanced Analytics:**
    - **Performance Over Time**: Portfolio value evolution and cumulative returns
    - **Drawdown Analysis**: Detailed drawdown tracking and recovery periods
    - **Strategy Comparison**: Side-by-side performance metrics
    - **Risk Contribution**: Understand each asset's contribution to portfolio risk
    - **Transaction Analysis**: Turnover rates and rebalancing costs
   
    **📈 Performance Metrics:**
    - **Risk-Adjusted Returns**: Sharpe, Sortino, and Calmar ratios
    - **Drawdown Statistics**: Maximum drawdown and recovery analysis
    - **Win Rates**: Percentage of positive return periods
    - **Value at Risk**: 95% and 99% VaR calculations
    - **Rolling Performance**: Time-varying risk and return metrics
   
    **🎨 Interactive Visualizations:**
    - **Portfolio Evolution**: Value over time for all strategies
    - **Cumulative Returns**: Performance comparison charts
    - **Drawdown Charts**: Visual risk assessment over time
    - **Transaction Costs**: Rebalancing cost analysis
    - **Risk Attribution**: Weight vs risk contribution analysis
   
    **📥 Enhanced Export:**
    - **Strategy Comparison**: Comprehensive performance table
    - **Backtest Data**: Full time series of portfolio values and metrics
    - **Transaction Records**: Complete rebalancing history
    - **Professional Reports**: Detailed analysis with backtesting results
   
    ### 🔧 Key Improvements Over Previous Version:
   
    ✅ **Replaced Rolling Sharpe Ratio** with comprehensive backtesting simulation  
    ✅ **Added Transaction Cost Modeling** for realistic performance assessment  
    ✅ **Multiple Rebalancing Strategies** with configurable frequencies  
    ✅ **Benchmark Comparison** against reference CAGR strategy  
    ✅ **Enhanced Risk Analysis** with detailed drawdown tracking  
    ✅ **Professional Reporting** with backtesting methodology  
   
    ### 🚀 Getting Started:
    1. **Configure** your analysis in the sidebar (minimum 3 assets required)
    2. **Set Backtesting Parameters** - rebalancing frequency and transaction costs
    3. **Choose Analysis Mode** - backward-looking or forward-looking
    4. **Select Assets** using presets or custom tickers
    5. **Run Complete Analysis** - optimization + backtesting
    6. **Explore Results** across multiple interactive tabs with backtesting insights
   
    ### 💡 Backtesting Pro Tips:
    - **Start with quarterly rebalancing** for most practical strategies
    - **Consider transaction costs** - they significantly impact real returns
    - **Compare multiple strategies** to understand trade-offs
    - **Analyze drawdowns** to assess risk tolerance requirements
    - **Review turnover rates** to optimize rebalancing frequency
   
    The backtesting engine provides realistic performance simulation that accounts for:
    - **Historical market conditions** and volatility regimes
    - **Rebalancing drift** due to differential asset performance  
    - **Transaction costs** at each rebalancing event
    - **Market timing** effects of periodic rebalancing
    """)
   
    # Asset selection reminder
    if len(tickers) < 3:
        st.warning("⚠️ Please select at least 3 assets in the sidebar to begin analysis.")
    else:
        st.success(f"✅ Ready to analyze {len(tickers)} assets: {', '.join(tickers)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>🛡️ Vantage Capital Portfolio Optimization & Backtesting Platform</strong></p>
    <p style="font-style: italic;">Professional Investment Analytics, Risk Management & Performance Simulation</p>
</div>
""", unsafe_allow_html=True)



# At the bottom of the main content area, add a light grey 'Restart Optimization' button
st.markdown("""
    <style>
    .restart-opt-btn button {
        background-color: #f0f0f0 !important;
        color: #162d73 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        margin-top: 2rem;
        margin-bottom: 2rem;
        transition: background 0.2s;
    }
    .restart-opt-btn button:hover {
        background-color: #e0e0e0 !important;
        color: #162d73 !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.container():
    col = st.columns([1, 6, 1])[1]
    with col:
        if st.button("Restart Optimization", key="restart_main", type="primary", use_container_width=True, help="Restart the optimization tool"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    st.markdown('<div class="restart-opt-btn"></div>', unsafe_allow_html=True)
