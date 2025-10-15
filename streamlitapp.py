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
        self.mode = None  # 'backward_looking' or 'forward_looking'
        self.tickers = []
        self.asset_names = []
        self.expected_returns = None
        self.volatilities = None
        self.min_weights = None
        self.max_weights = None
        self.initial_weights = None
        self.lookback_years = None
        self.risk_free_rate = 0.02  # Default 2% annual risk-free rate
       
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
        self.transaction_costs = 0.001  # 0.1% default transaction cost
   
    def portfolio_stats(self, weights: np.ndarray):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        weights = np.array(weights).flatten()
        weights = weights / weights.sum()  # Normalize to ensure sum = 1
       
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
        cumulative = price_series / price_series.iloc[0]  # Normalize to start at 1
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
                current_weights /= current_weights.sum()  # Renormalize
               
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
            cagr = total_return / 100  # Convert percentage to decimal
       
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
            'transaction_costs': [0] * len(dates),  # No transaction costs for reference
            'turnover': [0] * len(dates)  # No turnover for reference
        }, index=dates)
       
        # Calculate metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / initial_value) - 1
        annualized_return = reference_CAGR  # This is the target CAGR
        volatility = 0  # Reference CAGR has no volatility
        sharpe_ratio = float('inf') if volatility == 0 else annualized_return / volatility
        max_drawdown = 0  # Reference CAGR has no drawdowns
       
        # Calculate win rate (all periods are "wins" for CAGR)
        win_rate = 100.0
       
        # Calculate cumulative returns and drawdowns for plotting
        cumulative_returns = portfolio_df['portfolio_value'] / initial_value
        drawdowns = pd.Series([0] * len(dates), index=dates)  # No drawdowns for reference CAGR
       
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
   
    /
