# 🛡️ Vantage Capital Portfolio Optimization

Professional portfolio optimization platform built with NiceGUI and advanced financial algorithms.

## 🚀 Live Demo
[Your deployed URL here]

## ✨ Features
- **Real-time Analysis**: Direct Python integration with Portfolio_Optimization_V7.py
- **Dual Modes**: Backward-looking (historical) and Forward-looking (custom)
- **Interactive Charts**: Plotly.js visualizations
- **Professional UI**: Modern, responsive design
- **Export Functions**: CSV and detailed reports

## 🔧 Local Development
```bash
pip install -r requirements.txt
python portfolio_nicegui_app.py
📊 Technology Stack

Framework: NiceGUI (Python web framework)
Charts: Plotly
Optimization: SciPy + Custom algorithms
Data: yfinance for market data
UI: Quasar/Vue.js (via NiceGUI)

📋 Usage

Configure analysis parameters in sidebar
Select asset tickers or use presets
Run portfolio optimization
Review interactive results
Export data and reports


Powered by Portfolio_Optimization_V7.py

## 🚀 Quick Deploy Commands

### Railway (Recommended)
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway deploy

# OR use web interface:
# Go to railway.app → New Project → Deploy from GitHub
Render
bash# Go to render.com → New Web Service
# Connect GitHub repo
# Build: pip install -r requirements.txt
# Start: python portfolio_nicegui_app.py
Heroku
bash# Install Heroku CLI, then:
heroku create vantage-portfolio
git push heroku main
🔧 Environment Variables (if needed)
PORT=8080                    # App port
PYTHON_VERSION=3.11.6        # Python version
WEB_CONCURRENCY=1            # Process count
🐛 Troubleshooting
Common Issues:

Import Error: Ensure Portfolio_Optimization_V7.py is in repo
Port Issues: NiceGUI handles port automatically
Memory Limits: Reduce asset limit if needed
Build Timeout: Simplify requirements.txt

Performance Tips:

Use caching for market data
Limit to 8 assets for optimal speed
Enable gzip compression
Use CDN for static assets

📞 Support

Check deployment logs for errors
Test locally first
Verify all files are committed to Git