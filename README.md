<<<<<<< HEAD
# Stock Prediction 📈

A Django-based web application that uses LSTM (Long Short-Term Memory) neural networks to predict stock prices for the next 7 days. The application supports both global stocks (via Yahoo Finance) and Indian stocks (NSE).

## Features

✨ **Key Features:**
- 🔐 User Authentication (Sign Up & Login)
- 📊 Stock Price Prediction using LSTM deep learning model
- 📈 Technical Indicators (Moving Averages, Volatility)
- 💰 Buy/Sell/Hold Recommendations with confidence levels
- 📉 Real-time Stock Data Fetching (Yahoo Finance & NSE)
- 📊 Model Performance Metrics (MAE, RMSE, R², MAPE)
- 📱 Interactive Web Interface with Matplotlib Visualizations
- 🧮 Calculator Tool (bonus feature)

## Tech Stack

- **Backend**: Django 5.1.5
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Data Source**: yfinance, nsepy
- **Database**: SQLite (default, use PostgreSQL for production)

## Installation

### Prerequisites
- Python 3.10 or higher
- Git
- Virtual Environment (recommended)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/stock-prediction.git
   cd stock-prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your actual credentials
   nano .env  # or use your preferred editor
   ```

   **Required variables in `.env`:**
   - `SECRET_KEY`: Django secret key (generate one if not set)
   - `DEBUG`: Set to `False` for production
   - `EMAIL_HOST_USER`: Gmail address for email notifications
   - `EMAIL_HOST_PASSWORD`: Gmail app-specific password
   - `ALLOWED_HOSTS`: Comma-separated list of allowed hosts

5. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Create a superuser (optional, for admin panel):**
   ```bash
   python manage.py createsuperuser
   ```

7. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

   The application will be available at: `http://127.0.0.1:8000/`

## Usage

### Available Routes

| Route | Purpose | Auth Required |
|-------|---------|---------------|
| `/` | Home page | ❌ No |
| `/login/` | User login | ❌ No |
| `/signup/` | User registration | ❌ No |
| `/predict/` | Stock price prediction | ✅ Yes |
| `/calcy/` | Calculator tool | ✅ Yes |
| `/about/` | About page | ❌ No |
| `/admin/` | Django admin panel | ✅ Yes (Staff) |

### Making Predictions

1. **Sign up** or **log in** to your account
2. Navigate to the **Predict** page
3. Enter a stock ticker symbol:
   - Global stocks: `AAPL`, `GOOGL`, `MSFT` (assumes USD)
   - Indian stocks: `INFY`, `TCS`, `RELIANCE` (assumes INR)
   - Or use NSE format: `INFY.NS`
4. View:
   - Historical price chart (last 90 days)
   - Predicted prices for the next 7 days
   - Model accuracy metrics
   - Trading recommendation (Buy/Sell/Hold)

## Model Architecture

The LSTM model uses the following architecture:

```
Input Layer: 60 days of historical data (8 features)
↓
LSTM Layer 1: 100 units + Dropout(0.2)
↓
LSTM Layer 2: 100 units + Dropout(0.2)
↓
LSTM Layer 3: 100 units + Dropout(0.2)
↓
Dense Layer 1: 25 units
↓
Output Layer: 1 unit (predicted price)
```

### Features Used:
- Close Price
- Open Price
- High Price
- Low Price
- Volume
- 5-day Moving Average (MA5)
- 20-day Moving Average (MA20)
- 20-day Volatility

## Performance Metrics

The model evaluates performance using:
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Standard deviation of errors
- **R²** (Coefficient of Determination): Goodness of fit
- **MAPE** (Mean Absolute Percentage Error): Percentage accuracy

## Project Structure

```
Stock Prediction/
├── Basics/                    # Django app
│   ├── migrations/            # Database migrations
│   ├── templates/             # HTML templates
│   │   ├── home.html
│   │   ├── predict.html
│   │   ├── login.html
│   │   ├── signup.html
│   │   └── ...
│   ├── models.py              # Database models (OTP)
│   ├── views.py               # View logic (prediction, auth)
│   ├── urls.py                # App URLs
│   └── ML.ipynb               # Jupyter notebook with model training
├── Pradeep/                   # Django project settings
│   ├── settings.py            # Configuration
│   ├── urls.py                # Project URLs
│   ├── wsgi.py
│   └── asgi.py
├── media/                     # Generated plots and uploads
├── models/                    # Saved ML models
│   └── stock_model.keras
├── manage.py                  # Django management script
├── requirements.txt           # Python dependencies
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Deployment

### For Production:

1. **Update `.env` with production settings:**
   ```
   DEBUG=False
   ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
   ```

2. **Use a production database:**
   ```bash
   # Install PostgreSQL adapter
   pip install psycopg2-binary
   ```

   Update `DATABASE_URL` in `.env`:
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/stock_prediction
   ```

3. **Collect static files:**
   ```bash
   python manage.py collectstatic
   ```

4. **Deploy using Gunicorn & Nginx or Heroku/PythonAnywhere**

## Important Notes ⚠️

- **Disclaimer**: Stock predictions are for educational purposes only. They should not be used as financial advice. Always do your own research before investing.
- **Model Limitations**: LSTM models are sensitive to market changes and may not capture sudden market events (crashes, pandemics, etc.)
- **Email Setup**: Gmail requires an app-specific password (not your regular password). [Generate one here](https://myaccount.google.com/apppasswords)
- **Database**: SQLite is fine for development but use PostgreSQL for production

## Troubleshooting

### Common Issues

**Issue: "Could not fetch data for stock"**
- Check if the stock ticker is valid
- Try with `.NS` suffix for Indian stocks (e.g., `INFY.NS`)
- Check your internet connection

**Issue: "Model not found"**
- The model trains automatically on first prediction
- First prediction may take 2-3 minutes

**Issue: Email not sending**
- Verify `EMAIL_HOST_USER` and `EMAIL_HOST_PASSWORD` in `.env`
- Check if Gmail 2-factor authentication is enabled
- Use app-specific password instead of regular password

**Issue: Port 8000 already in use**
```bash
python manage.py runserver 8001  # Use a different port
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

Created by: [Your Name]

## Support

For issues, feature requests, or questions, please open an issue on GitHub.

---

**Happy Trading! 📊💰**
=======
# StockVision
This project leverages machine learning techniques to analyze historical stock market data and predict future stock prices. By capturing trends and patterns, it provides valuable insights to help investors make informed decisions.
>>>>>>> 593b8d109d20f886aede44296687fbb0304eef7e
