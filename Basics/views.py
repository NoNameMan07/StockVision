# Create your views here.
from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.cache import cache  # To temporarily store OTP
from django.core.mail import send_mail
from django.conf import settings
import random
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.core.exceptions import ValidationError as EmailValidationError
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.decorators import login_required
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def generate_otp():
    """Generates a 6-digit OTP"""
    return str(random.randint(100000, 999999))

def send_otp_email(email, otp):
    """Sends OTP to user's email"""
    subject = "Your OTP for Signup Verification"
    message = f"Your OTP is {otp}. Please enter this code to verify your signup."
    send_mail(subject, message, settings.EMAIL_HOST_USER, [email])

def signup(request):
    """
    Handle user registration.
    
    Validates form data, creates a new user account if validation passes,
    and automatically logs in the new user.
    """
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        # Form validation
        error = False
        
        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            error = True
            
        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists")
            error = True
            
        # Check if passwords match
        if password1 != password2:
            messages.error(request, "Passwords do not match")
            error = True
            
            
        # If no errors, create user
        if not error:
            # Create user
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password1
            )
            
            # Log the user in
            user = authenticate(username=username, password=password1)
            if user is not None:
                auth_login(request, user)  # Using auth_login to avoid name conflict
                messages.success(request, f"Welcome {username}! Your account has been created successfully.")
                return redirect('predict')  # Redirect to predict page after signup
        
    # If GET request or form had errors, render the signup page
    return render(request, 'signup.html')
def about(request):
    return render(request, 'about.html')

def new(request):
    return render(request, 'new.html')

def home(request):
   return render(request, 'home.html')
def logout(request):
    return redirect(login)

def login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")  # Corrected to match form input
        
        # Authenticate user
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # If credentials are correct, log in the user
            auth_login(request, user)
            
            # Redirect to predict page
            return redirect("predict")
        else:
            # If credentials are incorrect, show error message
            messages.error(request, "Invalid username or password. Please try again.")
    
    # Render login page (for GET requests or failed login attempts)
    return render(request, "login.html")

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from .models import OTP
from .utils import send_otp_email

def signup(request):
    '''
    Handle user registration with email verification.
    
    Validates form data, creates an inactive user account if validation passes,
    sends a verification email with OTP, and redirects to the verification page.
    '''
    if request.method == 'POST':
        # Retrieve form data
        username = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        # Initialize error tracking
        error = False
        
        # Validate username
        if not username:
            messages.error(request, "Username is required")
            error = True
        elif User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            error = True
        
        # Validate email
        try:
            # Validate email format
            validate_email(email)
            
            # Check if email already exists
            if User.objects.filter(email=email).exists():
                messages.error(request, "Email is already registered")
                error = True
        except EmailValidationError:
            messages.error(request, "Invalid email format")
            error = True
        
        # Validate passwords
        if not password1 or not password2:
            messages.error(request, "Both password fields are required")
            error = True
        elif password1 != password2:
            messages.error(request, "Passwords do not match")
            error = True
        
        # Validate password strength
        try:
            validate_password(password1)
        except ValidationError as e:
            for error_message in e.messages:
                messages.error(request, error_message)
            error = True
        
        # If no errors, create user
        if not error:
            try:
                # Create user
                user = User.objects.create_user(
                    username=username,
                    email=email,
                    password=password1
                )
                
                # Add success message
                messages.success(request, "Account created successfully. Please log in.")
                
                # Redirect to login page
                return redirect('login')
            
            except Exception as e:
                # Catch any unexpected errors during user creation
                messages.error(request, f"An error occurred: {str(e)}")
        
        # If there are errors, re-render the signup page
        return render(request, 'signup.html')
    
    # Handle GET request
    return render(request, 'signup.html')
"""
def verify_email(request):
    
    Handle email verification with OTP.
    
    Validates the OTP entered by the user, activates the account if OTP is valid,
    and logs in the user.
    
    user_id = request.session.get('user_id_to_verify')
    
    # Redirect to signup if no user_id in session
    if not user_id:
        messages.error(request, "Signup process interrupted. Please try again.")
        return redirect('signup')
    
    if request.method == 'POST':
        otp_entered = request.POST.get('otp')
        
        try:
            user = User.objects.get(id=user_id)
            otp_obj = OTP.objects.get(user=user)
            
            # Check if OTP is expired
            if otp_obj.is_expired():
                # Generate and send new OTP
                otp_code = otp_obj.generate_otp()
                send_otp_email(user.email, otp_code, user.username)
                messages.error(request, "OTP expired. A new verification code has been sent to your email.")
                return render(request, 'verify_email.html')
            
            # Verify OTP
            if otp_obj.otp_code == otp_entered:
                # Activate user account
                user.is_active = True
                user.save()
                
                # Mark as verified
                otp_obj.is_verified = True
                otp_obj.save()
                
                # Clear session
                if 'user_id_to_verify' in request.session:
                    del request.session['user_id_to_verify']
                
                # Log in the user
                auth_login(request, user)
                messages.success(request, f"Welcome {user.username}! Your email has been verified successfully.")
                return redirect('predict')  # Redirect to predict page after verification
            else:
                messages.error(request, "Invalid verification code. Please try again.")
        except (User.DoesNotExist, OTP.DoesNotExist):
            messages.error(request, "Invalid verification attempt. Please try signing up again.")
            return redirect('signup')
    
    return render(request, 'verify_email.html')

def resend_otp(request):
    Resend OTP to user's email address
    user_id = request.session.get('user_id_to_verify')
    
    if not user_id:
        messages.error(request, "Signup process interrupted. Please try again.")
        return redirect('signup')
    
    try:
        user = User.objects.get(id=user_id)
        otp_obj = OTP.objects.get(user=user)
        
        # Generate new OTP and send
        otp_code = otp_obj.generate_otp()
        send_otp_email(user.email, otp_code, user.username)
        
        messages.success(request, f"A new verification code has been sent to {user.email}")
    except (User.DoesNotExist, OTP.DoesNotExist):
        messages.error(request, "Invalid request. Please try signing up again.")
        return redirect('signup')
    
    return redirect('verify_email')
"""
import os

# Ensure correct paths based on the new project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Stock Prediction","models", "stock_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "Stock Prediction","models", "scaler.pkl")


@login_required
def calcy(request):

    if(request.method=="POST"):
        data=request.POST
        n1 = request.POST.get('1num')
        n2 = request.POST.get('2num')
        if not n1 or not n2:
            return render(request, 'calcy.html', context={'RESULT': 'Please enter both numbers.'})
        
        else:
            n1 = float(n1) 
            n2 = float(n2)
            if 'add'in request.POST:
                return render(request,'calcy.html',context={'RESULT':f"The sum {n1+n2}"})
            
            elif 'sub'in request.POST:
                return render(request,'calcy.html',context={'RESULT':f"The difference {n1-n2}"})
            
            elif 'mul'in request.POST:
                return render(request,'calcy.html',context={'RESULT':f"The product {n1*n2}"})

            elif 'div'in request.POST:
                return render(request,'calcy.html',context={'RESULT':f"The quoitent {n1/n2}"})
    
    return render(request, 'calcy.html' )
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Prevents Matplotlib GUI issues in Django
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pickle
from django.conf import settings
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# Paths for saving model & scaler
import os
"""
# Ensure correct paths based on the new project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Stock Prediction","models", "stock_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "Stock Prediction","models", "scaler.pkl")
import os
import json
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
from nsepy import get_history
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse


# Custom callback to track training progress
class EpochProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        
    def on_epoch_end(self, epoch, logs=None):
        progress_data = {
            'current_epoch': epoch + 1,
            'total_epochs': self.params['epochs'],
            'loss': float(logs['loss']),
            'val_loss': float(logs['val_loss']) if 'val_loss' in logs else 0,
            'mae': float(logs['mae']),
            'val_mae': float(logs['val_mae']) if 'val_mae' in logs else 0
        }
        progress_path = os.path.join(settings.MEDIA_ROOT, "training_progress.json")
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f)

def train_lstm(x_train, y_train):
    """ Train an LSTM model """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    progress_callback = EpochProgressCallback()
    history = model.fit(
        x_train, y_train, 
        epochs=8, 
        batch_size=32, 
        validation_split=0.2, 
        verbose=1,
        callbacks=[progress_callback]
    )
    model.save(MODEL_PATH)
    return model, history

def fetch_stock_data(stock_name):
    """ Fetch historical stock data with fallback to Indian NSE, return data and currency """
    # Try yfinance with raw stock name (global stocks, assume USD)
    try:
        data = yf.download(stock_name, start="2020-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        if not data.empty:
            return data, "$"
    except Exception as e:
        print(f"yfinance failed for {stock_name}: {str(e)}")

    # Try yfinance with .NS suffix (Indian stocks, assume INR)
    try:
        ns_stock_name = f"{stock_name}.NS"
        data = yf.download(ns_stock_name, start="2020-01-01", end=datetime.now().strftime('%Y-%m-%d'))
        if not data.empty:
            return data, "₹"
    except Exception as e:
        print(f"yfinance failed for {ns_stock_name}: {str(e)}")

    # Fallback to nsepy for Indian stocks (assume INR)
    try:
        end_date = datetime.now()
        start_date = datetime.strptime("2020-01-01", '%Y-%m-%d')
        data = get_history(symbol=stock_name, start=start_date, end=end_date)
        if not data.empty:
            data = data.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            data.index = pd.to_datetime(data.index)
            return data, "₹"
    except Exception as e:
        print(f"nsepy failed for {stock_name}: {str(e)}")
    
    return None, None

def get_buy_sell_recommendation(historical_prices, predicted_prices):
    """Generate buy/sell recommendation based on predicted trend"""
    last_close = historical_prices[-1]
    avg_predicted = np.mean(predicted_prices)
    pct_change = ((avg_predicted - last_close) / last_close) * 100
    
    if pct_change > 3:
        recommendation = "Strong Buy"
        confidence = "High"
    elif pct_change > 1:
        recommendation = "Buy"
        confidence = "Medium"
    elif pct_change < -3:
        recommendation = "Strong Sell"
        confidence = "High"
    elif pct_change < -1:
        recommendation = "Sell"
        confidence = "Medium"
    else:
        recommendation = "Hold"
        confidence = "Low"
    
    return recommendation, confidence, pct_change

def calculate_model_accuracy(model, x_test, y_test, scaler):
    """Calculate model accuracy metrics"""
    predictions = model.predict(x_test)
    dummy_y = np.zeros((len(y_test), scaler.scale_.shape[0]))
    dummy_y[:, 0] = y_test
    dummy_pred = np.zeros((len(predictions), scaler.scale_.shape[0]))
    dummy_pred[:, 0] = predictions.flatten()
    y_test_inv = scaler.inverse_transform(dummy_y)[:, 0]
    predictions_inv = scaler.inverse_transform(dummy_pred)[:, 0]
    
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    mse = mean_squared_error(y_test_inv, predictions_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, predictions_inv)
    mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
    
    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2': round(r2, 2),
        'mape': round(mape, 2)
    }

def training_progress(request):
    """AJAX endpoint to get training progress"""
    progress_path = os.path.join(settings.MEDIA_ROOT, "training_progress.json")
    try:
        with open(progress_path, 'r') as f:
            data = json.load(f)
        return JsonResponse(data)
    except:
        return JsonResponse({'error': 'No training in progress'}, status=404)

def predict(request):
    """ Handle stock price prediction request """
    context = {}
    
    if request.method == "POST":
        stock_name = request.POST.get("stock_name", "").strip().upper()
        if not stock_name:
            context["error_message"] = "Stock name is required!"
            return render(request, "predict.html", context)

        # Fetch stock data with currency
        data, currency = fetch_stock_data(stock_name)
        if data is None:
            context["error_message"] = f"Invalid Stock Code! Could not fetch data for {stock_name}."
            return render(request, "predict.html", context)

        try:
            # Feature Engineering
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['Volatility'] = data['Close'].rolling(window=20).std()
            data = data[['Close', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'Volatility']].dropna()

            # Data Scaling
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
            with open(SCALER_PATH, "wb") as f:
                pickle.dump(scaler, f)

            # Prepare training data
            sequence_length = 60
            x_data, y_data = [], []
            for i in range(sequence_length, len(scaled_data)):
                x_data.append(scaled_data[i-sequence_length:i])
                y_data.append(scaled_data[i, 0])

            x_data, y_data = np.array(x_data), np.array(y_data)
            train_size = int(len(x_data) * 0.8)
            x_train, x_test = x_data[:train_size], x_data[train_size:]
            y_train, y_test = y_data[:train_size], y_data[train_size:]

            # Train or Load Model
            if os.path.exists(MODEL_PATH):
                model = load_model(MODEL_PATH)
                accuracy_metrics = calculate_model_accuracy(model, x_test, y_test, scaler)
            else:
                model, history = train_lstm(x_train, y_train)
                accuracy_metrics = calculate_model_accuracy(model, x_test, y_test, scaler)

            # Prediction Logic (Next Week)
            input_data = scaled_data[-sequence_length:]
            predicted_values = []
            future_dates = []

            for i in range(7):
                pred_input = input_data.reshape(1, sequence_length, scaled_data.shape[1])
                prediction = model.predict(pred_input, verbose=0)[0][0]
                predicted_values.append(prediction)
                input_data = np.roll(input_data, -1, axis=0)
                input_data[-1, 0] = prediction
                next_date = data.index[-1] + timedelta(days=i+1)
                while next_date.weekday() > 4:
                    next_date += timedelta(days=1)
                future_dates.append(next_date)

            predicted_values = np.array(predicted_values).reshape(-1, 1)
            dummy_array = np.zeros((len(predicted_values), scaled_data.shape[1]))
            dummy_array[:, 0] = predicted_values.flatten()
            predicted_prices = scaler.inverse_transform(dummy_array)[:, 0]

            # Generate Buy/Sell recommendation
            recommendation, confidence, pct_change = get_buy_sell_recommendation(
                data['Close'].values[-5:], 
                predicted_prices
            )

            # Plot Predictions
            plt.figure(figsize=(12, 6))
            plt.plot(data.index[-90:], data['Close'].values[-90:], label="Actual Prices", linewidth=2)
            plt.plot(future_dates, predicted_prices, linestyle='dashed', color='g', marker='o', label="Predicted Prices")
            plt.title(f"{stock_name} Stock Price Prediction")
            plt.xlabel("Date")
            plt.ylabel(f"Price ({currency})")
            plt.legend()
            plt.grid()
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            plot_path = os.path.join(settings.MEDIA_ROOT, "plot.png")
            plt.savefig(plot_path)
            plt.close()

            plot_url = f"{settings.MEDIA_URL}plot.png"

            # Ensure scalar values for rounding
            latest_price = float(data['Close'].values[-1])
            avg_predicted = float(np.mean(predicted_prices))

            context = {
                "stock_name": stock_name,
                "image_url": plot_url,
                "accuracy_metrics": accuracy_metrics,
                "recommendation": recommendation,
                "confidence": confidence,
                "pct_change": round(float(pct_change), 2),
                "latest_price": round(latest_price, 2),
                "avg_predicted": round(avg_predicted, 2),
                "currency": currency  # Pass currency to template
            }
            
            return render(request, "predict.html", context)
            
        except Exception as e:
            context["error_message"] = f"An error occurred: {str(e)}"
            return render(request, "predict.html", context)
    
    return render(request, "predict.html", {})