from django.core.mail import send_mail
from django.conf import settings

def send_otp_email(email, otp_code, username):
    """Send OTP verification email to user"""
    subject = 'Verify Your Email Address'
    message = f"""
    Hello {username},
    
    Thank you for signing up! Your verification code is:
    
    {otp_code}
    
    This code will expire in 10 minutes. Please enter this code on the verification page.
    
    If you didn't request this code, please ignore this email.
    
    Best regards,
    Your App Team
    """
    
    from_email = settings.DEFAULT_FROM_EMAIL
    recipient_list = [email]
    
    send_mail(subject, message, from_email, recipient_list, fail_silently=False)