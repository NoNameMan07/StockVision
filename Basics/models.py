from django.db import models

# Create your models here.
from django.contrib.auth.models import User
import random
from datetime import datetime, timedelta

class OTP(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    otp_code = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now_add=True)
    is_verified = models.BooleanField(default=False)
    
    def generate_otp(self):
        """Generate a 6-digit OTP code"""
        self.otp_code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        self.save()
        return self.otp_code
    
    def is_expired(self):
        """Check if OTP is expired (valid for 10 minutes)"""
        time_difference = datetime.now().astimezone() - self.created_at
        return time_difference > timedelta(minutes=10)
