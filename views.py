from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.views import View
from django.contrib.auth.decorators import login_required 
from django.contrib.auth import logout as auth_logout
import numpy as np
import joblib
from .forms import RegisterForm, LoginForm, UpdateUserForm, UpdateProfileForm
import base64
from io import BytesIO
import seaborn as sns

import matplotlib.pyplot as plt
from django.http import JsonResponse

import tensorflow 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder





def home(request):
    return render(request, 'users/home.html')

@login_required(login_url='users-register')


def index(request):
    return render(request, 'app/index.html')

class RegisterView(View):
    form_class = RegisterForm
    initial = {'key': 'value'}
    template_name = 'users/register.html'

    def dispatch(self, request, *args, **kwargs):
        # will redirect to the home page if a user tries to access the register page while logged in
        if request.user.is_authenticated:
            return redirect(to='/')

        # else process dispatch as it otherwise normally would
        return super(RegisterView, self).dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)

        if form.is_valid():
            form.save()

            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}')

            return redirect(to='login')

        return render(request, self.template_name, {'form': form})


# Class based view that extends from the built in login view to add a remember me functionality

class CustomLoginView(LoginView):
    form_class = LoginForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')

        if not remember_me:
            # set session expiry to 0 seconds. So it will automatically close the session after the browser is closed.
            self.request.session.set_expiry(0)

            # Set session as modified to force data updates/cookie to be saved.
            self.request.session.modified = True

        # else browser session will be as long as the session cookie time "SESSION_COOKIE_AGE" defined in settings.py
        return super(CustomLoginView, self).form_valid(form)


class ResetPasswordView(SuccessMessageMixin, PasswordResetView):
    template_name = 'users/password_reset.html'
    email_template_name = 'users/password_reset_email.html'
    subject_template_name = 'users/password_reset_subject'
    success_message = "We've emailed you instructions for setting your password, " \
                      "if an account exists with the email you entered. You should receive them shortly." \
                      " If you don't receive an email, " \
                      "please make sure you've entered the address you registered with, and check your spam folder."
    success_url = reverse_lazy('users-home')


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'users/change_password.html'
    success_message = "Successfully Changed Your Password"
    success_url = reverse_lazy('users-home')

from .models import Profile

def profile(request):
    user = request.user
    # Ensure the user has a profile
    if not hasattr(user, 'profile'):
        Profile.objects.create(user=user)
    
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)
        profile_form = UpdateProfileForm(request.POST, request.FILES, instance=request.user.profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile is updated successfully')
            return redirect(to='users-profile')
    else:
        user_form = UpdateUserForm(instance=request.user)
        profile_form = UpdateProfileForm(instance=request.user.profile)

    return render(request, 'users/profile.html', {'user_form': user_form, 'profile_form': profile_form})


from .models import Prediction
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = joblib.load('E:/SARAVANAN-2024-2025/NEW TITLES-2024/NATURAL LANGUAGE PROCESSING/ITPNP02/Deploy/Project/App/VECTOR.pkl')
model = joblib.load('E:/SARAVANAN-2024-2025/NEW TITLES-2024/NATURAL LANGUAGE PROCESSING/ITPNP02/Deploy/Project/App/EMOTION.pkl')

def Deploy(request): 
    if request.method == "POST":
        int_features = [x for x in request.POST.values()]
        input_text2 = int_features[1:]
        print(input_text2)

        if isinstance(input_text2[0], str):
            result = input_text2[0]
        else:
            result = None

        print(result)

        def preprocess_text(text):    
            if pd.isnull(text):
                return ""
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            stop_words = set(stopwords.words('english'))
            words = [word for word in word_tokenize(text) if word not in stop_words]
            ps = PorterStemmer()
            words = [ps.stem(word) for word in words]
            preprocessed_text = ' '.join(words)
            return preprocessed_text

        preprocessed_input = preprocess_text(result)

        input_features = vectorizer.transform([preprocessed_input])

        predicted_class = model.predict(input_features)[0]

        
        if predicted_class == 0:
            predicted_class = "anger"
        elif predicted_class == 1:
            predicted_class = "disgust"
        elif predicted_class == 2:
            predicted_class = "fear"
        elif predicted_class == 3:
            predicted_class = "joy"
        elif predicted_class == 4:
            predicted_class = "neutral"
        elif predicted_class == 5:
            predicted_class = "sadness"
        elif predicted_class == 6:
            predicted_class = "shame"
        elif predicted_class == 7:
            predicted_class = "surprise"
        
        print(f"Predicted Label: {predicted_class}")
        Prediction.objects.create(input_text=preprocessed_input, output_label=predicted_class)
        
        return render(request, 'app/output.html', {"prediction_text":predicted_class})
    else:
        return render (request, 'app/Deploy.html')

def database(request):
    recent_predictions = Prediction.objects.all().order_by('-created_at')[:50]  # Get the 10 most recent predictions
    return render(request, 'app/database.html', {"recent_predictions": recent_predictions})




def report(request):
    return render(request, 'app/report.html')



def logout_view(request):  
    auth_logout(request)
    return redirect('/')