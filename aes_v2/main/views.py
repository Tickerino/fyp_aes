# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.contrib.auth import login, authenticate
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from django.contrib.auth.models import User
from django.core.mail import EmailMessage

from .forms import *

from .tokens import account_activation_token

from models import Essays, Questions

from aes_function.layers import Conv1DMask, GatePositional, MaxPooling1DMask
from aes_function.layers import MeanOverTime
from aes_function.reader import process_essay, convert_to_dataset_friendly_scores
from keras.models import model_from_json
from aes_function.model import Model

model = Model()


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def home(request):
    if request.user.is_authenticated:
        return render(request, 'main/homed.html')
    else:
        return render(request, 'main/home.html')

    return HttpResponse("System Error!")

def signup(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            current_site = get_current_site(request)
            mail_subject = 'Activate your account.'
            message = render_to_string('main/acc_active_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uid':urlsafe_base64_encode(force_bytes(user.pk)),
                'token':account_activation_token.make_token(user),
            })
            to_email = form.cleaned_data.get('email')
            email = EmailMessage(
                        mail_subject, message, to=[to_email]
            )
            email.send()
            return HttpResponse('Please confirm your email address to complete the registration')
    else:
        form = SignupForm()
    return render(request, 'main/signup.html', {'form': form})

def activate(request, uidb64, token):
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except(TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.save()
        login(request, user)
        # return redirect('home')
        return HttpResponse('Thank you for your email confirmation. You are now logged in.')
    else:
        return HttpResponse('Activation link is invalid!')

def get_essay(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            form = EssayForm(request.POST)
            if form.is_valid():
                text_buffer = form.cleaned_data.get("essay_content").encode('utf-8')
                question_buffer = form.cleaned_data.get("question")
                result = model.calculate_score(text_buffer,question_buffer)
                if result >= 0:
                    Essays.objects.create(question = question_buffer, \
                    essay_content = text_buffer,\
                    predicted_score = result)
                    result = round(result,1)
                    return render(request,'main/result.html', {'result': result})
                else:
                    result = 0
                    Essays.objects.create(question = question_buffer, \
                    essay_content = text_buffer,\
                    predicted_score = result)
                    
                    return render(request,'main/result.html',{'result':'Sorry, this questions are not ready to be scored by system yet, but your submission will help the system developing.'})


        else:
            form = EssayForm()

        return render(request,'main/scoring.html',{'form':form})
    else:
        return render(request, 'main/home.html')
