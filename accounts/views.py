"""账户视图：注册 / 登录 / 登出。"""
from django.contrib.auth import login, logout
from django.contrib.auth.views import LoginView
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from .forms import RegisterForm


def register_view(request):
    if request.user.is_authenticated:
        return redirect('/')
    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect('/')
    return render(request, 'accounts/register.html', {'form': form})


class CustomLoginView(LoginView):
    template_name = 'registration/login.html'


@require_http_methods(['POST'])
def logout_view(request):
    logout(request)
    return redirect('/accounts/login/')
