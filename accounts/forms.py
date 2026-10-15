"""账户表单：注册。"""
from django import forms
from django.contrib.auth.models import User


class RegisterForm(forms.Form):
    """用户注册。

    层级说明：
    - 注册创建的均为普通用户（可上传、分析自己的作品）；
    - 管理员层级 = is_staff（可访问系统设置页与 Django 后台），
      由已有管理员在 Django 后台 (/admin/) 中勾选"员工状态"授予。
    """
    username = forms.CharField(label='用户名', max_length=30,
                               widget=forms.TextInput(attrs={'class': 'input'}))
    email = forms.EmailField(label='邮箱', required=False,
                             widget=forms.EmailInput(attrs={'class': 'input'}))
    password1 = forms.CharField(label='密码', widget=forms.PasswordInput(attrs={'class': 'input'}))
    password2 = forms.CharField(label='确认密码', widget=forms.PasswordInput(attrs={'class': 'input'}))

    def clean(self):
        data = super().clean()
        p1, p2 = data.get('password1'), data.get('password2')
        if p1 and p2 and p1 != p2:
            raise forms.ValidationError('两次输入的密码不一致。')
        if User.objects.filter(username=data.get('username')).exists():
            raise forms.ValidationError('用户名已存在。')
        return data

    def save(self):
        user = User.objects.create_user(
            username=self.cleaned_data['username'],
            email=self.cleaned_data.get('email') or '',
            password=self.cleaned_data['password1'],
        )
        return user
