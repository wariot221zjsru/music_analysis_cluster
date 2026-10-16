from django import forms


class UploadForm(forms.Form):
    title = forms.CharField(label='作品标题', max_length=200, required=False,
                            widget=forms.TextInput(
                                attrs={'class': 'input',
                                       'placeholder': '留空则使用文件名'}))
    source_file = forms.FileField(label='音乐文件',
                                  widget=forms.ClearableFileInput(
                                      attrs={'class': 'input'}))
    contains_voice = forms.BooleanField(
        label='音频含人声（MuScriptor 将走"器乐+人声"两遍转录管线）',
        required=False)
    is_public = forms.BooleanField(
        label='公开该作品（其他登录用户可查看其分析结果）',
        required=False)
