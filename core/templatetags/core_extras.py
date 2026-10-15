"""模板过滤器。"""
from django import template

register = template.Library()


@register.filter
def get_item(d, key):
    """字典 / QueryDict 取值（模板里 {{ dict|get_item:key }}）。"""
    if isinstance(d, dict):
        return d.get(key)
    try:
        return d[key]
    except (KeyError, TypeError, IndexError):
        return None
