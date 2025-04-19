from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Template filter to access dictionary items with a variable key.
    Usage: {{ my_dict|get_item:my_key }}
    """
    if dictionary is None:
        return None
    return dictionary.get(key)