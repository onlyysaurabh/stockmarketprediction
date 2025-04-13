from django import template
from django.utils.safestring import mark_safe
from datetime import datetime, timezone
import logging
import re

register = template.Library()
logger = logging.getLogger(__name__)

@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Custom template filter to allow accessing dictionary items using
    a variable key in Django templates.
    Usage: {{ my_dictionary|get_item:my_key_variable }}
    Returns None if the key doesn't exist or the input is not a dictionary.
    """
    if not isinstance(dictionary, dict):
        logger.warning(f"Attempted to use get_item filter on non-dict type: {type(dictionary)}")
        return None
    return dictionary.get(key)

@register.filter(name='format_epoch')
def format_epoch(epoch_timestamp, format_string="%Y-%m-%d %H:%M:%S %Z"):
    """Converts an epoch timestamp (seconds or ms) to a formatted datetime string."""
    if epoch_timestamp is None:
        return "N/A"
    try:
        # Attempt to convert, assuming seconds first. If it's way off, try ms.
        timestamp = float(epoch_timestamp)
        # Crude check for milliseconds (e.g., timestamp > 3 billion seconds is likely ms)
        if timestamp > 3e9:
            timestamp /= 1000.0
        dt_object = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt_object.strftime(format_string)
    except (ValueError, TypeError, OSError):
        logger.warning(f"Could not format epoch timestamp: {epoch_timestamp}")
        return str(epoch_timestamp) # Return original if conversion fails

@register.filter(name='format_percentage')
def format_percentage(value, decimal_places=2):
    """Formats a decimal value as a percentage string."""
    if value is None:
        return "N/A"
    try:
        percentage = float(value) * 100
        return f"{percentage:.{decimal_places}f}%"
    except (ValueError, TypeError):
        logger.warning(f"Could not format value as percentage: {value}")
        return str(value)

@register.filter(name='format_currency')
def format_currency(value, symbol='$', decimal_places=2):
    """Formats a numerical value as a currency string."""
    if value is None:
        return "N/A"
    try:
        amount = float(value)
        # Basic formatting, consider locale-specific formatting for production
        return f"{symbol}{amount:,.{decimal_places}f}"
    except (ValueError, TypeError):
        logger.warning(f"Could not format value as currency: {value}")
        return str(value)

@register.filter(name='format_summary')
def format_summary(text):
    """Replaces newline characters with <br> tags."""
    if not text or not isinstance(text, str):
        return text
    # Replace multiple newlines with a single <br><br> for paragraph breaks?
    # For now, just replace single newlines with <br>
    formatted_text = text.replace('\n', '<br>')
    return mark_safe(formatted_text) # Mark as safe to render HTML

@register.filter(name='format_label')
def format_label(key):
    """Converts a camelCase or snake_case key into a Title Case label."""
    if not key or not isinstance(key, str):
        return key
    # Add spaces before capital letters (CamelCase to Title Case)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', key)
    # Add spaces before capital letters followed by another capital (e.g., PEGRatio -> PEG Ratio)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    # Replace underscores with spaces
    s3 = s2.replace('_', ' ')
    # Capitalize words
    return s3.title()

@register.filter(name='parse_officers')
def parse_officers(officer_list):
    """Parses the companyOfficers list into a cleaner format for display."""
    if not isinstance(officer_list, list):
        return []

    parsed_officers = []
    for officer in officer_list:
        if isinstance(officer, dict):
            name = officer.get('name', 'N/A')
            # Normalize whitespace in name
            name = ' '.join(name.split())

            parsed_officers.append({
                'name': name,
                'title': officer.get('title', 'N/A'),
                'age': officer.get('age', 'N/A'),
                # Add other fields if needed, e.g., 'totalPay'
                # 'totalPay': officer.get('totalPay')
            })
    return parsed_officers

@register.filter(name='subtract')
def subtract(value, arg):
    """
    Subtracts the argument from the value.
    Usage: {{ value|subtract:arg }}
    """
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        logger.warning(f"Could not subtract values: {value} and {arg}")
        return 0

@register.filter(name='percent_change')
def percent_change(value, arg):
    """
    Calculates the percentage change between two values.
    Usage: {{ new_value|percent_change:old_value }}
    Returns the percentage change as (new - old) / old * 100
    """
    try:
        value = float(value)
        arg = float(arg)
        if arg == 0:  # Avoid division by zero
            return 0
        return ((value - arg) / arg) * 100
    except (ValueError, TypeError):
        logger.warning(f"Could not calculate percent change between values: {value} and {arg}")
        return 0
