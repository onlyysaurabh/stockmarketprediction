from django.contrib import messages
from django.http import HttpResponseRedirect
from django.template.response import TemplateResponse
from django.urls import reverse
from django.contrib.admin.models import LogEntry, ADDITION
from django.contrib.contenttypes.models import ContentType
from datetime import date, timedelta
import json

from .models import Stock, StockNews
from .forms import FetchNewsForm
from .services import process_news_for_stock

def fetch_stock_news_view(request, admin_site):
    if request.method == 'POST':
        form = FetchNewsForm(request.POST)
        if form.is_valid():
            stock_symbol = form.cleaned_data['stock_symbol']
            days = form.cleaned_data['days']
            
            try:
                stock = Stock.objects.get(symbol=stock_symbol)
                start_date = date.today() - timedelta(days=days)
                news_items = process_news_for_stock(stock, start_date)
                
                if news_items:
                    messages.success(request, f'Successfully fetched {len(news_items)} news items for {stock_symbol}')
                    
                    # Log the action
                    LogEntry.objects.log_action(
                        user_id=request.user.id,
                        content_type_id=ContentType.objects.get_for_model(StockNews).pk,
                        object_id=stock.id,
                        object_repr=stock_symbol,
                        action_flag=ADDITION,
                        change_message=json.dumps([{'added': {'name': 'Stock News', 
                                                            'object': f'{len(news_items)} news items'}}])
                    )
                else:
                    messages.warning(request, f'No new news items found for {stock_symbol}')
                    
                return HttpResponseRedirect(
                    reverse('admin:stocks_stocknews_changelist')
                )
            except Stock.DoesNotExist:
                messages.error(request, f'Stock {stock_symbol} not found')
            except Exception as e:
                messages.error(request, f'Error fetching news: {str(e)}')
                
    else:
        form = FetchNewsForm()
    
    context = {
        'form': form,
        'title': 'Fetch Stock News',
        **admin_site.each_context(request),
    }
    return TemplateResponse(request, 'admin/fetch_stock_news.html', context)
