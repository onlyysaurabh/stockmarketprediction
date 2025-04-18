import logging
from django.core.management.base import BaseCommand
from stocks.models import Stock
from stocks.news_service import fetch_and_store_news

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Fetches news for all stocks and stores in MongoDB'

    def add_arguments(self, parser):
        parser.add_argument(
            '--ticker',
            type=str,
            help='Specific ticker symbol to fetch news for (optional)',
        )
        parser.add_argument(
            '--days',
            type=int,
            default=7,
            help='Number of days back to fetch news for (default: 7)',
        )

    def handle(self, *args, **options):
        ticker = options.get('ticker')
        days_back = options.get('days', 7)
        
        if ticker:
            try:
                stock = Stock.objects.get(symbol=ticker.upper())
                self.stdout.write(f"Fetching news for {stock.symbol}...")
                processed = fetch_and_store_news(stock.symbol, days_back=days_back)
                if processed:
                    self.stdout.write(f"Processed {processed} news items for {stock.symbol}")
                else:
                    self.stdout.write(self.style.WARNING(f"No news items found for {stock.symbol}"))
            except Stock.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"Stock with symbol {ticker} does not exist"))
                return
        else:
            self.stdout.write("Fetching news for all stocks...")
            stocks = Stock.objects.all()
            total_processed = 0
            
            for stock in stocks:
                try:
                    self.stdout.write(f"Processing {stock.symbol}...")
                    processed = fetch_and_store_news(stock.symbol, days_back=days_back)
                    if processed:
                        total_processed += processed
                        self.stdout.write(f"Processed {processed} news items for {stock.symbol}")
                    else:
                        self.stdout.write(self.style.WARNING(f"No news items found for {stock.symbol}"))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error processing news for {stock.symbol}: {str(e)}"))
                    logger.error(f"Error processing news for {stock.symbol}: {str(e)}")
                    continue
            
            self.stdout.write(self.style.SUCCESS(f"Successfully processed {total_processed} news items for all stocks"))