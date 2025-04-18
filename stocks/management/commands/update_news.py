import logging
from django.core.management.base import BaseCommand
from stocks.models import Stock
from stocks.services import process_news_for_stock

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Fetches news for all stocks and analyzes sentiment'

    def add_arguments(self, parser):
        parser.add_argument(
            '--ticker',
            type=str,
            help='Specific ticker symbol to fetch news for (optional)',
        )

    def handle(self, *args, **options):
        ticker = options.get('ticker')
        
        if ticker:
            try:
                stock = Stock.objects.get(ticker=ticker.upper())
                self.stdout.write(f"Fetching news for {stock.ticker}...")
                processed = process_news_for_stock(stock)
                self.stdout.write(f"Processed {processed} news items for {stock.ticker}")
            except Stock.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"Stock with ticker {ticker} does not exist"))
                return
        else:
            self.stdout.write("Fetching news for all stocks...")
            stocks = Stock.objects.all()
            total_processed = 0
            
            for stock in stocks:
                try:
                    self.stdout.write(f"Processing {stock.ticker}...")
                    processed = process_news_for_stock(stock)
                    total_processed += processed
                    self.stdout.write(f"Processed {processed} news items for {stock.ticker}")
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error processing news for {stock.ticker}: {str(e)}"))
                    logger.error(f"Error processing news for {stock.ticker}: {str(e)}")
                    continue
            
            self.stdout.write(self.style.SUCCESS(f"Successfully processed {total_processed} news items for all stocks"))