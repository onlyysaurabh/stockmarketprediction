import time
from django.core.management.base import BaseCommand
from django.utils import timezone
from stocks.services import update_stock_prices
from stocks.models import Stock
from django.db.models import Q


class Command(BaseCommand):
    help = 'Update stock prices for all or selected stocks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--symbols',
            nargs='+',
            type=str,
            help='Specify stock symbols to update (e.g., AAPL MSFT GOOGL)'
        )
        parser.add_argument(
            '--delay',
            type=float,
            default=1.0,
            help='Delay in seconds between API calls (default: 1.0)'
        )
        parser.add_argument(
            '--skip-django',
            action='store_true',
            help='Skip updating Django models (update MongoDB only)'
        )
        parser.add_argument(
            '--report',
            action='store_true',
            help='Show detailed report of failures'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS(f"Starting stock price update at {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}")
        )
        
        symbols = options.get('symbols')
        delay = options.get('delay', 1.0)
        update_django = not options.get('skip_django', False)
        
        if not symbols:
            self.stdout.write(self.style.WARNING(f"Updating ALL stocks in database..."))
        else:
            self.stdout.write(self.style.WARNING(f"Updating {len(symbols)} selected stocks: {', '.join(symbols)}"))
        
        self.stdout.write(f"Using delay of {delay}s between API calls")
        
        # Call the robust update function
        start_time = time.time()
        results = update_stock_prices(
            symbols=symbols,
            delay=delay,
            update_django_model=update_django
        )
        
        # Calculate elapsed time and format it
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        # Print summary
        self.stdout.write("\n" + "="*40)
        self.stdout.write(self.style.SUCCESS(f"Update completed in {minutes}m {seconds}s"))
        self.stdout.write(f"Successfully updated: {results['success']}")
        
        if results['not_found'] > 0:
            self.stdout.write(self.style.WARNING(f"Not found/No data: {results['not_found']}"))
        
        if results['failed'] > 0:
            self.stdout.write(self.style.ERROR(f"Failed updates: {results['failed']}"))
            
            # Show detailed error report if requested
            if options.get('report') and results['errors']:
                self.stdout.write("\nError details:")
                for symbol, error in results['errors'].items():
                    self.stdout.write(self.style.ERROR(f"  {symbol}: {error}"))
        
        self.stdout.write("="*40)