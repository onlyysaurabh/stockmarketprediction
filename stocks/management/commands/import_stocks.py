import csv
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from stocks.models import Stock
from stocks.services import update_stock_prices
from django.db import transaction

class Command(BaseCommand):
    help = 'Import stocks from the stock_symbols.csv file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Number of stocks to import (default: all)'
        )
        parser.add_argument(
            '--popular',
            action='store_true',
            help='Import only popular stocks (pre-defined list)'
        )
        parser.add_argument(
            '--update',
            action='store_true',
            help='Update prices after importing'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Batch size for bulk operations (default: 50)'
        )
        parser.add_argument(
            '--update-batch',
            type=int,
            default=10,
            help='Number of stocks to update in each batch (default: 10)'
        )

    def handle(self, *args, **options):
        limit = options.get('limit')
        use_popular = options.get('popular')
        update_after_import = options.get('update')
        batch_size = options.get('batch_size')
        update_batch_size = options.get('update_batch')
        
        csv_path = os.path.join(settings.BASE_DIR, 'stock_symbols.csv')
        
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f"CSV file not found at {csv_path}"))
            return
            
        # Define a list of popular stocks
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'NFLX',
            'INTC', 'PYPL', 'CSCO', 'ADBE', 'CMCSA', 'PEP', 'AVGO', 'COST', 'QCOM',
            'TXN', 'TMUS', 'CHTR', 'AMGN', 'SBUX', 'INTU', 'AMAT', 'MDLZ', 'BKNG',
            'ADI', 'MRNA', 'ISRG', 'GILD', 'ADP', 'REGN', 'LRCX', 'MU', 'CSX',
            'VRTX', 'FISV', 'ATVI', 'ADSK', 'BIDU', 'ILMN', 'WDAY', 'MAR', 'MNST',
            'KDP', 'MELI', 'KLAC', 'NXPI', 'ASML', 'JPM', 'V', 'DIS', 'BAC', 'KO',
            'PG', 'XOM', 'WMT', 'JNJ', 'CVX', 'HD', 'PFE', 'VZ', 'T', 'MRK'
        ]
        
        imported_count = 0
        existing_count = 0
        symbols_to_update = []
        stocks_to_create = []
        
        try:
            # Get existing symbols for faster lookup
            existing_symbols = set(Stock.objects.values_list('symbol', flat=True))
            self.stdout.write(f"Found {len(existing_symbols)} existing stocks in database")
            
            # Read the CSV file
            self.stdout.write("Reading stock symbols from CSV file...")
            with open(csv_path, 'r') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    symbol = row.get('Symbol')
                    name = row.get('Name')
                    
                    if not symbol:
                        continue
                    
                    # Skip if not in popular list when using --popular
                    if use_popular and symbol not in popular_stocks:
                        continue
                    
                    # Check if stock already exists
                    if symbol in existing_symbols:
                        existing_count += 1
                        continue
                    
                    # Add to batch for bulk creation
                    stocks_to_create.append(Stock(symbol=symbol, name=name))
                    symbols_to_update.append(symbol)
                    imported_count += 1
                    
                    # Stop if we've reached the limit
                    if limit and imported_count >= limit:
                        break
                    
                    # Process in batches
                    if len(stocks_to_create) >= batch_size:
                        self.stdout.write(f"Creating batch of {len(stocks_to_create)} stocks...")
                        with transaction.atomic():
                            Stock.objects.bulk_create(stocks_to_create)
                        stocks_to_create = []
            
            # Create any remaining stocks
            if stocks_to_create:
                self.stdout.write(f"Creating final batch of {len(stocks_to_create)} stocks...")
                with transaction.atomic():
                    Stock.objects.bulk_create(stocks_to_create)
            
            self.stdout.write(self.style.SUCCESS(f"\nImported {imported_count} new stocks"))
            self.stdout.write(f"Skipped {existing_count} existing stocks")
            
            # Update prices if requested, in batches
            if update_after_import and symbols_to_update:
                self.stdout.write("\nUpdating prices for newly imported stocks...")
                total_batches = (len(symbols_to_update) + update_batch_size - 1) // update_batch_size
                
                success_count = 0
                failed_count = 0
                not_found_count = 0
                
                for i in range(0, len(symbols_to_update), update_batch_size):
                    batch = symbols_to_update[i:i+update_batch_size]
                    batch_num = i // update_batch_size + 1
                    
                    self.stdout.write(f"Updating batch {batch_num}/{total_batches} ({len(batch)} stocks)...")
                    
                    results = update_stock_prices(symbols=batch, delay=1.0)
                    
                    success_count += results['success']
                    failed_count += results['failed']
                    not_found_count += results['not_found']
                
                self.stdout.write(self.style.SUCCESS(
                    f"Updated prices for {success_count} stocks\n"
                    f"Failed: {failed_count}, Not found: {not_found_count}"
                ))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing stocks: {e}"))