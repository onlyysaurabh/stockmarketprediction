import logging
from django.core.management.base import BaseCommand
from stocks.services import fetch_and_store_commodity_data

# Configure logging for the command
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Fetches and stores historical data for specified commodities (Gold, Oil, Bitcoin).'

    # Define the commodities to update
    COMMODITIES = {
        'GC=F': 'Gold',
        'CL=F': 'Crude Oil',
        'BTC-USD': 'Bitcoin',
    }

    def handle(self, *args, **options):
        self.stdout.write("Starting commodity data update...")
        
        success_count = 0
        fail_count = 0

        for symbol, name in self.COMMODITIES.items():
            self.stdout.write(f"Processing {name} ({symbol})...")
            try:
                # Call the service function to fetch/store data
                # Using force_update=True ensures we get the latest data when the command is run
                result = fetch_and_store_commodity_data(symbol=symbol, name=name, force_update=True)
                if result:
                    self.stdout.write(self.style.SUCCESS(f"Successfully updated data for {name} ({symbol})."))
                    success_count += 1
                else:
                    # fetch_and_store_commodity_data returns None on failure
                    self.stdout.write(self.style.WARNING(f"Could not fetch or store data for {name} ({symbol}). Check logs."))
                    fail_count += 1
            except Exception as e:
                logger.error(f"Command error processing {name} ({symbol}): {e}", exc_info=True)
                self.stderr.write(self.style.ERROR(f"An unexpected error occurred for {name} ({symbol}): {e}"))
                fail_count += 1

        self.stdout.write(self.style.SUCCESS(f"Commodity data update finished. Success: {success_count}, Failed: {fail_count}"))
