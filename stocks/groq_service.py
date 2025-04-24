"""
Groq API service with API key rotation mechanism
"""
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# --- Groq API Configuration ---
GROQ_API_KEYS_VAR = "GROQ_API_KEYS"  # Environment variable name
GROQ_MODEL_NAME = os.environ.get('GROQ_MODEL_NAME', 'llama-3.3-70b-versatile')

class GroqClient:
    """
    Client for interacting with the Groq API, handling API key rotation.
    """
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        if not self.api_keys:
            # Fall back to single key if multiple keys aren't configured
            single_key = os.environ.get('GROQ_API_KEY')
            if single_key:
                self.api_keys = [single_key]
                logger.info("Using single Groq API key from GROQ_API_KEY environment variable.")
            else:
                logger.warning(f"No Groq API keys found in environment variables. API calls will fail.")

    def _load_api_keys(self) -> List[str]:
        """Loads API keys from the environment variable."""
        keys_str = os.environ.get(GROQ_API_KEYS_VAR)
        if keys_str:
            # Split by comma and remove whitespace
            return [key.strip() for key in keys_str.split(',') if key.strip()]
        return []

    def _get_next_key(self) -> Optional[str]:
        """Rotates to the next available API key."""
        if not self.api_keys:
            return None
        
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.debug(f"Using Groq API key index: {self.current_key_index}")  # Avoid logging the key itself
        return key

    def call_groq(self, prompt):
        """
        Makes a request to the Groq API, handling key rotation on failure.
        Returns the generated text or an error message.
        """
        if not self.api_keys:
            error_msg = "Cannot make Groq request: No API keys configured."
            logger.error(error_msg)
            return {'error': error_msg}

        # Try each key until one works or all fail
        initial_key_index = self.current_key_index
        attempts = 0
        max_attempts = len(self.api_keys)

        while attempts < max_attempts:
            api_key = self._get_next_key()
            if not api_key:
                error_msg = "Ran out of API keys during rotation attempt."
                logger.error(error_msg)
                return {'error': error_msg}

            try:
                from groq import Groq
                
                # Initialize Groq client with current key
                client = Groq(api_key=api_key)
                
                # Make the API request
                response = client.chat.completions.create(
                    model=GROQ_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specializing in financial analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9
                )
                
                # Extract the generated content
                if response.choices and len(response.choices) > 0:
                    generated_text = response.choices[0].message.content
                    logger.info(f"Groq request successful")
                    return generated_text
                else:
                    logger.error("Unexpected response format from Groq API")
                    # This could be a legitimate error, but we'll try the next key just in case
                    
            except ImportError:
                return {'error': 'Groq client library not installed. Run: pip install groq'}
            except Exception as e:
                error_str = str(e).lower()
                logger.warning(f"Error with Groq API key index {self.current_key_index-1}: {e}")
                
                # Check if the error is related to authentication or rate limiting
                if ("auth" in error_str or 
                    "rate" in error_str or 
                    "limit" in error_str or 
                    "unauthorized" in error_str or
                    "invalid" in error_str):
                    # These errors suggest we should try another key
                    logger.info(f"Trying next Groq API key due to auth/rate limit error")
                else:
                    # For other errors, it might not be key-related, but we'll still rotate just in case
                    logger.warning(f"Unexpected error from Groq API: {e}")
            
            attempts += 1
            # If we've tried all keys and looped back to the start, stop.
            if self.current_key_index == initial_key_index and attempts > 0:
                logger.error(f"All Groq API keys failed.")
                break  # Avoid infinite loop if all keys fail immediately

        error_msg = f"Failed to get response from Groq API after {attempts} attempts with different keys."
        logger.error(error_msg)
        return {'error': error_msg}

# Create a singleton instance
groq_client = GroqClient()
