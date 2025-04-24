import json
import logging
import os
import requests
from django.conf import settings
from django.core.cache import cache
from .services import get_stock_data, get_model_predictions
from .sentiment_service import get_stock_sentiment
from .news_service import get_news_for_stock

# Set up logging
logger = logging.getLogger(__name__)

# Check for environment variables
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
# llama-cpp-python config
LLAMA_CPP_MODEL_PATH = os.environ.get('LLAMA_CPP_MODEL_PATH', 'SmolLM-135M.Q8_0.gguf')
LLAMA_CPP_N_THREADS = int(os.environ.get('LLAMA_CPP_N_THREADS', '4'))
LLAMA_CPP_N_CONTEXT = int(os.environ.get('LLAMA_CPP_N_CONTEXT', '4096'))
LLAMA_CPP_N_GPU_LAYERS = int(os.environ.get('LLAMA_CPP_N_GPU_LAYERS', '0'))
# Groq configuration and API keys are now handled in groq_service.py
DEFAULT_AI_ANALYZER = os.environ.get('DEFAULT_AI_ANALYZER', 'gemini')

# Global LLM model cache to avoid reloading
_llm_model = None

def get_ai_stock_analysis(symbol, analyzer_choice=None, collected_data=None):
    """
    Get AI-powered analysis for a stock symbol
    
    Parameters:
    - symbol: The stock symbol to analyze
    - analyzer_choice: Which AI model to use ('gemini', 'llama', 'local', or 'groq'), defaults to environment setting
    - collected_data: Pre-collected data from the frontend (optional)
    
    Returns:
    - Dictionary containing analysis or error message
    """
    # Use the specified analyzer or fall back to default
    analyzer = analyzer_choice or DEFAULT_AI_ANALYZER
    
    # Map 'local' to 'llama' for backward compatibility
    if analyzer == 'local':
        analyzer = 'llama'
    
    # Check cache first (only if not using pre-collected data)
    if not collected_data:
        cache_key = f"ai_analysis_{symbol}_{analyzer}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result
    
    # Process analysis request
    try:
        # Initialize variables
        stock_data = {}
        prediction_details = "No prediction data available"
        sentiment_details = "No sentiment data available"
        news_summary = "No recent news available"
        
        if collected_data:
            # Use data collected from the frontend
            stock_data = {
                'symbol': symbol,
                'price': collected_data.get('currentPrice'),
                'name': collected_data.get('name', symbol)
            }
            
            # Extract key metrics from collected data
            key_metrics = collected_data.get('keyMetrics', {})
            for key, value in key_metrics.items():
                # Convert key to snake_case for consistency
                normalized_key = key.lower().replace(' ', '_')
                stock_data[normalized_key] = value
            
            # Format prediction data from collected data
            predictions = collected_data.get('predictions', {})
            if predictions:
                prediction_lines = []
                for model, pred_data in predictions.items():
                    model_preds = []
                    if isinstance(pred_data, dict):
                        for timeframe, data in pred_data.items():
                            if timeframe != 'dates':  # Skip the dates dictionary
                                if isinstance(data, dict):  # Handle structured data format
                                    price = data.get('price')
                                    date = data.get('date')
                                    if price and date:
                                        model_preds.append(f"{timeframe.replace('_', ' ').title()} ({date}): ${price:.2f}")
                                elif isinstance(data, (int, float)):  # Handle numeric data
                                    date = pred_data.get('dates', {}).get(timeframe, '')
                                    model_preds.append(f"{timeframe.replace('_', ' ').title()} ({date}): ${data:.2f}")
                    
                    if model_preds:
                        prediction_lines.append(f"- {model.upper()}: {'; '.join(model_preds)}")
                
                if prediction_lines:
                    prediction_details = "Price Predictions:\n" + "\n".join(prediction_lines)
            
            # Format sentiment data from collected data
            sentiment = collected_data.get('sentiment', {})
            if sentiment:
                pos = sentiment.get('positive', 0)
                neu = sentiment.get('neutral', 0)
                neg = sentiment.get('negative', 0)
                sentiment_details = f"News Sentiment Distribution: Positive {pos:.1f}%, Neutral {neu:.1f}%, Negative {neg:.1f}%"
            
            # Format news data from collected data
            news_headlines = []
            for news_item in collected_data.get('newsHeadlines', []):
                if isinstance(news_item, dict):
                    headline = news_item.get('headline')
                    if headline:
                        news_headlines.append(headline)
                else:
                    news_headlines.append(news_item)
            
            if news_headlines:
                news_summary = "Recent headlines: " + "; ".join(news_headlines[:5])
        else:
            # Use backend services to fetch data (original implementation)
            # Get technical stock data
            stock_data = get_stock_data(symbol)
            if not stock_data or 'error' in stock_data:
                return {'error': f"Could not fetch stock data for {symbol}"}
            
            # Get prediction data
            prediction_data = get_model_predictions(symbol)
            if prediction_data and 'error' not in prediction_data:
                # Format predictions in a more detailed way for the AI
                models = prediction_data.get('model_types', [])
                predictions = prediction_data.get('predictions', {})
                dates = prediction_data.get('prediction_dates', {})
                if models and predictions:
                    prediction_lines = []
                    for model in models:
                        model_preds = []
                        for timeframe in ['next_day', 'next_week', 'next_month']:
                            pred_val = predictions.get(timeframe, {}).get(model)
                            pred_date = dates.get(timeframe)
                            if pred_val is not None and pred_date:
                                model_preds.append(f"{timeframe.replace('_', ' ').title()} ({pred_date}): ${pred_val:.2f}")
                        if model_preds:
                             prediction_lines.append(f"- {model.upper()}: {'; '.join(model_preds)}")
                    if prediction_lines:
                        prediction_details = "Price Predictions:\n" + "\n".join(prediction_lines)
                    else:
                        prediction_details = "Prediction models available but no specific forecast data found."
                elif models:
                     prediction_details = "Prediction models available but no specific forecast data found."
    
            # Get sentiment data
            sentiment_data = get_stock_sentiment(symbol)
            if sentiment_data and 'error' not in sentiment_data:
                # Include distribution percentages
                pos = sentiment_data.get('positive_percentage', 0)
                neu = sentiment_data.get('neutral_percentage', 0)
                neg = sentiment_data.get('negative_percentage', 0)
                sentiment_details = f"News Sentiment Distribution: Positive {pos:.1f}%, Neutral {neu:.1f}%, Negative {neg:.1f}% (Based on {sentiment_data.get('total_analyzed', 'N/A')} articles)"
    
            # Get recent news
            news_data = get_news_for_stock(symbol, limit=5)
            if news_data and 'error' not in news_data:
                news_headlines = [news.get('title', 'Unknown headline') for news in news_data[:5]]
                news_summary = "Recent headlines: " + "; ".join(news_headlines)
        
        # Construct prompt for AI
        prompt = f"""
        Analyze the following stock information for {symbol} and provide investment insights. Focus on the provided data points.

        Key Financial & Technical Data:
        - Current Price: ${stock_data.get('price', 'N/A')}
        - Previous Close: ${stock_data.get('previous_close', 'N/A')}
        - Open: ${stock_data.get('open', 'N/A')}
        - Day Range: {stock_data.get('day_range', 'N/A')}
        - 52 Week Range: {stock_data.get('fifty_two_week_range', 'N/A')}
        - Volume: {stock_data.get('volume', 'N/A')}
        - Avg Volume: {stock_data.get('avg_volume', 'N/A')}
        - Market Cap: {stock_data.get('market_cap', 'N/A')}
        - PE Ratio: {stock_data.get('pe_ratio', 'N/A')}

        {prediction_details}

        {sentiment_details}

        {news_summary}

        Based *only* on the information provided above, provide a concise stock analysis in the following JSON format:
        {{
            "overview": "A concise paragraph summarizing the stock's current situation",
            "pros": ["Pro point 1", "Pro point 2", "Pro point 3"],
            "cons": ["Con point 1", "Con point 2", "Con point 3"],
            "verdict": "buy/hold/sell"
        }}
        
        IMPORTANT: Pay special attention to JSON syntax - ensure all commas are correctly placed between array items and between fields. Make sure your response is ONLY valid parseable JSON. Include exactly 3 pros and 3 cons.
        """
        
        # Call the appropriate AI analyzer
        if analyzer == 'gemini':
            if not GEMINI_API_KEY:
                return {'error': 'Gemini API key not configured'}
            analysis = _call_gemini(prompt)
        elif analyzer == 'llama':
            # Check if model file exists
            if not os.path.exists(LLAMA_CPP_MODEL_PATH):
                return {'error': f'Llama model file not found: {LLAMA_CPP_MODEL_PATH}'}
            analysis = _call_local_llm(prompt)
        elif analyzer == 'groq':
            # Import our GroqClient with key rotation capability
            try:
                from .groq_service import groq_client
                if not groq_client.api_keys:
                    return {'error': 'Groq API keys not configured'}
                analysis = _call_groq(prompt)
            except ImportError:
                return {'error': 'Groq service module not found. Check project structure.'}
        else:
            return {'error': f"Invalid analyzer choice: {analyzer}"}
        
        # Parse and validate the JSON response
        try:
            # Extract JSON from response if needed
            analysis_dict = _extract_json(analysis)
            
            # Validate required fields
            required_fields = ['overview', 'pros', 'cons', 'verdict']
            for field in required_fields:
                if field not in analysis_dict:
                    return {'error': f"AI response missing required field: {field}"}
            
            # Validate verdict format
            verdict = analysis_dict['verdict'].lower()
            if verdict not in ['buy', 'sell', 'hold']:
                analysis_dict['verdict'] = 'hold'  # Default to hold if invalid
            
            # Add metadata
            analysis_dict['analyzer_used'] = analyzer
            analysis_dict['symbol'] = symbol
            
            # Cache the result for 2 hours (only if not using pre-collected data)
            if not collected_data:
                cache_key = f"ai_analysis_{symbol}_{analyzer}"
                cache.set(cache_key, analysis_dict, timeout=7200)
            
            return analysis_dict
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return {'error': f"Error processing AI response: {str(e)}", 'raw_response': analysis}
            
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        return {'error': f"Error performing analysis: {str(e)}"}

def _call_gemini(prompt):
    """Call Google's Gemini API with the given prompt"""
    try:
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Set up the model - use Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Return the text response
        return response.text
    except ImportError:
        return json.dumps({'error': 'Google Generative AI library not installed. Run: pip install google-generativeai'})
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return json.dumps({'error': f"Error calling Gemini API: {str(e)}"})

def _call_local_llm(prompt):
    """Call a local LLM using llama-cpp-python with the given prompt"""
    global _llm_model
    
    try:
        from llama_cpp import Llama
        
        # Check if model file exists
        if not os.path.exists(LLAMA_CPP_MODEL_PATH):
            logger.error(f"Model file not found: {LLAMA_CPP_MODEL_PATH}")
            return json.dumps({'error': f"Model file not found: {LLAMA_CPP_MODEL_PATH}"})
        
        # Preprocess the prompt to handle problematic characters
        # Replace newlines with spaces and sanitize the prompt for the tokenizer
        sanitized_prompt = prompt.replace('\n', ' ').replace('\r', ' ')
        
        try:
            # Initialize the Llama model - use cached model if available
            if _llm_model is None:
                logger.info(f"Loading LLM model from {LLAMA_CPP_MODEL_PATH}")
                _llm_model = Llama(
                    model_path=LLAMA_CPP_MODEL_PATH,
                    n_threads=LLAMA_CPP_N_THREADS,
                    n_ctx=LLAMA_CPP_N_CONTEXT,
                    n_gpu_layers=LLAMA_CPP_N_GPU_LAYERS,
                    verbose=False  # Reduce console output
                )
                logger.info("LLM model loaded successfully")
        except RuntimeError as e:
            logger.error(f"Error loading model: {e}")
            # Try loading with more conservative settings if first attempt failed
            _llm_model = None  # Reset the model variable
            logger.info("Retrying with more conservative settings...")
            _llm_model = Llama(
                model_path=LLAMA_CPP_MODEL_PATH,
                n_threads=max(1, LLAMA_CPP_N_THREADS - 2),  # Use fewer threads
                n_ctx=min(2048, LLAMA_CPP_N_CONTEXT),       # Use smaller context
                n_gpu_layers=0,                              # Disable GPU
                verbose=False
            )
            logger.info("LLM model loaded with conservative settings")
        
        # Create a system prompt
        system_prompt = "You are a helpful assistant specializing in financial analysis."
        
        # Format the prompt for chat completion or direct completion based on model capability
        try:
            # First try chat completion API
            logger.info("Attempting chat completion...")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sanitized_prompt}
            ]
            
            response = _llm_model.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                stop=["</s>", "```"]
            )
            
            # Extract the generated text
            if 'choices' in response and len(response['choices']) > 0:
                generated_text = response['choices'][0]['message']['content'].strip()
                return generated_text
        except (AttributeError, RuntimeError, Exception) as e:
            # If chat completion fails, fall back to standard completion
            logger.warning(f"Chat completion failed: {e}. Falling back to standard completion.")
            
            # Combine system prompt and user prompt for standard completion
            combined_prompt = f"{system_prompt}\n\n{sanitized_prompt}"
            
            response = _llm_model.create_completion(
                combined_prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "```", "\n\n\n"],
                echo=False
            )
            
            # Extract the generated text
            if 'choices' in response and len(response['choices']) > 0:
                generated_text = response['choices'][0]['text'].strip()
                return generated_text
        
        # If we get here, both methods failed but didn't throw exceptions
        logger.error("Unexpected response format from llama-cpp")
        return json.dumps({'error': "Invalid response format from llama-cpp"})
            
    except ImportError:
        logger.error("llama-cpp-python not installed or properly configured")
        return json.dumps({'error': 'llama-cpp-python not installed. Run: pip install llama-cpp-python'})
    except Exception as e:
        logger.error(f"Error calling llama-cpp: {e}")
        return json.dumps({'error': f"Error calling llama-cpp: {str(e)}"})

def _extract_json(text):
    """Extract and parse JSON from a text response that may contain additional content"""
    try:
        # First, try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from markdown code blocks
        import re
        
        # Look for JSON in markdown code blocks
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for JSON in text with regex (fallback)
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, text)
        if match:
            json_str = match.group(1)
            try:
                # Try to fix common JSON formatting errors
                json_str = _try_fix_json(json_str)
                return json.loads(json_str)
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Error fixing JSON: {e}")
                raise ValueError(f"Could not extract valid JSON from the AI response: {text}")
        
        raise ValueError(f"Could not extract valid JSON from the AI response: {text}")
        
def _call_groq(prompt):
    """Call Groq API with the given prompt for stock market analysis"""
    try:
        # Import our GroqClient with key rotation capability
        from .groq_service import groq_client
        
        # Call Groq with key rotation
        result = groq_client.call_groq(prompt)
        
        # Check if we got an error response (dict) or a successful text response (str)
        if isinstance(result, dict) and 'error' in result:
            logger.error(f"Error from Groq client: {result['error']}")
            return json.dumps(result)
        
        return result
            
    except ImportError:
        return json.dumps({'error': 'Groq service module not found. Check project structure.'})
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        return json.dumps({'error': f"Error calling Groq API: {str(e)}"})

def _try_fix_json(json_str):
    """Attempt to fix common JSON formatting errors"""
    import re
    
    # Fix missing commas between fields (looking for patterns like '"}' or '"' followed by another '"')
    json_str = re.sub(r'"(\s*)\n?\s*"', '",\n"', json_str)
    
    # Fix missing commas after entries in arrays
    json_str = re.sub(r'"([^"]*)"(\s*)\n?\s*]', '"\1"\n]', json_str)
    
    # Fix missing commas between array and next field
    json_str = re.sub(r'](\s*)\n?\s*"', '],\n"', json_str)
    
    # Fix case where verdict is inside cons array (a common error)
    verdict_in_cons_pattern = r'"cons":\s*\[(.*?)"verdict":\s*"(buy|sell|hold)"', re.DOTALL
    if re.search(verdict_in_cons_pattern, json_str):
        json_str = re.sub(verdict_in_cons_pattern, r'"cons": [\1],\n"verdict": "\2"', json_str)
    
    # Remove potential leading/trailing text
    json_str = re.sub(r'^[^{]*({.*})[^}]*$', r'\1', json_str)
    
    return json_str
