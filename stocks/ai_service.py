import json
import logging
import os
from django.conf import settings
from django.core.cache import cache
from .services import get_stock_data, get_model_predictions
from .sentiment_service import get_stock_sentiment
from .news_service import get_news_for_stock

# Set up logging
logger = logging.getLogger(__name__)

# Check for environment variables
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
LOCAL_LLM_MODEL_PATH = os.environ.get('LOCAL_LLM_MODEL_PATH')
DEFAULT_AI_ANALYZER = os.environ.get('DEFAULT_AI_ANALYZER', 'gemini')
LOCAL_LLM_N_GPU_LAYERS = int(os.environ.get('LOCAL_LLM_N_GPU_LAYERS', '0'))
LOCAL_LLM_N_CTX = int(os.environ.get('LOCAL_LLM_N_CTX', '2048'))

def get_ai_stock_analysis(symbol, analyzer_choice=None):
    """
    Get AI-powered analysis for a stock symbol
    
    Parameters:
    - symbol: The stock symbol to analyze
    - analyzer_choice: Which AI model to use ('gemini' or 'local'), defaults to environment setting
    
    Returns:
    - Dictionary containing analysis or error message
    """
    # Use the specified analyzer or fall back to default
    analyzer = analyzer_choice or DEFAULT_AI_ANALYZER
    
    # Check cache first
    cache_key = f"ai_analysis_{symbol}_{analyzer}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result
    
    # Collect all relevant stock data for analysis
    try:
        # Get technical stock data
        stock_data = get_stock_data(symbol)
        if not stock_data or 'error' in stock_data:
            return {'error': f"Could not fetch stock data for {symbol}"}
        
        # Get prediction data
        prediction_data = get_model_predictions(symbol)
        if not prediction_data or 'error' in prediction_data:
            prediction_summary = "No prediction data available"
        else:
            # Format predictions in a readable way
            models = prediction_data.get('model_types', [])
            if models:
                # Use the first model's prediction for next day as direction indicator
                first_model = models[0]
                next_day_pred = prediction_data.get('predictions', {}).get('next_day', {}).get(first_model)
                current_price = stock_data.get('price', 0)
                
                if next_day_pred and current_price:
                    direction = "up" if float(next_day_pred) > float(current_price) else "down"
                    prediction_summary = f"Predicted price movement: {direction} (based on {first_model.upper()} model)"
                else:
                    prediction_summary = "Predictions available but current comparison not possible"
            else:
                prediction_summary = "Models available but no specific prediction data"
        
        # Get sentiment data
        sentiment_data = get_stock_sentiment(symbol)
        if not sentiment_data or 'error' in sentiment_data:
            sentiment_summary = "No sentiment data available"
        else:
            sentiment_summary = f"Overall sentiment: {sentiment_data.get('overall_sentiment', 'Unknown')}"
        
        # Get recent news
        news_data = get_news_for_stock(symbol, limit=5)
        if not news_data or 'error' in news_data:
            news_summary = "No recent news available"
        else:
            news_headlines = [news.get('title', 'Unknown headline') for news in news_data[:5]]
            news_summary = "Recent headlines: " + "; ".join(news_headlines)
        
        # Construct prompt for AI
        prompt = f"""
        Analyze the following stock information for {symbol} and provide investment insights.
        
        Technical Data:
        - Current Price: ${stock_data.get('price')}
        - Previous Close: ${stock_data.get('previous_close')}
        - Open: ${stock_data.get('open')}
        - Day Range: {stock_data.get('day_range')}
        - 52 Week Range: {stock_data.get('fifty_two_week_range')}
        - Volume: {stock_data.get('volume')}
        - Avg Volume: {stock_data.get('avg_volume')}
        - Market Cap: {stock_data.get('market_cap')}
        - PE Ratio: {stock_data.get('pe_ratio')}
        
        {prediction_summary}
        {sentiment_summary}
        {news_summary}
        
        Based on this information, provide a comprehensive stock analysis in the following JSON format:
        {{
            "overview": "A concise paragraph summarizing the stock's current situation",
            "pros": ["Pro point 1", "Pro point 2", ...],
            "cons": ["Con point 1", "Con point 2", ...],
            "verdict": "buy/hold/sell"
        }}
        
        Ensure your response is ONLY the JSON format specified above. Include at least 3 pros and 3 cons.
        """
        
        # Call the appropriate AI analyzer
        if analyzer == 'gemini':
            if not GEMINI_API_KEY:
                return {'error': 'Gemini API key not configured'}
            analysis = _call_gemini(prompt)
        elif analyzer == 'local':
            if not LOCAL_LLM_MODEL_PATH:
                return {'error': 'Local LLM model path not configured'}
            analysis = _call_local_llm(prompt)
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
            
            # Cache the result for 2 hours
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
        
        # Set up the model
        model = genai.GenerativeModel('')
        
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
    try:
        from llama_cpp import Llama
        
        # Load the model
        model = Llama(
            model_path=LOCAL_LLM_MODEL_PATH,
            n_ctx=LOCAL_LLM_N_CTX,
            n_gpu_layers=LOCAL_LLM_N_GPU_LAYERS
        )
        
        # Generate response
        response = model.create_completion(
            prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "```"],
            echo=False
        )
        
        # Extract the generated text from the response
        generated_text = response['choices'][0]['text'].strip()
        return generated_text
    except ImportError:
        return json.dumps({'error': 'Llama-cpp-python library not installed. Run: pip install llama-cpp-python'})
    except Exception as e:
        logger.error(f"Error calling local LLM: {e}")
        return json.dumps({'error': f"Error calling local LLM: {str(e)}"})

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
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                raise ValueError(f"Could not extract valid JSON from the AI response: {text}")
        
        raise ValueError(f"Could not extract valid JSON from the AI response: {text}")
