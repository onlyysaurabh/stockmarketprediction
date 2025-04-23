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

def get_ai_stock_analysis(symbol, analyzer_choice=None, collected_data=None):
    """
    Get AI-powered analysis for a stock symbol
    
    Parameters:
    - symbol: The stock symbol to analyze
    - analyzer_choice: Which AI model to use ('gemini' or 'local'), defaults to environment setting
    - collected_data: Pre-collected data from the frontend (optional)
    
    Returns:
    - Dictionary containing analysis or error message
    """
    # Use the specified analyzer or fall back to default
    analyzer = analyzer_choice or DEFAULT_AI_ANALYZER
    
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
        model = genai.GenerativeModel('gemini-2.5-flash')
        
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
