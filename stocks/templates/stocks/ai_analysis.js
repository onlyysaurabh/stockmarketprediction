// AI Analysis JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Element references
    const resultDiv = document.getElementById('ai-analysis-result');
    const loadingDiv = document.getElementById('ai-loading');
    const loadingText = document.getElementById('ai-loading-text');
    const waitingDiv = document.getElementById('ai-waiting');
    const waitingSeconds = document.getElementById('ai-waiting-seconds');
    const waitingProgress = document.getElementById('ai-waiting-progress');
    const contentDiv = document.getElementById('ai-content');
    const errorDiv = document.getElementById('ai-error');
    const cacheBadge = document.getElementById('ai-cache-badge');
    const cacheInfo = document.getElementById('ai-cache-info');
    const cacheAge = document.getElementById('ai-cache-age');
    const refreshBtn = document.getElementById('ai-refresh-btn');
    
    // Get button elements
    const analyzerButtons = document.querySelectorAll('[data-analyzer]');
    
    // Add click event listeners to analyzer buttons
    analyzerButtons.forEach(button => {
        button.addEventListener('click', () => {
            const analyzer = button.getAttribute('data-analyzer');
            analyzeStock(analyzer);
        });
    });
    
    // Add click event listener to refresh button
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            // Get current analyzer from source text
            const sourceText = document.getElementById('ai-source-text').textContent;
            let currentAnalyzer = 'gemini'; // Default
            
            if (sourceText.includes('Groq')) {
                currentAnalyzer = 'groq';
            } else if (sourceText.includes('Local')) {
                currentAnalyzer = 'local';
            }
            
            // Force refresh with current analyzer
            analyzeStock(currentAnalyzer, true);
        });
    }
    
    // Main function to analyze stock
    async function analyzeStock(analyzer, forceRefresh = false) {
        // Reset UI
        resultDiv.style.display = 'block';
        loadingDiv.style.display = 'block';
        waitingDiv.style.display = 'none';
        contentDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        errorDiv.textContent = '';
        cacheBadge.style.display = 'none';
        cacheInfo.style.display = 'none';
        
        try {
            // Check for cached result first, unless forcing a refresh
            if (!forceRefresh) {
                loadingText.textContent = "Checking for cached analysis...";
                
                try {
                    const cacheResponse = await fetch(`/stocks/api/cached-ai-analysis/${stockSymbol}?analyzer=${analyzer}&cached_only=true`);
                    
                    if (cacheResponse.ok) {
                        const cacheData = await cacheResponse.json();
                        
                        // We have cached data - show waiting UI with delay
                        loadingDiv.style.display = 'none';
                        waitingDiv.style.display = 'block';
                        
                        // Start countdown timer (15 seconds)
                        let secondsLeft = 15;
                        waitingSeconds.textContent = secondsLeft;
                        waitingProgress.style.width = '0%';
                        
                        const countdownInterval = setInterval(() => {
                            secondsLeft--;
                            waitingSeconds.textContent = secondsLeft;
                            waitingProgress.style.width = `${((15 - secondsLeft) / 15) * 100}%`;
                            
                            if (secondsLeft <= 0) {
                                clearInterval(countdownInterval);
                                // Display the cached data
                                displayAnalysisResult(cacheData);
                            }
                        }, 1000);
                        
                        // Early return - we're now waiting for the timer to complete
                        return;
                    }
                } catch (cacheError) {
                    console.error("Error checking cache:", cacheError);
                    // Continue with normal request if cache checking fails
                }
            }
            
            // If we get here, we're doing a fresh analysis
            loadingText.textContent = forceRefresh ? 
                "Fetching fresh analysis..." : 
                "Analyzing stock data...";
            
            // Gather all relevant data for analysis
            const collectedData = collectStockData(analyzer);
            
            // Make API request for fresh analysis
            let apiUrl = forceRefresh ? 
                `/stocks/api/cached-ai-analysis/${stockSymbol}?analyzer=${analyzer}&force_refresh=true` :
                `/stocks/ajax-ai-analysis/${stockSymbol}`;
                
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify(collectedData)
            });
            
            const data = await response.json();
            
            // Display the analysis result
            displayAnalysisResult(data);
            
        } catch (error) {
            // Handle errors
            loadingDiv.style.display = 'none';
            waitingDiv.style.display = 'none';
            errorDiv.style.display = 'block';
            errorDiv.textContent = `Error: ${error.message}`;
            console.error('AI analysis error:', error);
        }
    }
    
    // Function to collect stock data from the page
    function collectStockData(analyzer) {
        return {
            symbol: stockSymbol,
            name: stockName,
            analyzer: analyzer,
            currentPrice: stockCurrentPrice,
            keyMetrics: keyMetricsData,
            predictions: stockPredictions,
            sentiment: sentimentData,
            newsHeadlines: newsHeadlines
        };
    }
    
    // Function to display analysis result
    function displayAnalysisResult(data) {
        // Hide loading UI
        loadingDiv.style.display = 'none';
        waitingDiv.style.display = 'none';
        
        if (data.error) {
            // Show error
            errorDiv.textContent = data.error;
            errorDiv.style.display = 'block';
            contentDiv.style.display = 'none';
        } else {
            // Update UI with analysis results
            document.getElementById('ai-overview-text').textContent = data.overview;
            
            // Update pros
            const prosList = document.getElementById('ai-pros-list');
            prosList.innerHTML = '';
            data.pros.forEach(pro => {
                const li = document.createElement('li');
                li.textContent = pro;
                li.className = 'pro-item';
                prosList.appendChild(li);
            });
            
            // Update cons
            const consList = document.getElementById('ai-cons-list');
            consList.innerHTML = '';
            data.cons.forEach(con => {
                const li = document.createElement('li');
                li.textContent = con;
                li.className = 'con-item';
                consList.appendChild(li);
            });
            
            // Update verdict
            const verdictElement = document.getElementById('ai-verdict');
            verdictElement.textContent = data.verdict.toUpperCase();
            verdictElement.className = 'verdict ' + data.verdict.toLowerCase();
            
            // Update source text
            const analyzerName = getAnalyzerDisplayName(data.analyzer_used || analyzer);
            document.getElementById('ai-source-text').textContent = 'Analysis provided by ' + analyzerName;
            
            // Handle cache metadata display
            if (data.cached) {
                cacheBadge.style.display = 'inline';
                cacheInfo.style.display = 'block';
                
                // Show cache age information
                const cacheAgeMinutes = data.cache_age || 0;
                let ageText = '';
                
                if (cacheAgeMinutes < 60) {
                    ageText = `Cached ${cacheAgeMinutes} minute${cacheAgeMinutes !== 1 ? 's' : ''} ago`;
                } else {
                    const hours = Math.floor(cacheAgeMinutes / 60);
                    const mins = cacheAgeMinutes % 60;
                    ageText = `Cached ${hours} hour${hours !== 1 ? 's' : ''}${mins > 0 ? ` ${mins} minute${mins !== 1 ? 's' : ''}` : ''} ago`;
                }
                
                cacheAge.textContent = ageText;
            } else {
                cacheBadge.style.display = 'none';
                cacheInfo.style.display = 'none';
            }
            
            contentDiv.style.display = 'block';
        }
    }
    
    // Helper function to get display name for analyzer
    function getAnalyzerDisplayName(analyzer) {
        switch(analyzer) {
            case 'gemini': return 'Google Gemini';
            case 'groq': return 'Groq AI';
            case 'llama':
            case 'local': return 'Local LLM';
            default: return analyzer;
        }
    }
});
