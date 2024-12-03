import pandas as pd
import google.generativeai as genai
from datetime import datetime
import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error

def load_and_prepare_data(file_path):
    """This function loads and prepares the stock data with target values"""
    df = pd.read_csv(file_path)

    # Extract first 15 rows, The last row has "N/A" value in the target column so isnt required
    sample_data = df.head(15)  
    return sample_data

def create_prompt_level1(df):
    """This function defines Level1 prompt with price prediction"""
    latest_data = df.iloc[-1]
    
    prompt = f"""
    Based on today's stock data:
    Open: ${latest_data['Open']:.2f}
    High: ${latest_data['High']:.2f}
    Low: ${latest_data['Low']:.2f}
    Close: ${latest_data['Close']:.2f}
    Volume: {latest_data['Volume']}
    
    Provide two predictions:
    1. What will be tomorrows price(higher (1) or lower (0) compared to today's closing price?
    2. What will be the exact value of closing price tomorrow?
    
    Format your response exactly as follows:
    Direction: [0 or 1]
    Price: [predicted price]
    Reasoning: [your explanation]
    """
    return prompt

def create_prompt_level2(df):
    """This function defines Level2 prompt with price prediction"""
    recent_data = df.tail(3)
    
    prompt = f"""
    You are a financial analyst. Based on the last 3 days of trading data:

    {recent_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_string()}
    
    Please analyze:
    1. Price trend over these 3 days
    2. Daily trading ranges (High - Low)
    3. Volume patterns
    
    Provide two predictions:
    1. What will be tomorrows price(higher (1) or lower (0) compared to today's closing price?
    2. What will be the exact value of closing price tomorrow?

    Please use the following format for your response:
    Direction: [0 or 1]
    Price: [predicted price]
    Confidence: [confidence level]
    Analysis: [your detailed analysis]
    """
    return prompt

def create_prompt_level3(df):
    """This function defines Level3 prompt with price prediction"""
    recent_data = df.tail(5)
    latest_close = recent_data['Close'].iloc[-1]
    avg_volume = recent_data['Volume'].mean()
    price_range = recent_data['High'].max() - recent_data['Low'].min()
    
    prompt = f"""
    You are a financial analyst. Analyze this 5-day trading data:

    Recent Trading Data:
    {recent_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_string()}

    Key Metrics:
    - Current Close: ${latest_close:.2f}
    - 5-day Average Volume: {avg_volume:,.0f}
    - 5-day Price Range: ${price_range:.2f}

    Provide a comprehensive analysis and two specific predictions:
    1. What will be tomorrows price(higher (1) or lower (0) compared to today's closing price?
    2. What will be the exact value of closing price tomorrow?

    Please use the following format for your response:
    Direction: [0 or 1]
    Price: [predicted price]
    Confidence: [percentage]
    Analysis: [your detailed analysis including pattern analysis, risk assessment, and key factors]
    """
    return prompt

def create_prompt_level4(df):
    """This function defines Level4 prompt with price prediction"""
    recent_data = df.tail(10)
    latest_data = recent_data.iloc[-1]
    avg_volume_10d = recent_data['Volume'].mean()
    price_change_5d = ((recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[-5]) - 1) * 100
    
    prompt = f"""
    Role: You are a financial analyst specializing in short-term price movement prediction.

    Current Market State:
    - Latest Close: ${latest_data['Close']:.2f}
    - Today's Range: ${latest_data['High']:.2f} - ${latest_data['Low']:.2f}
    - 5-day Price Change: {price_change_5d:.2f}%
    - 10-day Avg Volume: {avg_volume_10d:,.0f}

    Recent Trading History:
    {recent_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_string()}

    Provide a detailed analysis and two specific predictions:
    1. What will be tomorrows price(higher (1) or lower (0) compared to today's closing price?
    2. What will be the exact value of closing price tomorrow?

    Please use the following format for your response:
    Direction: [0 or 1]
    Price: [predicted price]
    Confidence: [percentage]
    Analysis: [your comprehensive analysis including all required sections]
    Price Targets:
    - Support: [price level]
    - Resistance: [price level]
    Risk Factors: [list key risks]
    """
    return prompt

def parse_prediction(response_text):
    """This function parses the model's response to extract predictions"""
    try:
        if not response_text or 'Price:' not in response_text:
            print("Invalid response format")
            return 0, 0  # Return default values
            
        lines = response_text.strip().split('\n')
        # Defaues
        direction = 0  
        price = 0.0    
        
        for line in lines:
            if line.startswith('Direction:'):
                try:
                    direction = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    direction = 0
            elif line.startswith('Price:'):
                try:
                    price_str = line.split(':')[1].strip()
                    # Remove any non-numeric characters except decimal point
                    price_str = ''.join(c for c in price_str if c.isdigit() or c == '.')
                    price = float(price_str)
                except (ValueError, IndexError):
                    price = 0.0
                    
        return direction, price
    except Exception as e:
        print(f"Error parsing prediction: {str(e)}")
        return 0, 0  


def calculate_metrics(actual_values, predicted_values):
    """This function calculates the RMSE and other metrics with validation"""
    try:
        # Input conversion to numpy arrays and ensure they're float
        actual_values = np.array(actual_values, dtype=float)
        predicted_values = np.array(predicted_values, dtype=float)
        
        # Validate inputs
        if len(actual_values) == 0 or len(predicted_values) == 0:
            return {
                'rmse': 0.0,
                'directional_accuracy': 0.0,
                'error': 'Empty input arrays'
            }
        
        if len(actual_values) != len(predicted_values):
            return {
                'rmse': 0.0,
                'directional_accuracy': 0.0,
                'error': 'Length mismatch between actual and predicted values'
            }
            
        # Clean any NaN values
        mask = ~np.isnan(actual_values) & ~np.isnan(predicted_values)
        actual_values = actual_values[mask]
        predicted_values = predicted_values[mask]
        
        if len(actual_values) == 0:
            return {
                'rmse': 0.0,
                'directional_accuracy': 0.0,
                'error': 'No valid data points after removing NaN values'
            }
        
        # Calculate the RMSE
        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        
        # Calculate directional accuracy
        actual_direction = [1 if actual_values[i] > actual_values[i-1] else 0 
                          for i in range(1, len(actual_values))]
        pred_direction = [1 if predicted_values[i] > actual_values[i-1] else 0 
                         for i in range(1, len(predicted_values))]
        
        directional_accuracy = sum(1 for a, p in zip(actual_direction, pred_direction) 
                                 if a == p) / len(actual_direction) if actual_direction else 0.0
        
        return {
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'error': None
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'rmse': 0.0,
            'directional_accuracy': 0.0,
            'error': str(e)
        }

def execute_prompt(prompt, model):
    """This function executes the prompt using the Gemini API"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in prompt execution: {str(e)}"

def save_results(prompts, results, metrics, output_dir="llm_results"):
    """This function saves prompts, results, and metrics to files"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save prompts, results, and metrics
    prompts_file = os.path.join(output_dir, f"prompts_{timestamp}.json")
    with open(prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    markdown_file = os.path.join(output_dir, f"analysis_{timestamp}.md")
    with open(markdown_file, 'w') as f:
        f.write("# Stock Prediction Analysis\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Metrics summary
        f.write("## Performance Metrics\n\n")
        for level, metric in metrics.items():
            f.write(f"### {level}\n")
            f.write(f"- RMSE: {metric['rmse']:.4f}\n")
            f.write(f"- Directional Accuracy: {metric['directional_accuracy']:.2%}\n\n")
        
        # Detailed results
        for level in prompts.keys():
            f.write(f"## {level}\n\n")
            f.write("### Prompt\n```\n")
            f.write(prompts[level])
            f.write("\n```\n\n")
            f.write("### Response\n")
            f.write(results[level])
            f.write("\n\n")
    
    return {
        'prompts_file': prompts_file,
        'markdown_file': markdown_file
    }

if __name__ == "__main__":
    # API Configuration
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise EnvironmentError("API key not found.Please set GEMINI_API_KEY environment variable.")
    
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
    dataset = "dataset/data_llm.csv"
    df = load_and_prepare_data(dataset)
    
    # Prompts
    prompts = {
        "Level 1": create_prompt_level1(df),
        "Level 2": create_prompt_level2(df),
        "Level 3": create_prompt_level3(df),
        "Level 4": create_prompt_level4(df)
    }
    
    results = {}
    predictions = {}
    metrics = {}
    
    for level, prompt in prompts.items():
        print(f"{level}...")
        response = execute_prompt(prompt, model)
        results[level] = response
        
        # Parse predictions
        direction, price = parse_prediction(response)
        if price is not None:
            predictions[level] = price
        
        # Calculate metrics if actual values are available
        if 'Target' in df.columns:
            actual_values = df['Target'].dropna().values
            if len(actual_values) > 0:
                # We must repeat the prediction for comparison
                pred_values = [price] * len(actual_values)  
                metrics[level] = calculate_metrics(actual_values, pred_values)
    
    # Save results
    output_files = save_results(prompts, results, metrics)
    
    print("\nFiles generated:")
    for file_type, file_path in output_files.items():
        print(f"{file_type}: {file_path}")
    