import pandas as pd
import google.generativeai as genai
from datetime import datetime
import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error


def create_prompt_level1(df):
    """This function defines Level1 prompt with price prediction"""
    recent_data = df.tail(10).reset_index(drop=True)
    prompt = f"""
    Based on last 10 days of stock data:

    {recent_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Target']].to_string()}
    Provide two predictions:
    1. What will be tomorrows price(higher (1) or lower (0) compared to today's closing price?
    2. What will be the exact value of closing price tomorrow?
    
    Format your response exactly as follows:
    Direction: [0 or 1]
    Price: 
    Analysis: 
    """
    return prompt


def create_prompt_level2(df):
    """This function defines Level2 prompt with price prediction"""
    recent_data = df.tail(25).reset_index(drop=True)

    prompt = f"""
    You are a financial analyst. Based on the last 25 days of trading data:

    {recent_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Target']].to_string()}
    
    Please analyze:
    1. Price trend over these 25 days
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
    recent_data = df.tail(50).reset_index(drop=True)
    latest_close = recent_data["Close"].iloc[-1]
    avg_volume = recent_data["Volume"].mean()
    price_range = recent_data["High"].max() - recent_data["Low"].min()

    prompt = f"""
    You are a financial analyst. Analyze this 50-day trading data:

    Recent Trading Data:
    {recent_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Target']].to_string()}

    Key Metrics:
    - Current Close: ${latest_close:.2f}
    - 50-day Average Volume: {avg_volume:,.0f}
    - 50-day Price Range: ${price_range:.2f}

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
    recent_data = df.tail(100).reset_index(drop=True)
    latest_data = recent_data.iloc[-1]
    avg_volume_100d = recent_data["Volume"].mean()
    price_change_50d = (
        (recent_data["Close"].iloc[-1] / recent_data["Close"].iloc[-5]) - 1
    ) * 100

    prompt = f"""
    Role: You are a financial analyst specializing in price movement prediction.

    Current Market State:
    - Latest Close: ${latest_data['Close']:.2f}
    - Today's Range: ${latest_data['High']:.2f} - ${latest_data['Low']:.2f}
    - 50-day Price Change: {price_change_50d:.2f}%
    - 100-day Avg Volume: {avg_volume_100d:,.0f}

    Recent Trading History:
    {recent_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Target']].to_string()}

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
        if not response_text or "Price:" not in response_text:
            print("Invalid response format")
            return 0, 0  # Return default values

        lines = response_text.strip().split("\n")
        # Defaues
        direction = 0
        price = 0.0

        for line in lines:
            if line.startswith("Direction:") or line.startswith("**Direction:"):
                try:
                    direction = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    direction = 0
            elif line.startswith("Price:") or line.startswith("**Price:"):
                try:
                    price_str = line.split(":")[1].strip()
                    # Remove any non-numeric characters except decimal point
                    price_str = "".join(c for c in price_str if c.isdigit() or c == ".")
                    price = float(price_str)
                except (ValueError, IndexError):
                    price = 0.0
        return direction, price

    except Exception as e:
        print(f"Error parsing prediction: {str(e)}")
        return 0, 0


def calculate_metrics(actual_values, predicted_values):
    """
    This function calculates the absolute error and directional accuracy for the 15th day prediction
    """
    try:
        # Convert numpy array to scalar for calculations
        actual_value = float(actual_values[0])
        predicted_value = float(predicted_values[0])

        # Calculate absolute error(just one value differences)
        absolute_error = abs(actual_value - predicted_value)

        # Get the last closing price from the dataframe
        current_close = df.iloc[-1]["Close"]

        actual_direction = 1 if actual_value > current_close else 0
        predicted_direction = 1 if predicted_value > current_close else 0
        directional_accuracy = 1.0 if actual_direction == predicted_direction else 0.0

        return {
            "absolute_error": absolute_error,
            "directional_accuracy": directional_accuracy,
            "error": None,
        }
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {"absolute_error": 0.0, "directional_accuracy": 0.0, "error": str(e)}


def execute_prompt(prompt, model):
    """This function executes the prompt using the Gemini API"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in prompt execution: {str(e)}"


def save_results(prompts, results, metrics, output_dir="llm_results"):
    """This function saves prompts, results, and metrics"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save prompts, results, and metrics
    prompts_file = os.path.join(output_dir, f"prompts_{timestamp}.json")
    with open(prompts_file, "w") as f:
        json.dump(prompts, f, indent=2)

    markdown_file = os.path.join(output_dir, f"analysis_{timestamp}.md")
    with open(markdown_file, "w") as f:
        f.write("# Stock Prediction Analysis\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Metrics summary
        f.write("## Performance Metrics\n\n")
        for level, metric in metrics.items():
            f.write(f"### {level}\n")
            f.write(f"- Absolute Error: {metric['absolute_error']:.4f}\n")
            f.write(f"- Directional Accuracy: {metric['directional_accuracy']:.2%}\n\n")

        # Results
        for level in prompts.keys():
            f.write(f"## {level}\n\n")
            f.write("### Prompt\n```\n")
            f.write(prompts[level])
            f.write("\n```\n\n")
            f.write("### Response\n")
            f.write(results[level])
            f.write("\n\n")

    return {"prompts_file": prompts_file, "markdown_file": markdown_file}


def run_multiple_iterations(df, num_iterations=3):
    """
    This function runs multiple iterations of stock price prediction and calculates RMSE
    """
    all_predictions = {}
    all_actual_values = {}
    ACTUAL_VALUE = df["Target"].iloc[-1]

    # Set the last target value to NaN for prediction
    df.iloc[-1, df.columns.get_loc("Target")] = np.nan

    for iteration in range(1, num_iterations + 1):
        print(f"\nIteration {iteration}")

        # Prompts
        prompts = {
            "Level 1": create_prompt_level1(df),
            "Level 2": create_prompt_level2(df),
            "Level 3": create_prompt_level3(df),
            "Level 4": create_prompt_level4(df),
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

            # Calculate the metrics
            pred_values = np.array([price])
            actual_values = np.array([ACTUAL_VALUE])
            metrics[level] = calculate_metrics(actual_values, pred_values)

        # Store results for this iteration
        all_predictions[iteration] = predictions
        all_actual_values[iteration] = ACTUAL_VALUE

        # Save results for each iteration
        save_results(
            prompts, results, metrics, output_dir=f"llm_results_iteration_{iteration}"
        )

    # Calculate the RMSE across iterations
    rmse_results = calculate_rmse(all_predictions, all_actual_values)

    return rmse_results


def calculate_rmse(all_predictions, all_actual_values):
    """
    Calculate Root Mean Squared Error across iterations and prediction levels
    """

    rmse_results = {}

    for level in ["Level 1", "Level 2", "Level 3", "Level 4"]:
        level_predictions = [
            predictions.get(level, 0) for predictions in all_predictions.values()
        ]
        actual_values = [
            all_actual_values[iteration] for iteration in all_predictions.keys()
        ]

        # Remove NaN values if any
        valid_predictions = [
            p for p, a in zip(level_predictions, actual_values) if not np.isnan(a)
        ]
        valid_actual_values = [a for a in actual_values if not np.isnan(a)]

        if valid_predictions and valid_actual_values:
            rmse = np.sqrt(mean_squared_error(valid_actual_values, valid_predictions))
            rmse_results[level] = rmse
        else:
            rmse_results[level] = None

    return rmse_results


if __name__ == "__main__":
    # API Configuration
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        raise EnvironmentError(
            "API key not found. Please set GEMINI_API_KEY environment variable."
        )

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    dataset_path = "dataset/data_llm.csv"
    df = pd.read_csv(dataset_path)

    # Run multiple iterations and calculate RMSE
    rmse_results = run_multiple_iterations(df)

    # Print RMSE results
    print("\nRoot Mean Squared Error (RMSE) Results:")
    for level, rmse in rmse_results.items():
        print(f"{level}: {rmse:.4f}" if rmse is not None else f"{level}: No valid data")
