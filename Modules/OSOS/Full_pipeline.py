## Import libraries

import subprocess

import os
import glob

# Ensure Hugging Face uses a project-local cache directory (works with older transformers)
os.environ.setdefault("HF_HOME", os.path.join("Modules", "OSOS", "model_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join("Modules", "OSOS", "model_cache"))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import snscrape  # unused when scraping is disabled
# import openpyxl  # unused in this pipeline
from transformers import pipeline

#%%
# Set paths (use downloaded data + local model cache)
# If you used `python Modules/OSOS/task1_download.py --base-path data ...`,
# then base_path should be "data".
base_path = input("Enter the path where data is stored [data]: ").strip() or "data"
cache_dir = os.path.join("Modules", "OSOS", "model_cache")
os.makedirs(cache_dir, exist_ok=True)


#%%

# Part 1 (scraping) is disabled because data was downloaded offline
# using Modules/OSOS/task1_download.py.
# If you want to re-enable scraping, uncomment this block.
#
# print("Part 1:Get Twitter data")
# start = input("Enter the start date (e.g., 2020-01-01): ")
# keywords = input("Enter keywords (comma-separated): ")
# location = input("Enter the location (comma-separated) (e.g., Germany): ")
# end = input("Enter the end date (e.g., 2022-12-31): ")
#
# # Format keywords with OR and appropriate brackets
# formatted_keywords = " OR ".join([f"({kw})" if " " in kw else kw for kw in map(str.strip, keywords.split(","))])
#
# # Format location with single quotes
# formatted_location = " OR ".join([f"({kw})" if " " in kw else kw for kw in map(str.strip, location.split(","))])
#
# # Construct the command string
# snscrape_command = (
#     f"snscrape --jsonl --since {start} twitter-search "
#     f"\"{formatted_keywords} near:'{formatted_location}' until:{end}\" > text-query-tweets.json"
# )
# # Print the generated command for sanity
# print("Generated Command: ", snscrape_command)


#%%

intermediate_file = "Scrap_Results_bursty.csv"

# Scrape + conversion disabled (offline data already exists).
# If you want to enable scraping, uncomment this block.
#
# ## Run the command as a subprocess
# try:
#     completed_process = subprocess.run(
#         snscrape_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#     )
#     
#     print("Getting Twitter data based on the supplied keywords")
#     return_code = completed_process.returncode
#
#     if return_code == 0:
#         print("Command executed successfully.")
#         print("Converting file to csv")
#         tweets_df = pd.read_json('text-query-tweets.json', lines=True)
#         filtered_df = tweets_df[["rawContent", "date"]]
#         filtered_df = filtered_df.rename(columns={"rawContent": "tweet"})
#         filtered_df['completion'] = 'neutral'
#         data_path = os.path.join(base_path, intermediate_file)
#         filtered_df.to_csv(data_path)
#
#     else:
#         print("Command execution failed.")
#         print("Standard Output:\n%s", completed_process.stdout)
#         print("Standard Error:\n%s", completed_process.stderr)
# except subprocess.CalledProcessError as e:
#     print("Command execution failed: %s", e)
  


#%%

## Perform binary classification on the scrapped and pre-processed data

print("Part 2: Performing binary classification on the scrapped and pre-processed data")

## Peform finetuning on the data
print("Part 2a: Preparing data for classification")
filepath = os.path.join(base_path, intermediate_file)
use_prepare_data = False  # Set True only if your CSV already has prompt/completion columns.
openai_command = f"openai tools fine_tunes.prepare_data -f {filepath}"


## Run the command to load the fine-tuned data into a jsonl file
base_filename = intermediate_file.replace(".csv", "")

if use_prepare_data:
    # Overwrite existing prepared outputs to avoid "(1)" suffix files.
    prepared_patterns = [
        os.path.join(base_path, f"{base_filename}_prepared*.jsonl"),
    ]
    for pattern in prepared_patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                # Best-effort cleanup; if removal fails, prepare_data may create a suffixed file.
                pass

    try:
        subprocess.check_call(openai_command, shell=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Command execution failed: %s", e)
        raise

    # openai prepare_data may split into train/valid when you choose Y.
    # Use the train file by default if present.
    jsonl_filename = base_filename + "_prepared_train.jsonl"
    json_filepath = os.path.join(base_path, jsonl_filename)
    if not os.path.exists(json_filepath):
        jsonl_filename = base_filename + "_prepared.jsonl"
        json_filepath = os.path.join(base_path, jsonl_filename)

    # include the date information from the original csv file
    # Original file is CSV, prepared file is JSONL
    data1 = pd.read_csv(filepath)
    data2 = pd.read_json(json_filepath, lines=True)

    # Merge dataframes based on the key_column with a left outer join
    combined_df = data2[['prompt', 'completion']].join(data1['date'])
else:
    # Use the raw CSV (tweet/date/completion) for zero-shot classification.
    data1 = pd.read_csv(filepath)
    if 'completion' not in data1.columns:
        data1['completion'] = 'neutral'
    # Build a prompt column for compatibility with later steps.
    data1['prompt'] = data1['tweet'].astype(str)
    combined_df = data1[['prompt', 'completion', 'date']]

print(combined_df)

## Splitting the data into 4 chunks for faster processing
print("Part 2b: Splitting the data into 4 chunks for faster processing")

# Calculate the number of rows in the DataFrame
num_rows = len(combined_df)
part_size = num_rows // 4

parts = [combined_df[i * part_size:(i + 1) * part_size] for i in range(4)]
part_files = []
# Save each part as a separate JSONL file
for i, part in enumerate(parts):
    jsonl_partfilename = f'{base_filename}_part_{i+1}.jsonl'
    part_filepath = os.path.join(base_path, jsonl_partfilename)
    part_files.append(part_filepath)  # Append the filename to the list
    part.to_json(part_filepath, orient='records', lines=True)
    
print("The 4 chunks are: ")
print(part_files)

## Performing binary classfication on the 4 chunks using a local model
print("Part 2c: Performing binary classfication on the 4 chunks using a fine-tuned model")
# Local zero-shot classifier (no API calls, runs on CPU/GPU).
# Smaller model for faster CPU inference.
zero_shot = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1",
)
candidate_labels = ["informative", "not_informative"]

## Running a loop for binary classfication on all chunks
total_rows = sum(len(pd.read_json(f, lines=True)) for f in part_files)
processed_rows = 0
print(f"Total tweets to label: {total_rows}")

for file in part_files:
  test = pd.read_json(file, lines=True)

  # Create a list to store the responses
  responses = []

  # Iterate over each prompt in the DataFrame
  for i, prompt in enumerate(test['prompt'], start=1):
    # Prepare a clean tweet text (strip the " ->" suffix added by prepare_data).
    tweet_text = prompt.replace(" ->", "")
    # Local zero-shot classification
    result = zero_shot(tweet_text, candidate_labels)
    response = result["labels"][0]

    # Append the response to the list
    responses.append(response)
    processed_rows += 1
    if processed_rows % 10 == 0 or processed_rows == total_rows:
        print(f"Labeled {processed_rows}/{total_rows}")

  # Add the responses as a new column in the DataFrame
  test['response'] = responses

  # Save the DataFrame to a new JSON file
  test.to_json(file, orient='records', lines=True)
  

# Create an empty DataFrame to store the merged data
merged_data = pd.DataFrame()


## Merging the classified dataframes into one
print("Part 2d:Merging the classified dataframes into one")
# Iterate over each file
for file_name in part_files:
    # Read the JSONL file into a DataFrame
    data = pd.read_json(file_name, lines=True)

    # Append the data to the merged DataFrame
    merged_data = merged_data.append(data, ignore_index=True)

# Save the merged DataFrame to a new JSONL file
new_file = base_filename + "_final.jsonl"
new_file_csv= base_filename + "_final.csv"
merged_data.to_json(new_file, orient='records', lines=True)  


# Read JSONL file into a DataFrame
data = pd.read_json(new_file, lines=True)
data = data.drop('completion', axis=1)

print("Binary Classfication completed successfully")


#%% Sentiment Analysis

## Sentiment Analysis
print("Part 3: Perfomring sentiment analysis on the tweets")

sentiment = pipeline(
    "sentiment-analysis",
    model="typeform/distilbert-base-uncased-mnli",
)

def sentiment_analysis(classifier, tweets, text_col="prompt"):
    '''
    classfier : the classifier object
    tweets    : the tweets dataset
    '''
    
    # Perform sentiment analysis on each row of the DataFrame
    tweets['sentiment'] = tweets[text_col].apply(lambda x: classifier(str(x))[0]['label'])
    
    return tweets


# data = sentiment_analysis(sentiment, data)
# data_filtered = data[data['sentiment'].isin(['CONTRADICTION', 'NEUTRAL'])]
data_filtered = data

  
print("Senitment Analysis completed successfully")



#%%
## Checking for bursts and analysing the data counts
print("Part 4:Checking for bursts and analysing the data trend")


## Use all sentiment-filtered data for bursts (more rows)
inf_data = data_filtered

## Performing burst detection on the data after grouping it into months
print("Part 4a:Performing burst detection on the data")

def burst_detection_daily(data, threshold, min_length):
    '''
    data      : the tweets data
    threshold : the threshold above which a burst is detected
    min_length: the min time period for which the burst should be active
    '''
    data['date'] = pd.to_datetime(data['date'])
    tweet_count_by_day = data.groupby(data['date'].dt.to_period('D'))['prompt'].count()
    tweet_count_by_day.index = tweet_count_by_day.index.to_timestamp()

    trend_change = tweet_count_by_day.diff().fillna(0)

    burst_indices = []
    burst_start = None
    burst_diff = []

    for index, diff_value in enumerate(trend_change):
        if diff_value > threshold:
            if burst_start is None:
                burst_start = index - 1
        else:
            if burst_start is not None:
                if index - burst_start >= min_length:
                    burst_indices.append((tweet_count_by_day.index[burst_start], 
                                          tweet_count_by_day.index[index - 1]))
                    start_value = tweet_count_by_day.iloc[burst_start]
                    end_value = tweet_count_by_day.iloc[index - 1]
                    diff_value = end_value - start_value
                    burst_diff.append(diff_value)

                elif index - burst_start == 1:
                    burst_indices.append((tweet_count_by_day.index[burst_start], 
                                          tweet_count_by_day.index[burst_start]))
                    start_value = tweet_count_by_day.iloc[burst_start]
                    end_value = tweet_count_by_day.iloc[index - 1]
                    diff_value = end_value - start_value
                    burst_diff.append(diff_value)
                burst_start = None

    # Append the last series if it ended with a value greater than threshold
    if burst_start is not None and len(trend_change) - burst_start >= min_length:
        burst_indices.append((tweet_count_by_day.index[burst_start], 
                              tweet_count_by_day.index[len(trend_change) - 1]))
        start_value = tweet_count_by_day.iloc[burst_start]
        end_value = tweet_count_by_day.iloc[len(trend_change) - 1]
        diff_value = end_value - start_value
        burst_diff.append(diff_value)

    elif burst_start is not None and len(trend_change) - burst_start == 1:
        burst_indices.append((tweet_count_by_day.index[burst_start], tweet_count_by_day.index[burst_start]))
        start_value = tweet_count_by_day.iloc[burst_start]
        end_value = tweet_count_by_day.iloc[len(trend_change) - 1]
        diff_value = end_value - start_value
        burst_diff.append(diff_value)

    result = [list(x) for x in zip(burst_indices, burst_diff)]
    return result



## Lower thresholds to detect bursts in small datasets
bursts_data = burst_detection_daily(inf_data, 4, 2)
print(f"Bursts detected: {len(bursts_data)}")

## Plotting the signals, prices and burst periods together
print("Plotting the signals, prices and burst periods together")
def plot_with_bursts(data, bursts, freq="D", out_path=None):
    '''
    data      : the tweets data
    bursts : the detected burst periods
    freq : 'D' for daily, 'M' for monthly
    out_path : optional path to save PNG
    '''
    data['date'] = pd.to_datetime(data['date'])
    tweet_count = data.groupby(data['date'].dt.to_period(freq))['prompt'].count()
    tweet_count.index = tweet_count.index.to_timestamp()

    # Plot trends in tweets
    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.plot(tweet_count.index, tweet_count.values, label='Trends in tweets', color='tab:blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('No. of tweets', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Show only every 3rd label along the x-axis
    xticks = tweet_count.index[::3]
    xticklabels = [d.strftime('%B %Y') for d in xticks]
    
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, rotation=45)

    # Plot burst periods as shaded regions
    burst_periods = [x[0] for x in bursts]
    for burst_start, burst_end in burst_periods:
        ax1.axvspan(burst_start, burst_end, color='orange', alpha=0.3)

    # Add dummy element to display burst periods label in a separate legend
    ax1.fill_between([], [], color='orange', alpha=0.3, label='Burst Periods')
    ax1.legend(loc='upper left')

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.show()

ylabel = input("Enter the label for price axis: ")
llabel = input("Enter the name of the quantity you're analysing (eg, gas, electricity, coal): ")

daily_png = os.path.join(base_path, "burst_plot_daily.png")
monthly_png = os.path.join(base_path, "burst_plot_monthly.png")
plot_with_bursts(inf_data, bursts_data, freq="D", out_path=daily_png)
plot_with_bursts(inf_data, bursts_data, freq="M", out_path=monthly_png)


## Performing burst detection on the data after grouping it into months
print("Part 3b:Saving the burst periods to a file")
def save_to_file(series, filename):
    '''
    series      : the detected burst periods
    filename      : the path and filename
    '''
    try:
        # Convert the list of tuples to a DataFrame

        final_list = [[x[0][0].strftime('%Y-%m-%d'), x[0][1].strftime('%Y-%m-%d'), x[1]] for x in series]
        burst_df = pd.DataFrame(final_list, columns=['Start', 'End', 'Burst Diff'])
        
        # Split the 'Start' and 'End' columns into separate columns
        #burst_df[['Start', 'End']] = pd.DataFrame(burst_df['Start'].tolist(), index=burst_df.index)
        
        # Save the DataFrame to a CSV file
        burst_df.to_csv(filename, index=False)
        
        return 0  # Successful
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1  # Error

bursts_file = input("Enter the filename to which burst data should be saved to (.csv): ")
bursts_path = os.path.join(base_path, bursts_file)
save_to_file(bursts_data, bursts_path)
