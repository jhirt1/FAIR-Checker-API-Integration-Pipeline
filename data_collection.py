###################################################
# Data Collection Script for Fairness Analysis  #####
# Author: Juli Hirt                             #####
# Date: 2025-10-15                              #####
###################################################


#########
# Imports
#########
import pandas as pd
import requests
import json
import numpy as np
import logging 
import datetime as dt
import os
import shutil
import unicodedata
import re
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time
import glob
from kneed import KneeLocator

#########
# Parameters
#########
API_URL = "https://fair-checker.france-bioinformatique.fr/api/check/legacy/metrics_all?url="
MODE = 'test' 
# MODE = 'full'
# MODE = 'rerun' 
logger = logging.getLogger(__name__)
ROOT = '.'
COL = 'Total Citations'
ZERO_WIDTH = "\u200B\u200C\u200D\uFEFF"  # Zero-width characters and BOM
THIN_NBSP = "\u202F\xa0"                 # Thin non-breaking spaces
POW_NORM = 0.25                           # Power for normalization
JITTER_MAX = 15                           # Max jitter for API calls in seconds
TARGET = 'Sample.xlsx'
SHEET_NAME = 'Chemistry'  


# Production safety (minimal)
REQUEST_TIMEOUT = 60  # seconds


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def pjoin(*parts: str) -> str:
    return os.path.join(*parts)

#########
# Functions
#########
def reset_environment(folder):
    """
    Clean up the specified folder by deleting all files and subdirectories.
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return

def mk_dirs(folder):
    """
    Create directory if it does not exist.
    """
    ## error reports
    error_sub = os.path.join(ROOT, 'Outbound/Error Record Reports', folder)
    if not os.path.exists(error_sub):
        os.makedirs(error_sub)
    
    ## logging
    log_sub = os.path.join(ROOT, 'Outbound/Logging', folder)
    if not os.path.exists(log_sub):
        os.makedirs(log_sub)

    ## raw data
    raw_sub = os.path.join(ROOT, 'Outbound/Raw Data', folder)
    if not os.path.exists(raw_sub):
        os.makedirs(raw_sub)
    
    ## results
    results_sub = os.path.join(ROOT, 'Outbound/Results', folder)
    if not os.path.exists(results_sub):
        os.makedirs(results_sub)

    ## sampling (new) - all sampling-related output files go here
    sampling_sub = os.path.join(ROOT, 'Outbound/Sampling', folder)
    if not os.path.exists(sampling_sub):
        os.makedirs(sampling_sub)

    return

def clean_numstr(s: str) -> str:
    """
    Normalize, strip, and remove hidden characters or formatting
    that interfere with numeric conversion.
    """
    if s is None:
        return s
    s = unicodedata.normalize("NFKC", str(s))     ## normalize Unicode
    s = s.strip()                                  ## strip whitespace
    s = re.sub(f"[{ZERO_WIDTH}]", "", s)           ## remove zero-width chars/BOM
    s = re.sub(rf"[ \t\r\n{THIN_NBSP}]", "", s)    ## remove spaces (incl thin nbsp)
    s = s.replace(",", "")                         ## remove thousands-separators
    return s

def sample_size(row):
    """
    Calculate the suggested sample size based on the provided formula.
    Ensures the sample size does not exceed the available count.
    """
    min_sample = row['Elbow_Value']
    sugg_samp = np.ceil(row['Count']*row['Perc_to_Select'])

    if sugg_samp < min_sample:
        samp = min_sample
    else:
        samp = sugg_samp

    if samp > row['Count']:
        return row['Count']
    else:
        return np.ceil(samp)
    
def findable_sum_calc(row):
    """
    Calculate the sum of findable metrics.
    """
    return int(row['F1A']) + int(row['F1B']) + int(row['F2A']) + int(row['F2B'])

def findable_perc_calc(row):
    """
    Calculate the percentage of findable metrics.
    """
    return round((row['FSum']/8)*100,3)

def accessible_sum_calc(row):
    """
    Calculate the sum of accessible metrics.
    """
    return int(row['A1.1']) + int(row['A1.2'])

def accessible_perc_calc(row):
    """
    Calculate the percentage of accessible metrics.
    """
    return round((row['ASum']/4)*100, 3)

def interroperable_sum_calc(row):
    """
    Calculate the sum of interoperable metrics.
    """
    return int(row['I1']) + int(row['I2']) + int(row['I3'])

def interroperable_perc_calc(row):
    """
    Calculate the percentage of interoperable metrics.
    """
    return round((row['ISum']/6)*100,3)

def reusable_sum_calc(row):
    """
    Calculate the sum of reusable metrics.
    """
    return int(row['R1.1']) + int(row['R1.2']) + int(row['R1.3'])

def reusable_perc_calc(row):
    """
    Calculate the percentage of reusable metrics.
    """
    return round((row['RSum']/6)*100,3)

def weighted_fair_score(row):
    """
    Calculate the weighted FAIR score.
    """
    sum_metrics = (row['FSum']) + (row['ASum']) + (row['ISum']) + (row['RSum'])
    return round(sum_metrics/24 *100,3)

def all_scores_zero(api_items):
    """
    Return True if every 'score' in the API payload is numeric and equals 0.
    Returns False if any score is non-zero or non-numeric; True only if
    at least one score is found and all are zero.
    """
    found_any = False
    for it in api_items:
        if 'score' in it:
            try:
                if int(str(it['score']).strip()) != 0:
                    return False
                found_any = True
            except Exception:
                return False
    return found_any

def pick_replacement(df_universe, used_dois, used_urs, citation_value, current_row_key):
    """
    From the full universe (df2), pick a row with the same citation bucket (Total Citations)
    whose DOI/UR hasn't been used yet and isn't the current row. Returns a Series or None.
    """
    candidates = df_universe[
        (df_universe[COL] == citation_value) &
        (~df_universe['DOI'].astype(str).str.strip().isin(used_dois)) &
        (~df_universe['UR'].astype(str).str.strip().isin(used_urs))
    ]
    if candidates.empty:
        return None

    cand = candidates.iloc[0]
    cand_key = (str(cand.get('DOI', '')).strip(), str(cand.get('UR', '')).strip())
    if cand_key == current_row_key:
        if len(candidates) > 1:
            cand = candidates.iloc[1]
        else:
            return None
    return cand

def normalize_identifier(x):
    """
    Normalize DOI/URL to improve join stability:
    - Lower-case and strip.
    - If starts with '10.' => convert to 'https://doi.org/<suffix>'.
    - Force https://doi.org/ for any DOI-like URL.
    """
    if x is None:
        return ''
    s = str(x).strip().lower()
    if s == '':
        return ''
    if s.startswith('10.'):
        return f"https://doi.org/{s}"
    if 'doi.org/' in s:
        suf = s.split('doi.org/', 1)[1]
        return f"https://doi.org/{suf}"
    return s

#########
# Main Script
#########
def main():
    try:
        #reset environment
        if MODE == 'test' or MODE == 'full':
            reset_environment(f'{ROOT}/Outbound/Logging')
            reset_environment(f'{ROOT}/Outbound/Results')
            reset_environment(f'{ROOT}/Outbound/Error Record Reports')
            reset_environment(f'{ROOT}/Outbound/Raw Data')
            reset_environment(f'{ROOT}/Outbound/Sampling')

        mk_dirs(SHEET_NAME)

        #initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            filename=f'{ROOT}/Outbound/Logging/{SHEET_NAME}/{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}_data_collection.log'
        )
        logger.info("Script started")
        logger.info(f"Running in {MODE} mode")

        #####################################
        #----------DATA PREPARATION----------

        #load data set
        logger.info("Loading data")
        input_path = os.path.join(ROOT, "Inbound", TARGET)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input Excel file not found: {input_path}")
        df = pd.read_excel(
            input_path,
            sheet_name=SHEET_NAME,
            engine="openpyxl",          
            dtype=str,                  
            header=0,   
            na_filter=False             
        )
        df_shape = df.shape
        logger.info(f"Data loaded with shape: {df_shape}")

        #rename columns
        logger.info("Renaming columns")
        df.rename(columns={'Total Times Cited Count (Web of Science Core Collection, Arabic Citation Index, BIOSIS Citation Index, Chinese Science Citation Database, Data Citation Index, Russian Science Citation Index, SciELO Citation Index)':'Total Citations'}, inplace=True)

        #cleaning data set
        ## Step 1: Apply cleaning to the column
        df['_TC_clean'] = df[COL].map(clean_numstr)

        ## Step 2: Define numeric pattern and identify non-numeric rows
        numeric_pattern = re.compile(r"^[+-]?\d+(\.\d+)?$")

        mask_non_numeric = (
            df['_TC_clean'].notna() &
            (df['_TC_clean'] != "") &
            ~df['_TC_clean'].str.match(numeric_pattern)
        )

        ## Step 3: Split data into correct and incorrect subsets
        df3 = df[mask_non_numeric].reset_index(drop=True)   # Incorrect / non-numeric
        df2 = df[~mask_non_numeric].copy().reset_index(drop=True)  # Correct / numeric
        logger.info(f"Identified {df3.shape[0]} incorrect records with non-numeric '{COL}' values.")

        ## Step 4: Coerce numeric column in cleaned data
        df2[COL] = pd.to_numeric(df2['_TC_clean'], errors='coerce')

        ## Step 5: Perform Filtering
        df2 = df2[df2['Document Type'] != 'Repository'].reset_index(drop=True)
        df2['Published Year'] = pd.to_numeric(df2['Published Year'], errors='coerce')
        df2 = df2[df2['Published Year'] >= 2018].reset_index(drop=True)
        logger.info(f"After filtering, cleaned data shape: {df2.shape}")

        ## Step 5: Export incorrect records
        df3.to_csv(
            f'{ROOT}/Outbound/Error Record Reports/{SHEET_NAME}/incorrect_records_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            index=False
        )

        ## Step 6: Logging summary
        removed_records = df.shape[0] - df2.shape[0]
        logger.info(f"Total records removed during cleaning and filtering: {removed_records}")

        ## Step 7: Preview problematic values
        if not df3.empty:
            bad_vals = [repr(v) for v in df.loc[mask_non_numeric, COL].head(10).tolist()]
            logger.info(f"Sample offending raw values: {bad_vals}")

        #####################################
        #-----SAMPLING STRATEGY CALCULATION-----

        #sampling strategy
        logger.info("Calculating sampling strategy")

        #ONLY FOR TESTING PURPOSES - filter out records with 0 citations
        # curr_rec = df2.shape[0]
        # df2 = df2[df2[COL] > 0].reset_index(drop=True)
        # removed_records_0cit = curr_rec - df2.shape[0]

        ## Step 1: Value counts
        counts_df = df2[COL].value_counts().reset_index()
        counts_df.columns = ['Value', 'Count']
        counts_df['Value'] = counts_df['Value'].astype(int)
        counts_df['Count'] = counts_df['Count'].astype(int)

        ## Step 2: Calculate proportions and samples to select
        counts_df['Proportion'] = counts_df['Count'] / counts_df['Count'].sum()
        counts_df['Inverse_Proportion'] = 1 - counts_df['Proportion']
        counts_df['Denominator'] = pow(counts_df['Inverse_Proportion'].sum(), POW_NORM)
        counts_df['Perc_to_Select'] = counts_df['Inverse_Proportion'] / counts_df['Denominator']

        ## Step 2a: Identify elbow point using KneeLocator
        kl = KneeLocator(
            counts_df['Value'],
            counts_df['Count'],
            curve='convex',
            direction='decreasing',
            S=1.0
        )
        elbow_loc = kl.knee

        if elbow_loc is None:
            logger.warning("KneeLocator did not detect an elbow. Using minimum Count as fallback.")
            elbow_value = counts_df['Count'].min()
        else:
            elbow_value = counts_df.loc[counts_df['Value'] == elbow_loc, 'Count'].iloc[0]
            logger.info(f"Elbow point identified at Value: {elbow_loc}. Setting min sample to: {elbow_value}.")

        counts_df['Elbow_Value'] = elbow_value

        counts_df['Samples_to_Select'] = counts_df.apply(sample_size, axis=1).astype(int)

        counts_df.sort_values(by='Value', ascending=True, inplace=True)
        counts_df.to_excel(
            f'{ROOT}/Outbound/Sampling/{SHEET_NAME}/sampling_strategy_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            index=False
        )
        tot_records = counts_df['Samples_to_Select'].sum()
        logger.info(f"Total records to sample: {tot_records}")

        ##step 3: graphical representation
        logger.info("Generating sampling strategy plot")
        ## Coerce types, drop bad rows, aggregate duplicates, and sort
        tmp = counts_df[['Value', 'Count', 'Samples_to_Select']].copy()
        for c in ['Value', 'Count', 'Samples_to_Select']:
            tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
        tmp = tmp.dropna(subset=['Value', 'Count', 'Samples_to_Select'])

        agg = (
            tmp.groupby('Value', as_index=False)
            .agg(Count=('Count', 'sum'),
                 Samples_to_Select=('Samples_to_Select', 'sum'))
            .sort_values('Value')
            .reset_index(drop=True)
        )

        ## Build 1x2 subplots sharing the X axis
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        ## Left panel: Count
        axes[0].plot(agg['Value'], agg['Count'], color='tab:blue', linewidth=2, marker='o', markersize=3)
        axes[0].set_title('Count by Value')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Count')
        axes[0].set_xscale('log')          ## log-x for long tail
        axes[0].set_yscale('log')          ## helps readability with heavy head
        axes[0].grid(True, which='both', axis='both', alpha=0.25)

        ## Right panel: Samples to Select
        axes[1].plot(agg['Value'], agg['Samples_to_Select'], color='tab:orange', linewidth=2, linestyle='--', marker='s', markersize=3)
        axes[1].set_title('Samples to Select by Value')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Samples to Select')
        axes[1].set_xscale('log')          ## same x-scale for apples-to-apples
        axes[1].grid(True, which='both', axis='both', alpha=0.25)

        ## Overall title and save
        fig.suptitle('Distributions by Value (Side-by-Side)', y=1.03)
        fig.tight_layout()

        png_path = f"{ROOT}/Outbound/Sampling/{SHEET_NAME}/distributions_side_by_side_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

        logger.info(f"Saved plot to: {png_path}")

        #################################
        #----------SAMPLING EXECUTION----------
        logger.info("Executing sampling")
        sampled_dfs = []

        for idx, row in counts_df.iterrows():
            val = row['Value']
            samp_size = int(row['Samples_to_Select'])
            subset = df2[df2[COL] == val]

            if samp_size >= subset.shape[0]:
                sampled = subset.copy()
                logger.info(f"Value {val}: Requested {samp_size}, available {subset.shape[0]}. Taking all available.")
            else:
                sampled = subset.sample(n=samp_size, random_state=42)
                logger.info(f"Value {val}: Requested {samp_size}, available {subset.shape[0]}. Sampling {samp_size}.")

            sampled_dfs.append(sampled)

        sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
        sampled_df.to_excel(
            f'{ROOT}/Outbound/Sampling/{SHEET_NAME}/sampled_data_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            index=False
        )
        logger.info(f"Sampled data saved with shape: {sampled_df.shape}")

        ## Track used identifiers so replacements don't duplicate previously sampled rows
        used_dois = set(str(x).strip() for x in sampled_df['DOI'].fillna(''))
        used_urs  = set(str(x).strip() for x in sampled_df['UR'].fillna(''))

        ## Keep flags and logs for replacements/removals
        performed_replacements = False
        replacement_log = []
        removed_log = []          ## new: always log all-zero cases here
        removed_indices = []      ## records we will actually drop from sampled_df at the end

        ##################################
        #--------API SUBMISSION------        
        
        logger.info("Submitting sampled data to API")
        
        failed_calls = []

        ## Step 1: Iterate over sampled data
        for idx, row in sampled_df.iterrows():


            if MODE == 'rerun':
                if os.path.exists(f'{ROOT}/Outbound/Raw Data/{SHEET_NAME}/api_response_{idx}.json'):
                    logger.info(f"Row {idx} already has API response saved. Skipping due to rerun mode.")
                    continue

            doi = row['DOI']
            uri = row['UR']
            successful = False
            repeat_attempts = 0

            #step 2: API call with retry logic
            while not successful:
                if str(doi).strip() == '':
                    logger.warning(f"Row {idx} missing DOI ({doi}). Defaulting to URL ({uri}).")
                    param = str(uri).strip()
                else:
                    logger.info(f"Row {idx} using DOI ({doi}).")
                    param = str(doi).strip()

                response = requests.get(API_URL + param, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    result = response.json()

                    ## Check for the "all zero" condition
                    is_all_zero = False
                    try:
                        is_all_zero = all_scores_zero(result)
                    except Exception:
                        is_all_zero = False

                    if is_all_zero:
                        logger.warning(f"Row {idx}: all FAIR scores are zero. Attempting replacement in same citation bucket.")

                        ## Identify current bucket & current key
                        current_val = row[COL]
                        current_key = (str(row.get('DOI', '')).strip(), str(row.get('UR', '')).strip())

                        ## Always log that this record had all-zero FAIR scores (new requirement)
                        removed_log.append({
                            'Index': idx,
                            'DOI': current_key[0],
                            'UR': current_key[1],
                            'Total_Citations': current_val,
                            'Reason': 'All-zero FAIR scores'
                        })

                        ## Try to pick a replacement from the full universe df2
                        repl = pick_replacement(df2, used_dois, used_urs, current_val, current_key)

                        if repl is not None:
                            ## Update sampled_df row in-place with replacement
                            sampled_df.loc[idx, :] = repl

                            ## Update used sets
                            new_doi = str(repl.get('DOI', '')).strip()
                            new_ur  = str(repl.get('UR', '')).strip()
                            if new_doi: used_dois.add(new_doi)
                            if new_ur:  used_urs.add(new_ur)
                            performed_replacements = True

                            ## Re-run the API immediately for the replacement row
                            repl_param = new_doi if new_doi != '' else new_ur
                            logger.info(f"Row {idx}: replacement chosen. Re-calling API with {repl_param}")

                            success_repl = False
                            attempts_repl = 0
                            while not success_repl:
                                r2 = requests.get(API_URL + repl_param, timeout=REQUEST_TIMEOUT)
                                if r2.status_code == 200:
                                    result2 = r2.json()
                                    ## Overwrite the previous JSON for this index with the replacement response
                                    with open(f'{ROOT}/Outbound/Raw Data/{SHEET_NAME}/api_response_{idx}.json', 'w') as f:
                                        json.dump(result2, f, indent=4)
                                    logger.info(f"Row {idx}: replacement API call successful. Response saved.")

                                    jitter = random.uniform(0, JITTER_MAX)
                                    logger.info(f"Row {idx} sleeping for {jitter:.2f} seconds to avoid rate limiting.")
                                    time.sleep(jitter)
                                    success_repl = True
                                    successful = True
                                    replacement_log.append({
                                        'Index': idx,
                                        'Old_DOI': current_key[0],
                                        'Old_UR': current_key[1],
                                        'New_DOI': new_doi,
                                        'New_UR': new_ur,
                                        'Reason': 'All-zero FAIR scores'
                                    })
                                else:
                                    attempts_repl += 1
                                    logger.warning(f"Row {idx}: replacement API call failed with status {r2.status_code}. Retrying...")
                                    time.sleep(30)
                                    if attempts_repl > 1:
                                        logger.error(f"Row {idx}: replacement failed after 2 attempts; removing original all-zero record.")
                                        ## mark for deletion since we could not salvage it
                                        removed_indices.append(idx)
                                        successful = True
                            ## end while replacement
                        else:
                            logger.warning(f"Row {idx}: no eligible replacement found in citation bucket={current_val}. Removing record.")
                            ## Mark this row for deletion since we could not replace it
                            removed_indices.append(idx)
                            successful = True

                    else:
                        ## Normal case: save original response and continue
                        with open(f'{ROOT}/Outbound/Raw Data/{SHEET_NAME}/api_response_{idx}.json', 'w') as f:
                            json.dump(result, f, indent=4)
                        logger.info(f"Row {idx} API call successful. Response saved.")

                        jitter = random.uniform(0, JITTER_MAX)
                        logger.info(f"Row {idx} sleeping for {jitter:.2f} seconds to avoid rate limiting.")
                        time.sleep(jitter)
                        successful = True

                else:
                    logger.warning(f"Row {idx} API call failed with status code {response.status_code}. Retrying...")
                    time.sleep(30) #max jitter by default
                    repeat_attempts += 1

                    if (repeat_attempts > 1 and MODE == 'full') or (repeat_attempts > 0 and MODE == 'rerun') or (repeat_attempts > 0 and MODE == 'test'):
                        successful = True
                        logger.error(f"Row {idx} API call failed after {repeat_attempts} attempts. Logging failure.")
                        failed_calls.append(
                        {
                            'Index': idx,
                            'DOI': doi,
                            'UR': uri,
                            'Status_Code': response.status_code,
                            'Response_Text': response.text
                        }
                    )

            ## For testing purposes, limit to first 15 calls   
            if idx > 14 and MODE == 'test':
                break

        ## Step 3: Log failed calls
        failed_calls_df = pd.DataFrame(failed_calls)
        if not failed_calls_df.empty:
            failed_calls_df.to_excel(
                f'{ROOT}/Outbound/Error Record Reports/{SHEET_NAME}/api_failed_calls_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                index=False
            )
            logger.info(f"API failed calls logged with shape: {failed_calls_df.shape}")
        else:
            logger.info("All API calls successful.")

        #################################
        #--------APPLY REMOVALS / SAVE UPDATED SAMPLES----------

        ## drop rows that truly could not be salvaged
        if removed_indices:
            sampled_df = sampled_df.drop(index=removed_indices).reset_index(drop=True)
            logger.info(f"Removed {len(removed_indices)} sampled records due to all-zero FAIR with no viable replacement.")

        ## Save updated sampled data if replacements or removals were performed
        if performed_replacements or removed_indices:
            updated_path = f'{ROOT}/Outbound/Sampling/{SHEET_NAME}/sampled_data_UPDATED_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            sampled_df.to_excel(updated_path, index=False)
            logger.info(f"Updates performed. Updated sampled data saved to: {updated_path}")

        ## Save replacement log (if any)
        if replacement_log:
            repl_log_path = f'{ROOT}/Outbound/Sampling/{SHEET_NAME}/replacements_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            pd.DataFrame(replacement_log).to_csv(repl_log_path, index=False)
            logger.info(f"Replacement log saved to: {repl_log_path}")

        ## Save removed-records log (always write if any all-zero events occurred)
        if removed_log:
            removed_log_path = f'{ROOT}/Outbound/Sampling/{SHEET_NAME}/removed_records_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            pd.DataFrame(removed_log).to_csv(removed_log_path, index=False)
            logger.info(f"Removed-records log saved to: {removed_log_path}")

        #################################
        #--------ANALYSIS--------
        logger.info("Starting analysis phase")

        ## Step 1: Aggregate API responses
        json_directory = f'{ROOT}/Outbound/Raw Data/{SHEET_NAME}'
        metrics_arr = []
        
        for file in glob.glob(os.path.join(json_directory, 'api_response_*.json')):
            metrics_obj = {}
            with open(file, 'r') as f:
                data = json.load(f)
                metrics_obj['Source'] = data[0]['target_uri']

                for ob in data:
                    metrics_obj[ob['metric']] = ob['score']
            
            metrics_arr.append(metrics_obj) 

        metrics_df = pd.DataFrame(metrics_arr)
        metrics_df.to_excel(
            f'{ROOT}/Outbound/Results/{SHEET_NAME}/api_metrics_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            index=False
        )
        logger.info(f"API metrics saved with shape: {metrics_df.shape}")

        ## Step 2: Perform Calculations
        logger.info("Performing calculations on API metrics")

        metrics_df['FSum'] = metrics_df.apply(findable_sum_calc, axis=1)
        metrics_df['FPerc'] = metrics_df.apply(findable_perc_calc, axis=1)

        metrics_df['ASum'] = metrics_df.apply(accessible_sum_calc, axis=1)
        metrics_df['APerc'] = metrics_df.apply(accessible_perc_calc, axis=1)

        metrics_df['ISum'] = metrics_df.apply(interroperable_sum_calc, axis=1)
        metrics_df['IPerc'] = metrics_df.apply(interroperable_perc_calc, axis=1)

        metrics_df['RSum'] = metrics_df.apply(reusable_sum_calc, axis=1)
        metrics_df['RPerc'] = metrics_df.apply(reusable_perc_calc, axis=1)

        metrics_df['WeightedFAIRScore'] = metrics_df.apply(weighted_fair_score, axis=1)

        ## Step 3: Normalize identifiers and merge with sampled data metadata
        mini_sampled_df = sampled_df[['DOI', 'UR', 'Document Type', COL, 'Published Year']].copy()
        mini_sampled_df['DOI_norm'] = mini_sampled_df['DOI'].map(normalize_identifier)
        mini_sampled_df['UR_norm'] = mini_sampled_df['UR'].map(normalize_identifier)

        metrics_df['Source_norm'] = metrics_df['Source'].map(normalize_identifier)

        doi_df = pd.merge(
            mini_sampled_df,
            metrics_df,
            left_on='DOI_norm',
            right_on='Source_norm',
            how='inner'
        )
        ur_df = pd.merge(
            mini_sampled_df,
            metrics_df,
            left_on='UR_norm',
            right_on='Source_norm',
            how='inner'
        )
        final_df = pd.concat([doi_df, ur_df]).drop_duplicates().reset_index(drop=True)
        final_df.to_excel(
            f'{ROOT}/Outbound/Results/{SHEET_NAME}/final_analysis_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            index=False
        )
        logger.info(f"Final analysis saved with shape: {final_df.shape}")

    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        raise
        

if __name__ == "__main__":
    main()
