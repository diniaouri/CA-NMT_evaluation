# Import necessary modules
import pandas as pd
import os
import glob
import xml.etree.ElementTree as ET
import re
import numpy as np
import ast
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def csv_to_txt(csv_file_path, txt_file_path):
    """
    In this function we convert csv to txt so that data 
    are in an appropriate form to be fed to the CA-NMT model 
    """
    # Read CSV file into DataFrame and select "Sentence" column
    # Use the pandas read_csv function to read the CSV file at the specified file path
    df = pd.read_csv(csv_file_path)
    # Select the "File id" and "Sentence" columns from the DataFrame using bracket notation
    df = df[["File id", "Sentence"]]

    # Write DataFrame to text file
    with open(txt_file_path, "w") as f:     # Use the open function to open the specified text file for writing
        for i, row in df.iterrows():    # Loop over each row in the DataFrame using the iterrows function
            # Write each sentence to the file
            f.write(row["Sentence"] + "\n")     # Write the "Sentence" value for each row to the text file followed by a newline character
            # Write a blank line after each file (except the last one)
            if i < len(df) - 1 and df.iloc[i+1]["File id"] != row["File id"]:     # If the next row has a different "File id" value, write an additional newline character
                f.write("\n")     # This adds a blank line between groups of sentences belonging to the same file
        # Write a blank line after the last sentence
        f.write("\n")

def merge_csv_files_EN_DE(en_path, de_path, out_path):
    '''
    In this function we merge the english and german data in one csv file 
    '''

    # Read in the csv files
    df_en = pd.read_csv(en_path)
    df_de = pd.read_csv(de_path)

    # Extract the required columns from both dataframes
    df_en = df_en[['Sentence', 'File id', 'Mention','Order ID','ID_coref', 'Span List Coref','Tokens_Coref', 'Coref Class']]
    df_de = df_de[['Sentence', 'File id','Mention','Order ID', 'ID_coref','Span List Coref','Tokens_Coref', 'Coref Class']]

    # Rename the columns in the French dataframe to match the English dataframe
    df_en.columns = ['Sentence_en', 'File id','Mention', 'Order ID', 'ID_coref_en', 'Span List Coref_en', 'Tokens_Coref_en', 'Coref Class_en',]
    df_de.columns = ['Sentence_de', 'File id','Mention', 'Order ID', 'ID_coref_de', 'Span List Coref_de', 'Tokens_Coref_de', 'Coref Class_de']

    # Cast the Order ID column to integer to sort it numerically
    df_en['Order ID'] = df_en['Order ID'].astype(int)
    df_de['Order ID'] = df_de['Order ID'].astype(int)
    
    # Merge the dataframes based on the common 'File id', 'Order ID', 'Mention' information
    merged = pd.merge(df_en, df_de, on=['File id', 'Order ID', 'Mention'], how='outer', suffixes=('_en', '_de'))

    # Sort the merged dataframe by 'File id', 'Order ID', 'Mention'
    merged = merged.sort_values(by=['File id', 'Order ID', 'Mention'])

    # Fill missing values with 'NA'
    merged.fillna('NA', inplace=True)

    # Remove any duplicate rows (if any)
    merged.drop_duplicates(inplace=True)

    # Reset the index
    merged.reset_index(drop=True, inplace=True)

    # Reorder the columns in the merged dataframe
    merged = merged[['File id', 'Order ID','Mention', 'Sentence_en', 'ID_coref_en', 'Span List Coref_en', 'Tokens_Coref_en', 'Coref Class_en', 'Sentence_de', 'ID_coref_de','Span List Coref_de','Tokens_Coref_de', 'Coref Class_de']]
    # Save the merged dataframe to a csv file
    merged.to_csv(out_path, index=False)

    # Read the merged file and return the resulting dataframe
    return pd.read_csv(out_path)

def merge_csv_files_EN_FR(en_path, fr_path, out_path):
    '''
    In this function we merge english and french data in one csv file
    '''
    # Read in the csv files
    df_en = pd.read_csv(en_path)
    df_fr = pd.read_csv(fr_path)

    # Extract the required columns from both dataframes
    df_en = df_en[['Sentence', 'File id', 'Order ID', 'Tokens_Coref', 'Coref Class', 'ID_coref']]
    df_fr = df_fr[['Sentence', 'File id', 'Order ID', 'Tokens_Coref', 'Coref Class', 'ID_coref']]

    # Rename the columns in the French dataframe to match the English dataframe
    df_en.columns = ['Sentence_en', 'File id', 'Order ID', 'Tokens_Coref_en', 'Coref Class_en',"ID_coref"]
    df_fr.columns = ['Sentence_fr', 'File id', 'Order ID', 'Tokens_Coref_fr', 'Coref Class_fr', "ID_coref"]

    # Cast the Order ID column to integer to sort it numerically
    df_en['Order ID'] = df_en['Order ID'].astype(int)
    df_fr['Order ID'] = df_fr['Order ID'].astype(int)

    # Merge the dataframes horizontally
    merged = pd.merge(df_en, df_fr, on=['File id', 'Order ID', 'ID_coref'], how='outer')

    # Fill missing values in Order ID column with 'NA'
    merged['Order ID'] = merged['Order ID'].fillna('NA')

    # Reorder the columns in the merged dataframe
    merged = merged[['File id', 'Order ID','ID_coref', 'Sentence_en', 'Tokens_Coref_en', 'Coref Class_en', 'Sentence_fr', 'Tokens_Coref_fr', 'Coref Class_fr']]

    # Sort the dataframe by Order ID
    merged = merged.sort_values(by=['File id','Order ID', "ID_coref"])

    # Fill in missing values with 'NA'
    merged.fillna('NA', inplace=True)

    # Save the merged dataframe to a csv file
    merged.to_csv(out_path, index=False)

    # Read the merged file and return the resulting dataframe
    return pd.read_csv(out_path)

def map_first_pos_en_with_first_pos_de(file_path, output_path):

    """ 
    
    With this function we wanted to test whether combining english with german data by mapping the first occurance 
    of pos in one language with the corresponding one in the other language would be helpful, we decided it is not, yet the function is kept.
    It maps the mentions based on their pos information and their position in the sentence (their order)
    
    """

    merged = pd.read_csv(file_path)
    
    # replace invalid string with NaN
    merged['Span List Coref_en'] = merged['Span List Coref_en'].replace('invalid_string', np.nan)

    # define function to apply to the column
    def convert_to_list(x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    # apply function to the column
    merged['Span List Coref_en'] = merged['Span List Coref_en'].apply(convert_to_list)

    # replace invalid string with NaN
    merged['Span List Coref_de'] = merged['Span List Coref_de'].replace('invalid_string', np.nan)

    # define function to apply to the column
    def convert_to_list(x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    # apply function to the column
    merged['Span List Coref_de'] = merged['Span List Coref_de'].apply(convert_to_list)

    merged.replace('', np.nan, inplace=True) # replace empty strings with null values
    merged_filtered = merged.dropna(subset=['Span List Coref_en', 'Span List Coref_de'])
    merged_filtered['Coref_en_first'] = [lst[0] if len(lst) > 0 else '' for lst in merged_filtered['Span List Coref_en']]
    merged_filtered['Coref_de_first'] = [lst[0] if len(lst) > 0 else '' for lst in merged_filtered['Span List Coref_de']]

    merged_filtered['Coref_en_first'] = pd.to_numeric(merged_filtered['Coref_en_first'], errors='coerce')
    merged_filtered['Coref_de_first'] = pd.to_numeric(merged_filtered['Coref_de_first'], errors='coerce')
    merged_filtered['coref_diff'] = abs(merged_filtered['Coref_en_first'] - merged_filtered['Coref_de_first'])

    seen_numbers = []
    for index, row in merged_filtered.sort_values(['File id', 'Order ID', 'Mention']).iterrows():
        coref_en_first, coref_de_first = sorted([row['Coref_en_first'], row['Coref_de_first']])
        if coref_en_first in seen_numbers or coref_de_first in seen_numbers:
            merged_filtered.drop(index, inplace=True)
            continue
        seen_numbers.append(coref_en_first)
        seen_numbers.append(coref_de_first)
    return merged_filtered.to_csv(output_path, index=False)

def sort_EN_FR_based_on_coreference_class(input_file_path, output_file_path):
    """

    In this function we wanted to sort and group the english and french data 
    based on the coreference class information for better inspection of the data

    """
    # Read the input CSV file
    df = pd.read_csv(input_file_path)

    # Use pandas' astype method to convert the 'Coref Class_en' and 'Coref Class_fr' columns to string
    df['Coref Class_en'] = df['Coref Class_en'].astype(str)
    df['Coref Class_fr'] = df['Coref Class_fr'].astype(str)

    # Use pandas' str.extract method to extract the numeric values from the 'Coref Class' columns
    df['Coref Class_en'] = df['Coref Class_en'].str.extract(r'set_(\d+)')
    df['Coref Class_fr'] = df['Coref Class_fr'].str.extract(r'set_(\d+)')

    # Fill values that equal 'empty' with NaN
    df['Coref Class_en'] = df['Coref Class_en'].replace('empty', pd.np.nan)
    df['Coref Class_fr'] = df['Coref Class_fr'].replace('empty', pd.np.nan)

    # Convert the resulting strings to integer values using the astype method
    df['Coref Class_en'] = pd.to_numeric(df['Coref Class_en'], errors='coerce')
    df['Coref Class_fr'] = pd.to_numeric(df['Coref Class_fr'], errors='coerce')

    # Fill missing values with 0
    df['Coref Class_en'].fillna(100000, inplace=True)
    df['Coref Class_fr'].fillna(100000, inplace=True)

    # Convert the resulting float values to integer values using the astype method
    df['Coref Class_en'] = df['Coref Class_en'].astype(int)
    df['Coref Class_fr'] = df['Coref Class_fr'].astype(int)

    # Group the rows by File id and Coref Class_en
    grouped = df.groupby(['File id', 'Coref Class_en'])

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Iterate through each group
        for name, group in grouped:
            file_id, coref_class = name

            # Create a dictionary for the group
            group_dict = {}

            # Iterate through each row in the group
            for index, row in group.iterrows():
                # Add the row information to the dictionary
                order_id = row['Order ID']
                if order_id in group_dict:
                    group_dict[order_id].append({
                        'ID_coref': row['ID_coref'],
                        'Sentence_en': row['Sentence_en'],
                        'Tokens_Coref_en': row['Tokens_Coref_en'],
                        'Coref Class_en': row['Coref Class_en'],
                        'Sentence_fr': row['Sentence_fr'],
                        'Tokens_Coref_fr': row['Tokens_Coref_fr'],
                        'Coref Class_fr': row['Coref Class_fr']
                    })
                else:
                    group_dict[order_id] = [{
                        'ID_coref': row['ID_coref'],
                        'Sentence_en': row['Sentence_en'],
                        'Tokens_Coref_en': row['Tokens_Coref_en'],
                        'Coref Class_en': row['Coref Class_en'],
                        'Sentence_fr': row['Sentence_fr'],
                        'Tokens_Coref_fr': row['Tokens_Coref_fr'],
                        'Coref Class_fr': row['Coref Class_fr']
                    }]

            # Write the group header to the output file
            writer.writerow([f"File id: {file_id}", f"Coref Class_en: {coref_class}"])

            # Write the header row to the output file
            writer.writerow(['Order ID', 'ID_coref', 'Sentence_en', 'Tokens_Coref_en', 'Coref Class_en', 'Sentence_fr', 'Tokens_Coref_fr', 'Coref Class_fr'])

            # Write the rows for each Order ID to the output file
            for order_id, row_dicts in group_dict.items():
                for row_dict in row_dicts:
                    writer.writerow([order_id] + list(row_dict.values()))

            # Add an empty line between each group
            writer.writerow([])

def sort_EN_DE_based_on_coreference_class(input_file_path, output_file_path):

    """

    In this function we wanted to sort and group the english and german data 
    based on the coreference class information for better inspection of the data
    Note that the mapping is done based on the Coreference class information of English 
    which is different from the one in German

    """
    # Read the input CSV file
    df = pd.read_csv(input_file_path)

    # Use pandas' astype method to convert the 'Coref Class_en' and 'Coref Class_fr' columns to string
    df['Coref Class_en'] = df['Coref Class_en'].astype(str)
    df['Coref Class_de'] = df['Coref Class_de'].astype(str)

    # Use pandas' str.extract method to extract the numeric values from the 'Coref Class' columns
    df['Coref Class_en'] = df['Coref Class_en'].str.extract(r'set_(\d+)')
    df['Coref Class_de'] = df['Coref Class_de'].str.extract(r'set_(\d+)')

    # Fill values that equal 'empty' with NaN
    df['Coref Class_en'] = df['Coref Class_en'].replace('empty', pd.np.nan)
    df['Coref Class_de'] = df['Coref Class_de'].replace('empty', pd.np.nan)

    # Convert the resulting strings to integer values using the astype method
    df['Coref Class_en'] = pd.to_numeric(df['Coref Class_en'], errors='coerce')
    df['Coref Class_de'] = pd.to_numeric(df['Coref Class_de'], errors='coerce')

    # Fill missing values with 0
    df['Coref Class_en'].fillna(100000, inplace=True)
    df['Coref Class_de'].fillna(100000, inplace=True)


    # Convert the resulting float values to integer values using the astype method
    df['Coref Class_en'] = df['Coref Class_en'].astype(int)
    df['Coref Class_de'] = df['Coref Class_de'].astype(int)


    # Group the rows by File id and Coref Class_en
    grouped = df.groupby(['File id', 'Coref Class_en'])

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Iterate through each group
        for name, group in grouped:
            file_id, coref_class = name

            # Create a dictionary for the group
            group_dict = {}

            # Iterate through each row in the group
            for index, row in group.iterrows():
                # Add the row information to the dictionary
                order_id = row['Order ID']
                if order_id in group_dict:
                    group_dict[order_id].append({
                        'Mention': row['Mention'],
                        'Sentence_en': row['Sentence_en'],
                        'ID_coref_en': row['ID_coref_en'],
                        'Tokens_Coref_en': row['Tokens_Coref_en'],
                        'Coref Class_en': row['Coref Class_en'],
                        'Sentence_de': row['Sentence_de'],
                        'ID_coref_de': row['ID_coref_de'],
                        'Tokens_Coref_de': row['Tokens_Coref_de'],
                        'Coref Class_de': row['Coref Class_de']
                    })
                else:
                    group_dict[order_id] = [{
                        'Mention': row['Mention'],
                        'Sentence_en': row['Sentence_en'],
                        'ID_coref_en': row['ID_coref_en'],
                        'Tokens_Coref_en': row['Tokens_Coref_en'],
                        'Coref Class_en': row['Coref Class_en'],
                        'Sentence_de': row['Sentence_de'],
                        'ID_coref_de': row['ID_coref_de'],
                        'Tokens_Coref_de': row['Tokens_Coref_de'],
                        'Coref Class_de': row['Coref Class_de']
                    }]

            # Write the group header to the output file
            writer.writerow([f"File id: {file_id}", f"Coref Class_en: {coref_class}"])

            # Write the header row to the output file
            writer.writerow(['Order ID', 'Mention', 'Sentence_en','ID_coref_en', 'Tokens_Coref_en', 'Coref Class_en', 'Sentence_de', 'ID_coref_de','Tokens_Coref_de', 'Coref Class_de'])

            # Write the rows for each Order ID to the output file
            for order_id, row_dicts in group_dict.items():
                for row_dict in row_dicts:
                    writer.writerow([order_id] + list(row_dict.values()))

            # Add an empty line between each group
            writer.writerow([])





def main():


    ###  csv_to_txt ###
    # GERMAN
    # DiscoMT
    csv_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/sentence_data_DiscoMT_de.csv"
    txt_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/DiscoMT_sent_de.txt"
    # Calling the function csv_to_txt
    csv_to_txt(csv_file_path, txt_file_path)

    # news
    csv_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/news/sentence_data_news_de.csv"
    txt_file_path ="/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/news/news_sent_de.txt"
    # Calling the function csv_to_txt
    csv_to_txt(csv_file_path, txt_file_path)

    # Combine contents of both text files into a single file because we have DiscoMT and news data
    with open("/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT_news/DE.txt", "w") as f:
        for file_name in ["/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/DiscoMT_sent_de.txt", "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/news/news_sent_de.txt"]:
            with open(file_name, "r") as current_file:
                f.write(current_file.read())
    
    ### csv_to_txt ###
    # ENGLISH
    # DiscoMT
    csv_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/DiscoMT/sentence_data_DiscoMT_en.csv"
    txt_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/DiscoMT/DiscoMT_sent_en.txt"
    # Calling the function csv_to_txt
    csv_to_txt(csv_file_path, txt_file_path)
    
    # news 
    csv_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/news/sentence_data_news_en.csv"
    txt_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/news/news_sent_en.txt"
    # Calling the function csv_to_txt
    csv_to_txt(csv_file_path, txt_file_path)

    # Combine contents of both text files into a single file because we have DiscoMT and news data 
    with open("/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/DiscoMT_news/EN.txt", "w") as f:
        for file_name in ["/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/DiscoMT/DiscoMT_sent_en.txt", "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/news/news_sent_en.txt"]:
            with open(file_name, "r") as current_file:
                f.write(current_file.read())
    
    ### csv_to_txt ###
    # FRENCH
    # TED 
    csv_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/FR/TED/sentence_data_TED_fr.csv"
    txt_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/FR/TED/FR.txt"
    # Calling the function csv_to_txt
    csv_to_txt(csv_file_path, txt_file_path)

    ### csv_to_txt ###
    # ENGLISH
    # TED
    csv_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/TED/sentence_data_TED_en.csv"
    txt_file_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/TED/EN.txt"
    # Calling the function csv_to_txt
    csv_to_txt(csv_file_path, txt_file_path)






    ### merge_csv_files_EN_DE ###
    # merging csv files for DiscoMT and news in German
    # Read in the csv files
    df_de_DiscoMT = pd.read_csv('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/merged_data_sorted_by_coref_class_DiscoMT_de.csv')
    df_de_news = pd.read_csv('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/news/merged_data_sorted_by_coref_class_news_de.csv')
    # Concatenate the two dataframes vertically
    merged_de = pd.concat([df_de_DiscoMT, df_de_news], ignore_index=True)
    # Save the merged dataframe to a csv file
    merged_de.to_csv('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT_news/merged_data_sorted_by_coref_class_DiscoMT_news_de.csv', index=False)

    # merging csv files for DiscoMT and news in English
    # Read in the csv files
    df_en_DiscoMT = pd.read_csv('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/DiscoMT/merged_data_sorted_by_coref_class_DiscoMT_en.csv')
    df_en_news = pd.read_csv('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/news/merged_data_sorted_by_coref_class_news_en.csv')
    # Concatenate the two dataframes vertically
    merged_en = pd.concat([df_en_DiscoMT, df_en_news], ignore_index=True)
    # Save the merged dataframe to a csv file
    merged_en.to_csv('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/DiscoMT_news/merged_data_sorted_by_coref_class_DiscoMT_news_en.csv', index=False)

    en_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/DiscoMT_news/merged_data_sorted_by_coref_class_DiscoMT_news_en.csv"
    de_path = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT_news/merged_data_sorted_by_coref_class_DiscoMT_news_de.csv"
    out_path = '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-DE/en_de.csv'
    # calling the merge_csv_files_EN_DE function (here we want to combine the English and German data)
    df_en_de = merge_csv_files_EN_DE(en_path, de_path, out_path)
    # cleaner version without NA
    # Drop the rows that have at least one missing value
    df_en_de.dropna(axis=0, how='any', inplace=True)
    df_en_de.drop(["Span List Coref_en", "Span List Coref_de"],inplace=True, axis=1)
    df_en_de.to_csv('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-DE/en_de_clean.csv', index=False)






    ### merge_csv_files_EN_FR ###
    en_path = '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/TED/merged_data_sorted_by_coref_class_TED_en.csv'
    fr_path = '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/FR/TED/merged_data_sorted_by_coref_class_TED_fr.csv'
    out_path = '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-FR/en_fr.csv'
    # calling the merge_csv_files_EN_FR function (here we want to combine the english and french data)
    df_en_fr = merge_csv_files_EN_FR(en_path, fr_path, out_path)

    # cleaner version without NA
    # Drop the rows that have at least one missing value
    df_en_fr.dropna(axis=0, how='any', inplace=True)
    df_en_fr.to_csv('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-FR/en_fr_clean.csv', index=False)




    ### map_first_pos_en_with_first_pos_de ### 
    # calling the map_first_pos_en_with_first_pos_de function
    map_first_pos_en_with_first_pos_de('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-DE/en_de.csv', '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-DE/en_de_pos_order_of_appearance.csv')




    ### sort_EN_FR_based_on_coreference_class function ###
    # calling the sort_EN_FR_based_on_coreference_class function
    sort_EN_FR_based_on_coreference_class('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-FR/en_fr.csv', '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-FR/en_fr_sorted_by_coreference_class.csv')  





    ### sort_EN_DE_based_on_coreference_class function ###
    # calling the sort_EN_DE_based_on_coreference_class function
    sort_EN_DE_based_on_coreference_class('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-DE/en_de.csv', '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN-DE/en_de_sorted_by_coreference_class.csv')  




if __name__ == '__main__':
    main()
