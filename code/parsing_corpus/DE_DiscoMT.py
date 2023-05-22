# Import necessary modules
import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import re
import numpy as np
import ast
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def extract_tokens_from_files(directory_path, output_file):
    """
    Extracts the tokenized words from all XML files in a directory.

    Parameters:
    directory_path (str): The path to the directory containing the XML files.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the tokenized words from all XML files.
    """
    # Get a list of all XML files in the directory
    file_paths = sorted(glob.glob(os.path.join(directory_path, '*.xml')))

    # Function to extract the relevant information from each file
    def extract_words(file_path):
        # Open the file
        with open(file_path, 'r') as f:
            # Read the contents of the file
            contents = f.read()
            # Use regular expressions to extract the words from the XML file
            words = re.findall(r'<word id="(.*?)">(.*?)</word>', contents)
            # Get the file ID from the file path and remove unnecessary parts
            file_id = os.path.basename(file_path).split(".")[0].replace("_words", "")
            # Return a list of tuples containing the file ID, word ID, and token for each word
            return [(file_id, w[0], w[1]) for w in words]

    # Tokenize each file and combine into a single list
    tokenized_words = []
    for file_path in file_paths:
        # Extract the words from the file
        words = extract_words(file_path)
        # Add the words to the list of tokenized words
        tokenized_words.extend(words)

    # Convert the list of words to a pandas DataFrame
    df = pd.DataFrame(tokenized_words, columns=['File id', 'word_id', 'token'])
    # Convert df to csv
    df.to_csv(output_file, index=False)

    # Return the DataFrame
    return df

def create_sentence_df(source_dir, markables_dir, file_suffix_sent, output_file):
    """
    Creates a pandas DataFrame containing sentence-level data from a directory of .tok files and
    a directory of corresponding markable files. Each row of the DataFrame represents a single
    sentence, with columns for sentence text, markable ID, markable span, order ID, MMax level, and
    file ID. The function also adds a new column for the sentence span as a list of indices.
    """
    # Use glob to find all .tok files in the directory
    source_files = sorted(glob.glob(f'{source_dir}/**/*.tok*', recursive=True))

    # Get a sorted list of file names in the markables directory
    markables_files = sorted([filename for filename in os.listdir(markables_dir) if filename.endswith(file_suffix_sent)])

    # Create an empty dictionary to store the sentence data
    sentence_data = {
        'Sentence Number': [],
        'Sentence': [],
        'ID': [],
        'Span': [],
        'Order ID': [],
        'MMax Level': [],
        'File id': [],
        'Span List': []  # Add new column for span list
    }

    # Loop through each source file and its corresponding markable file, and add the sentence data to the dictionary
    for source_file, markables_file in zip(source_files, markables_files):

        # Read the sentences from the source file
        with open(source_file, 'r') as f:
            sentences = f.readlines()

        # Create an empty dictionary to store sentence markables
        sentence_markables = {}
        namespaces = {'s': 'www.eml.org/NameSpaces/sentence'}
        # Parse the markables file using ElementTree
        tree = ET.parse(os.path.join(markables_dir, markables_file))
        root = tree.getroot()
        # Loop through all markables in the markables file
        for markable in root.findall('.//s:markable', namespaces):
            # Extract the markable ID, span, order ID, and MMAX level from the markable element
            markable_id = markable.attrib['id']
            markable_span = markable.attrib['span']
            markable_orderid = markable.attrib['orderid']
            markable_mmax_level = markable.attrib['mmax_level']
            # Store the markable information in the sentence_markables dictionary
            sentence_markables[markable_id] = {
                'span': markable_span,
                'orderid': markable_orderid,
                'mmax_level': markable_mmax_level,
            }
        
        # Extract the file ID from the source file path
        file_id = os.path.splitext(os.path.basename(source_file))[0]

        # Add the sentence data to the dictionary
        for i, sentence in enumerate(sentences): # Loop through each sentence and add its data to the sentence_data dictionary
            # Generate a unique sentence ID and markable ID  
            sentence_id = f'sentence_{i}'
            markable_id = f'markable_{i}'
            # Retrieve the markable data for the current sentence
            if markable_id in sentence_markables:
                markable_data = sentence_markables[markable_id]
            else:
                continue  # skip this markable

            # Check if the 'span' value contains '..' to separate a range of indices
            if '..' in markable_data['span']:
            # Extract the start and end indices of the sentence span from the markable data
                start, end = markable_data['span'].split('..')
            else:
                start = end = markable_data['span'].split('_')[-1]
            # Generate a list of indices corresponding to the sentence span
            span_list = list(range(int(start.split('_')[-1]), int(end.split('_')[-1])+1))
            # Add the sentence data to the sentence_data dictionary
            sentence_data['Sentence Number'].append(sentence_id)
            sentence_data['Sentence'].append(sentence.strip())
            sentence_data['ID'].append(markable_id)
            sentence_data['Span'].append(markable_data['span'])
            sentence_data['Order ID'].append(markable_data['orderid'])
            sentence_data['MMax Level'].append(markable_data['mmax_level'])
            sentence_data['File id'].append(file_id)
            sentence_data['Span List'].append(span_list)

    # Create a pandas DataFrame from the sentence_data dictionary
    sentence_df = pd.DataFrame(sentence_data)
    sentence_df.to_csv(output_file, index=False)
    return sentence_df

def get_coref_markables(markables_dir, file_suffix_cor, tokens_file, output_file):

    # Initialize an empty list to store the extracted coreferent entities
    coref_markables = []

    # Get a sorted list of file names in the directory
    file_list = sorted([filename for filename in os.listdir(markables_dir) if filename.endswith(file_suffix_cor)])

    # Iterate over each file in the sorted list
    for filename in file_list:
        namespaces = {
            "c": "www.eml.org/NameSpaces/coref"
        }
        # Parse the XML file using ElementTree
        tree = ET.parse(os.path.join(markables_dir, filename))

        # Get the root element of the tree
        root = tree.getroot()

        # Extract the file id from the filename
        file_id = filename.split(".")[0].replace("_coref_level", "")

        # Iterate over each "markable" element in the XML file using XPath and the namespace dictionary
        for markable in root.findall(".//c:markable", namespaces):
            markable_id = markable.attrib["id"]
            markable_span = markable.attrib["span"]
            markable_type_of_pronoun = markable.get("type_of_pronoun")
            markable_agreement =  markable.attrib.get("agreement")
            markable_npmod = markable.attrib.get("npmod")
            markable_split = markable.attrib.get("split")
            markable_coref_class = markable.attrib["coref_class"]
            markable_comparative = markable.attrib.get("comparative")
            markable_mmax_level = markable.attrib["mmax_level"]
            markable_vptype = markable.attrib.get("vptype")
            markable_position = markable.attrib.get("position")
            markable_type = markable.attrib.get("type")
            markable_antetype = markable.attrib.get("antetype")
            markable_anacata = markable.attrib.get('anacata')
            markable_mention = markable.attrib.get("mention")


            # If the span of the markable element is a range of numbers, convert it into a list of integers
            span_list = []
            if ".." in markable_span: # Check if the markable span contains a range of values
                spans = markable_span.split(",") # Split the span into parts by comma
                for span in spans:  # Loop through each part of the span
                    if ".." in span: # Check if the span contains a range of values
                        start_str = span.split("..")[0].split("_")[1] # Get the starting string of the span
                        end_str = span.split("..")[-1].split("_")[1] # Get the ending string of the span
                        start_num = int(re.search(r'\d+', start_str).group()) # Get the starting number of the span
                        end_num = int(re.search(r'\d+', end_str).group()) # Get the ending number of the span
                        span_list.extend(list(range(start_num, end_num+1))) # Generate a list of numbers and add it to span_list
                    else:
                        span_list.append(int(span.split("_")[1].split(",")[0])) # Add the number to span_list
            else:
                span_list.append(int(markable_span.split("_")[1].split(",")[0])) # Add the number to span_list
            
            
    
            # Create a dictionary for the coreference markable and add it to the list
            coref_markables.append({
                "File id": file_id,
                "ID_coref": markable_id,
                "Span_coref": markable_span,
                "Type_of_pronoun": markable_type_of_pronoun,
                "Agreement":markable_agreement,
                "Npmod":markable_npmod,
                "Split":markable_split,
                "Coref Class": markable_coref_class,
                "Comparative": markable_comparative,
                "Mmax Level": markable_mmax_level,
                "Vptype": markable_vptype,
                "Position":markable_position,
                "Type": markable_type,
                "Antetype":markable_antetype,
                "Anacata":markable_anacata,
                "Mention": markable_mention,
                "Span List Coref": span_list, # Add the span list to the dictionary
            })

        # Create a DataFrame from the coreference markables
        coref_markables_df = pd.DataFrame(coref_markables)
        # Sort the DataFrame by 'File id' and 'Span_coref'
        coref_markables_df = coref_markables_df.sort_values(by=["File id", "Span_coref"])


    def get_word_ids(span_str):
    # check if span is a range
        if ',' in span_str:
            word_ids = []
            # split the span string by comma
            span_parts = span_str.split(',')
            for span_part in span_parts:
                # get word ids from each part of the span
                word_ids.extend(get_word_ids(span_part))
            return word_ids

        # split span into start and end indices
        if '..' in span_str:
            start, end = span_str.split('..')
            start_index = int(start.split('_')[1]) - 1
            end_index = int(end.split('_')[1])
            # return a list of word ids from start to end index
            return [f"word_{i+1}" for i in range(start_index, end_index)]
        else:
            # if span is a single word, return its word id
            return [span_str]
    df = pd.read_csv(tokens_file)
    # Add a new column to the DataFrame, 'Tokens_coref', which contains a list of tokens corresponding to the coreference span
    def map_tokens_to_span(row):
        file_id = row['File id']
        span = row['Span_coref']
        df_subset = df[df['File id'] == file_id]
        return [df_subset.loc[df_subset['word_id'] == word_id, 'token'].iloc[0] if len(df_subset.loc[df_subset['word_id'] == word_id]) > 0 else np.nan for word_id in get_word_ids(span) if get_word_ids(span) is not None]

    coref_markables_df['Tokens_coref'] = coref_markables_df.apply(map_tokens_to_span, axis=1)
    coref_markables_df.to_csv(output_file, index=False)
    return coref_markables_df   

def merge_data(coref_file, sent_file, output_file):
    # read in the csv files
    coref_df = pd.read_csv(coref_file)
    sent_df = pd.read_csv(sent_file)

    # add a new column for coreference info to sent_df
    sent_df['coreference_info'] = ''

    # iterate over each row of coref_df
    for index, row in coref_df.iterrows():
        # get the span list and file id from the current row
        span_list = row['Span List Coref']
        file_id = row['File id']

        # find the rows in sent_df with the same file id
        sent_rows = sent_df[sent_df['File id'] == file_id]

        # convert span list to a list of integers
        span_list_int = [int(s) for s in re.findall(r'\d+', span_list)]

        # iterate over each row in sent_rows
        for s_index, s_row in sent_rows.iterrows():
            # convert sentence span list to a list of integers
            sent_span_list_int = [int(s) for s in re.findall(r'\d+', s_row['Span List'])]
            # check if any value in span list exists in the span list of the sentence row
            if any(span in sent_span_list_int for span in span_list_int):
                # create a list from the row in coref_df
                coref_list = row.tolist()

                # remove the first two columns (file id and coref id)
                coref_list = coref_list[1:]

                # create dictionary
                keys = ["ID_coref","Span_coref", "Type_of_pronoun", "Agreement","Npmod","Split","Coref Class", "Comparative", "Mmax Level","Vptype","Position", "Type", "Antetype","Anacata","Mention","Span List Coref","Tokens_Coref"]
                coref_list = [item if isinstance(item, str) else "NA" for item in coref_list]
                coref_dict = dict(zip(keys, coref_list))
                #add comma between the dictionaries
                coref_str = str(coref_dict) + ','
                # add the coreference info to the coreference_info column of the sentence row
                sent_df.at[s_index, 'coreference_info'] = sent_df.at[s_index, 'coreference_info'] + str(coref_str)

    # drop Span List column
    sent_df = sent_df.drop(['Span List'], axis=1)

    # write the merged dataframe to a new csv file
    sent_df.to_csv(output_file, index=False)
    return sent_df

def sorting_by_coreference_class(input_file, output_file):
    # read input file into a pandas dataframe
    df = pd.read_csv(input_file)
    # fill any missing coreference_info values with an empty dictionary
    df['coreference_info'] = df['coreference_info'].fillna('{}')
    # create a new dataframe to hold the separated dictionaries
    new_df = pd.DataFrame(columns=['Sentence Number','Sentence', 'Sentence ID', 'Span', 'Order ID', 'MMax Level', 'File id'])
    # iterate over each row in the input dataframe
    for _, row in df.iterrows():
        # extract the coreference info from the row
        coref_info = ast.literal_eval(row['coreference_info'])
        # add a new row to the output dataframe for each entry in the coreference info
        for info_dict in coref_info:
            new_row = {
                'Sentence Number': row['Sentence Number'],
                'Sentence': row['Sentence'],
                'Sentence ID': row['ID'],
                'Span': row['Span'],
                'Order ID': row['Order ID'],
                'MMax Level': row['MMax Level'],
                'File id': row['File id'],
            }
            new_row.update(info_dict)
            new_df = new_df.append(new_row, ignore_index=True)
    # sorting the dataframe based on file id and coreference class columns
    new_df = new_df.sort_values(by=['File id', 'Coref Class'],ascending=True) 
    # write the output dataframe to a csv file
    new_df.to_csv(output_file, index=False)
    return new_df

def main():
    # directory containing the files to be copied and renamed
    source_dir_de = "/home/user/Documents/GitHub/CA-NMT_evaluation/parcor-full/corpus/DiscoMT/DE/Source/sentence"
    # new directory to store the copied and renamed files
    dest_dir_de = "/home/user/Documents/GitHub/CA-NMT_evaluation/parcor-full/corpus/DiscoMT/DE/Source/sentence"


    # create the new directory if it doesn't already exist
    if not os.path.exists(dest_dir_de):
        os.makedirs(dest_dir_de)

    # dictionary mapping old names to new names
    name_dict_en = {
        "talk000205.de-en.de": "010_205.tok",
        "talk001756.de-en.de": "000_1756.tok",
        "talk001819.de-en.de": "001_1819.tok",
        "talk001825.de-en.de": "002_1825.tok",
        "talk001894.de-en.de": "003_1894.tok",
        "talk001938.de-en.de": "005_1938.tok",
        "talk001950.de-en.de": "006_1950.tok",
        "talk001953.de-en.de": "007_1953.tok",
        "talk002043.de-en.de": "009_2043.tok",
        "talk002053.de-en.de": "011_2053.tok",
    }
    # loop through the files in the source directory
    for filename in os.listdir(source_dir_de):
        # check if the file is in the name dictionary
        if filename in name_dict_en:
            # construct the old and new file paths
            old_path = os.path.join(source_dir_de, filename)
            new_path = os.path.join(dest_dir_de, name_dict_en[filename])
            # copy the file and rename it
            shutil.copy(old_path, new_path)

    # Set the path to the directory containing the tokenised sentences
    source_dir = '/home/user/Documents/GitHub/CA-NMT_evaluation/parcor-full/corpus/DiscoMT/DE/Source/sentence'
    # Set the path to the directory containing the markables
    markables_dir = '/home/user/Documents/GitHub/CA-NMT_evaluation/parcor-full/corpus/DiscoMT/DE/Markables'
    # Set the path to the directory containing the basedata (word_level)
    basedata_dir ='/home/user/Documents/GitHub/CA-NMT_evaluation/parcor-full/corpus/DiscoMT/DE/Basedata/'

    
    # Set the file suffix for sentence markables
    file_suffix_sent = 'sentence_level.xml'
    # Set the file suffix for coreference markables
    file_suffix_cor = "coref_level.xml"
    
    # Call the extract_tokens_from_files function
    tokens_df = extract_tokens_from_files(basedata_dir,"/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/tokens_DiscoMT_de.csv") # extract_tokens_from_files(basedata_dir)
    print("Processing basedata...")
    print(tokens_df)
    # Call the create_sentence_df function
    sentences_df = create_sentence_df(source_dir, markables_dir, file_suffix_sent, '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/sentence_data_DiscoMT_de.csv')
    print("Processing sentence data...")
    print(sentences_df)
    # Call the get_coref_markables function
    coreferences_df = get_coref_markables(markables_dir, file_suffix_cor, '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/tokens_DiscoMT_de.csv', '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/coref_markables_DiscoMT_de.csv')
    print("Processing coreference markables...")
    print(coreferences_df)
    # Call the merge_data function
    merged_df = merge_data('/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/coref_markables_DiscoMT_de.csv', '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/sentence_data_DiscoMT_de.csv', '/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/merged_data_DiscoMT_de.csv')
    print("Processing merged data...")
    print(merged_df)
    # Call the sorting_by_coreference_class function
    merged_data_sorted_by_coref_class_df = sorting_by_coreference_class("/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/merged_data_DiscoMT_de.csv", "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/DE/DiscoMT/merged_data_sorted_by_coref_class_DiscoMT_de.csv")
    print("Processing merged data sorted by coreference class...")
    print(merged_data_sorted_by_coref_class_df)
    
if __name__ == '__main__':
    main()

