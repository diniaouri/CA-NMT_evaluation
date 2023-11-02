import torch
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def translate_and_convert_to_csv(input_file, output_csv, model_name="Helsinki-NLP/opus-mt-en-de", input_delimiter='\t'):
    # Step 1: Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Step 2: Read the input English sentences from the file
    with open(input_file, "r") as file:
        english_sentences = file.read().splitlines()

    # Step 3: Tokenize the input English sentences
    tokenized_inputs = tokenizer(english_sentences, return_tensors="pt", padding=True, max_length=512, truncation=True)

    # Step 4: Perform translation
    with torch.no_grad():
        output = model.generate(**tokenized_inputs, max_new_tokens=512)

    # Step 5: Decode the translated output
    translated_sentences = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Step 6: Save the translations to a new file
    with open(output_csv, "w") as file:
        for sentence in translated_sentences:
            file.write(sentence + "\n")

    print(f"Translations have been saved to '{output_csv}'.")

    # Step 7: Convert the translations to a CSV file
    with open(output_csv, 'r') as txt_file, open(output_csv, 'w', newline='') as csv_file:
        # Create a CSV writer object.
        csv_writer = csv.writer(csv_file)

        # Read each line from the input TXT file.
        for line in txt_file:
            # Split the line based on the delimiter to get individual values.
            values = line.strip().split(input_delimiter)
            # Write the values to the CSV file.
            csv_writer.writerow(values)

    print(f"Conversion successful. The CSV file '{output_csv}' has been created.")

def main():
    # Replace these paths with your actual input and output file paths
    input_file = "/home/user/Documents/GitHub/CA-NMT_evaluation/parsed_data/EN/DiscoMT_news/short_EN.txt"
    output_csv = "/home/user/Documents/GitHub/CA-NMT_evaluation/Multi-encoder_k3_model/translation_outputs/output_VANILLA_german.csv"

    translate_and_convert_to_csv(input_file, output_csv)

if __name__ == "__main__":
    main()