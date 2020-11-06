"""Preprocess a file containing yelp review json data.

This script takes an input a file containing reviews from the yelp
dataset and outputs a file containing preprocessed data. The reviews
for which the script fails will be written to a seperate file named
"failed_text_file.txt" along with the error.

Run script with:

python3 preprocess_review_data.py <file-path-to-original-file> <output-filename>

"""

import sys
import json
import time
from langdetect import detect
import langdetect
from gensim.parsing.preprocessing import remove_stopwords

CHARACTERS_TO_REMOVE = [";", "%", "=", "&", ":", "|", "/", "\"", ".", ",", "!", "(", ")", "-", "+", "–", "_", "\n", "?", "*", "$", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "#"]


def process_text(text):
    """Process review text for machine learning algorithm.

    If the input is not in english or langdetect fails, an error
    message will be returned together with the original input text.

    Args:
        text (string): The text to be processed

    Returns tuple with elements:
        result (String): The processed text (or just the input text in case of error)
        error (None or String): Error message if there is an error
    """

    # Check language of text
    try:
        language = detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return text, "Language detection failed for review text"

    # Return an error if text is not in english
    if language != 'en':
        return text, "Review does not seem to be in english. Got language code \"{}\"".format(language)

    # Remove special characters
    for character_to_remove in CHARACTERS_TO_REMOVE:
        text = text.replace(character_to_remove, " ")

    # Remove multiple spaces after one another
    for i in range(10, 1, -1):
        text = text.replace(" " * i, " ")

    # Lovercase text
    text = text.lower()

    # Remove last character if it is a space
    if text[-1] == " ":
        text = text[:-1]

    # Remove stopwords
    text = remove_stopwords(text)

    # Return successfully converted text
    return text, None


def main():
    if len(sys.argv) < 3:
        print("You have to specify a filename for the original file and a filename for the output file")
        print("You should run script as:")
        print("python3 preprocess_review_data.py <file-path-to-file> <output-filename>")
        sys.exit()

    filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    # Start timer (for measuring total running time)
    start_time = time.time()

    number_of_lines_to_read = 1000000000
    output_file = open(output_filepath, "w")
    failed_text_file = open("failed_text_file.txt", "w")
    output_file.write("training, label\n")

    cnt = 0
    with open(filepath) as fp:
        line = fp.readline()
        while line and cnt < number_of_lines_to_read:
            cnt += 1
            if (cnt % 10000 == 0):
                print("Finished processing first", cnt)

            # Get relevant json fields
            review_dict = json.loads(line)
            stars = review_dict["stars"]
            text = review_dict["text"]

            # Process text
            text, error = process_text(text)

            # Write to separate file if an error occured
            if (error):
                text_without_newlines = text.replace("\n", "")
                failed_text_file.write(text_without_newlines + " - error: " + error + "\n")

            # Write output to file
            output_file.write("\"" + text + "\", " + str(stars) + "\n")

            # Read next line
            line = fp.readline()

    output_file.close()
    failed_text_file.close()

    # Print total running time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed_time: " + str(round(elapsed_time, 2)) + " seconds")


if __name__ == "__main__":
    main()
