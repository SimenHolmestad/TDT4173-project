from numpy.lib.function_base import vectorize
#from langdetect import detect
#import langdetect
#from gensim.parsing.preprocessing import remove_stopwords
#import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.sequence import pad_sequences
try:
    import cPickle as pickle
except:
    import pickle

CHARACTERS_TO_REMOVE = [";", "%", "=", "&", ":", "|", "/", "\"", ".", ",", "!",
                        "(", ")", "-", "+", "–", "_", "\n", "?", "*", "$", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "#"]


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

    if text == "":
        return text, "Review was empty after removing stop words"

    # Find number of words after stop words are removed
    number_of_words = len(text.split(" "))

    # Return error if number of words after removing stop words is more than 100.
    if number_of_words > 100:
        return text, "Review contained more than 100 words after removing stop words. Number of words where {}".format(str(number_of_words))

    # Return successfully converted text
    return text, None


def do_prediction(text):
    f = open("balanced_model_2epochs_tokenizer.json", "r")
    json_string = f.read()
    f.close()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    X = tokenizer.texts_to_sequences([text])
    X = pad_sequences(X, maxlen=100)

    model = load_model("balanced_model_2epochs.h5", compile=True)
    return list(model.predict(X)[0])


def process_LSTM_message(message):
    processed_text, error_message = process_text(message)
    if (error_message):
        return "Error: " + error_message

    prediction = do_prediction(processed_text)
    final_prediction = prediction.index(max(prediction)) + 1
    return_text = "Final prediction is: " + str(final_prediction) + " star"

    if final_prediction != 1:
        return_text += "s"

    return_text += "|Processed text was: " + processed_text
    prediction = list(map(lambda x: round(x, 3), prediction))
    return_text += "|Raw output for prediction was: " + str(prediction)
    return return_text


def load_from_dump(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def process_kNN_message(message):
    vectorizer = load_from_dump("TFIDFvectorizer.bin")
    kNNClassifier = load_from_dump("kNNClassifier.bin")

    transformed_message = vectorizer.transform([message])
    final_prediction = kNNClassifier.predict(transformed_message)[0] + 1
    return_text = "Final prediction is: " + str(final_prediction) + " star"

    if final_prediction != 1:
        return_text += "s"
    return return_text


def handle_request(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        #flask.Flask.make_response>`.
        `make_response <http://flask.pocoo.org/docs/1.0/api/
    """
    if (request.args and 'message' in request.args and "model" in request.args):
        message = request.args.get('message')
        if request.args.get("model") == "LSTM":
            return process_LSTM_message(message)
        elif request.args.get("model") == "kNN":
            return process_kNN_message(message)
    else:
        return 'Please add a "message" argument to the request.'


def main(request):
    # Found at https://cloud.google.com/functions/docs/writing/http#functions_http_cors-python
    # For more information about CORS and CORS preflight requests, see
    # https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request
    # for more information.

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    return (handle_request(request), 200, headers)


# Uncomment code below for debugging
if __name__ == '__main__':
    print(process_kNN_message(
        "This was terrible, just really bad. Not a good experience"))
