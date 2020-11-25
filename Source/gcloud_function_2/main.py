import requests
try:
    import cPickle as pickle
except:
    import pickle


def load_from_dump(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def process_message(message):
    response = requests.get(
        "https://storage.googleapis.com/tdt4173-functions-data-bucket/kNNClassifier.bin")
    kNNClassifier = pickle.loads(response.content)
    vectorizer = load_from_dump("TFIDFvectorizer.bin")

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
    if (request.args and 'message' in request.args):
        message = request.args.get('message')
        return process_message(message)
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
