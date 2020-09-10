"""
Model server script that polls Redis for ptitle classification

Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""
import re
import base64
import json
import pickle
import time
import redis
import nltk
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize, sent_tokenize
import numpy as np
import keras.backend as K
from keras.utils import to_categorical 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Flatten
from tensorflow.python.keras.preprocessing.text import tokenizer_from_json
from twitterclient import*
# Prediction
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
import os
import sys
# sys.path.append(os.path.abspath('/home/saqueeb/titleclassifyer/modelserver'))
from nltk.stem.porter import PorterStemmer
use_stemmer = False

#-----get the env variable----
try: IMAGE_QUEUE= os.environ['IMAGE_QUEUE']
except: IMAGE_QUEUE ="image_queue"
try:  BATCH_SIZE= int(os.environ['BATCH_SIZE'])
except: BATCH_SIZE=32
try:  IMAGE_DTYPE= os.environ['IMAGE_DTYPE']
except: IMAGE_DTYPE="float32"
try:  IMAGE_HEIGHT= int(os.environ['IMAGE_HEIGHT'])
except: IMAGE_HEIGHT=224
try:  IMAGE_WIDTH= int(os.environ['IMAGE_WIDTH'])
except: IMAGE_WIDTH=224
try:  IMAGE_CHANS= int(os.environ['IMAGE_CHANS'])
except: IMAGE_CHANS=3
try:  SERVER_SLEEP= float(os.environ['SERVER_SLEEP'])
except: SERVER_SLEEP=0.25
try:  REDIS_HOST= os.environ['REDIS_HOST']
except: REDIS_HOST="redis"
#-- model related----
try:  MODEL_PATH= os.environ['MODEL_PATH']
except: MODEL_PATH="/app/"
try:  MODEL_FILE= os.environ['MODEL_FILE']
except:
    MODEL_FILE = "Tell_Sentiment.h5"
try:  TOKENIZER_FILE= os.environ['TOKENIZER_FILE']
except:
    TOKENIZER_FILE = "tokenizer_new.json"

# global LABEL_DICT
# try:
#     label_list = [line.rstrip().strip() for line in open(LABEL_FILE)]
#     LABEL_DICT = {i : label_list[i] for i in range(len(label_list))}
# except:
#     print("ERROR: {} not found or could not be oppened".format(LABEL_FILE))
def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = porter_stemmer.stem(word)
            processed_tweet.append(word)

    return ' '.join(processed_tweet)

def preprocess_tweets(tweetlist):
    tweets =[]
    for n in range(len(tweetlist)):
        line = str(tweetlist[n])
        tweet_id = line[:line.find(',')]
        line = line[1 + line.find(','):]
        tweet = line
        processed_tweet = preprocess_tweet(tweet)
        tweets.append('%s' %(processed_tweet))
    return tweets

def search_tweets(search_item):
    # creating object of TwitterClient Class 
    api = TwitterClient() 
    # calling function to get tweets 
    tweets = api.get_tweets(query = search_item, count = 200) 
    tweets = preprocess_tweets(tweets) # preprocess to remove @ # and user name etc and return
    return  tweets

def preprocess_csv(csv_file_name, processed_file_name, test_file=False):
    save_to_file = open(processed_file_name, 'w')

    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id = line[:line.find(',')]
            if not test_file:
                line = line[1 + line.find(','):]
                positive = int(line[:line.find(',')])
            line = line[1 + line.find(','):]
            tweet = line
            processed_tweet = preprocess_tweet(tweet)
            if not test_file:
                save_to_file.write('%s,%d,%s\n' %
                                   (tweet_id, positive, processed_tweet))
            else:
                save_to_file.write('%s,%s\n' %
                                   (tweet_id, processed_tweet))
            write_status(i + 1, total)
    save_to_file.close()
    print ('\nSaved processed tweets to: %s' % processed_file_name)
    return processed_file_name

db = redis.StrictRedis(host=REDIS_HOST)

# Load the pre-trained Keras model 
try:
    model = load_model(MODEL_PATH + MODEL_FILE)
except:
    print("ERROR: Unable to lead the model file {}. Make sure its .h5 formated".format(MODEL_PATH+MODEL_FILE))#,flush=True)
    sys.stdout.flush()
# Tokenize texts
try:
    with open(MODEL_PATH+TOKENIZER_FILE) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
    # with open(MODEL_PATH+TOKENIZER_FILE) as handle:
    #     tokenizer = pickle.load(handle)

except:
    print("ERROR: Unable to lead the tokenizer form {}".format(MODEL_PATH+TOKENIZER_FILE))#,flush=True)
    sys.stdout.flush()

# def sent_text(p):
#     k= int(p)
#     if k == 1:
#         return("POSITIVE")
#     elif k == 0:
#         return "NEGATIVE"
#     else:
#         return"NEGATIVE"

    # if positive_count> negative_count:
    #     print("Recent Tweets regarding the keyword you have entered are POSITIVE")
    # elif negative_count> positive_count:
    #     return "Recent Tweets regarding the keyword name you have entered are NEGATIVE"
    # else:
    #     return"Recent Tweets regarding the keyword name you have entered are NEGATIVE"

def classify_tweets():
    while True:
        # Pop off multiple items from Redis queue atomically
        with db.pipeline() as pipe:
            pipe.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
            pipe.ltrim(IMAGE_QUEUE, BATCH_SIZE, -1)
            queue, _ = pipe.execute()
        
        
        imageIDs = []
        reviews = []
        tweets = []
        for q in queue:
            # Deserialize the object and obtain the input image
            q = json.loads(q)  # .decode("utf-8"))
            # Update the list of titles and IDs
            imageIDs.append(q["id"])
            reviews.append(q['title'].lower().replace('\\n', ' '))
            
        if len(imageIDs)>0:
            for t in reviews:
           # t = next(s for s in reviews if s)
                tweets= search_tweets(t)
                tweets = np.array(tweets)
                tweets = preprocess_tweets(tweets)

        # with open('/app/tokenizer.pickle', 'rb') as handle:
        #     tokenizer = pickle.load(handle) 

#Taking the tweets to the data array and preprocess it accordingly
                data = []
                for line in tweets:
                    data.append(line)

#pad sequence to the length we made the model
                sequences = tokenizer.texts_to_sequences(data) 
                MAX_SEQUENCE_LENGTH = 100
                s = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#Lets feed the input to our model
        # my_model = load_model('/app/Twitter_Sentiment.h5')
                x = model.predict_on_batch(s)
                x = np.round(x) #rounding up the output
            # print(x)

#Generates end result: 
                p_count = np.count_nonzero( x == 1)
                n_count = np.count_nonzero( x == 0)
                preds = []
                def sentim(positive_count, negative_count):
                    if positive_count> negative_count:
                        return "Recent Tweets regarding the keyword you have entered are POSITIVE"
                    elif negative_count> positive_count:
                        return "Recent Tweets regarding the keyword name you have entered are NEGATIVE"
                    else:
                        return "Recent Tweets regarding the keyword name you have entered are NEGATIVE"
                
                p = sentim(p_count, n_count)
                preds.append(p)
            # print("**************Prediction on ",  t, " says ", sentim(p_count, n_count))
            # print(sentim(p_count, n_count))
            # sys.stdout.flush()
            # r = {"imageid": str(t), "Sentiment": str(sentim(p_count, n_count))}
            # db.set(str(t), json.dumps(r))
            for (imageID, title, p) in zip(imageIDs, reviews, preds):
                print("**************Prediction for id:",  imageID, "and Sentiment is: " , p)
                sys.stdout.flush()
                r = {"id":imageID, "Sentiment": str(p), "Review": str(title)  }
                db.set(imageID, json.dumps(r))
        # Sleep for a small amount
        time.sleep(SERVER_SLEEP)



# def classify_process():
#     # Continually poll for new images to classify
#     while True:
#         # Pop off multiple items from Redis queue atomically
#         with db.pipeline() as pipe:
#             pipe.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
#             pipe.ltrim(IMAGE_QUEUE, BATCH_SIZE, -1)
#             queue, _ = pipe.execute()

#         imageIDs = []
#         reviews =[]
#         for q in queue:
#             # Deserialize the object and obtain the input image
#             q = json.loads(q) #.decode("utf-8"))
           
#             # Update the list of titles and IDs
#             imageIDs.append(q["id"])
#             reviews.append(q['title'].lower().replace('\\n', ' '))#.strip('""'))

#         # Check to see if we need to process the batch
#         if len(imageIDs) > 0:
#             # Classify the batch
#             print("* Batch size: {}".format(len(imageIDs)))
#             # Tokenize texts
#             sequences = tokenizer.texts_to_sequences(reviews)
#             MAX_SEQUENCE_LENGTH = 100
#             data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) 
#             # sequences = tokenizer.texts_to_sequences(reviews)
#             # data = pad_sequences(sequences, maxlen=100)
#             print('Processing texts shape {} from db'.format(data.shape))
#             #-----predict------------------
#             preds = model.predict(data)
#             preds = np.round(preds)
#             preds = np.array(preds)
#             for (imageID, title, p) in zip(imageIDs, reviews, preds):
                
#                 print("**************Prediction for id:",  imageID, "and Sentiment is: " , p)
#                 sys.stdout.flush()
#                 r = {"id":imageID, "Sentiment": str(p), "confidence":str(np.max(p)), "review":title  }
#                 db.set(imageID, json.dumps(r))

#         # Sleep for a small amount
#         time.sleep(SERVER_SLEEP)

if __name__ == "__main__":
    classify_tweets()
