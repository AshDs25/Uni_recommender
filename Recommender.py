import nltk
import numpy as np
import math
import pickle
import json
from flask import Flask,request, jsonify
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from fetchData import fetch
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

with open('preprocessed_data.pkl', 'rb') as f:
    pre_course, pre_stream, pre_country, pre_test = pickle.load(f)

with open('word2vec_model.pkl','rb') as f:
   model = pickle.load(f)

with open('num_data.pkl', 'rb') as f:
    test_score, duration, fees = pickle.load(f)

with open('university_names.pkl', 'rb') as f:
    University = pickle.load(f)

def reshape_vector(vector):
  shape = vector.shape
  reshaped_vector = vector.reshape(shape[1],100)
  return reshaped_vector

#Similarity lists
sim = []
euc_sim = []

def text_sim(course_vector,stream_vector,country_vector,test_vector):
    course_sim = []
    stream_sim = []
    country_sim = []
    test_sim = []
    for i in range(len(pre_course)):
    # if not pre_course[i]:
    #   ignore.append(i)
    #   continue
        course = pre_course[i]
        stream = pre_stream[i]
        country_label = pre_country[i]
        test_label = pre_test[i]
        course_data = model.wv[course]
        stream_data = model.wv[stream]
        country_data = model.wv[country_label]
        test_data = model.wv[test_label]
        similarity = cosine_similarity(course_vector, course_data)[0][0]
        course_sim.append(similarity)
        similarity = cosine_similarity(stream_vector, stream_data)[0][0]
        stream_sim.append(similarity)
        similarity = cosine_similarity(country_vector.reshape(1,-1), country_data)[0][0]
        country_sim.append(similarity)
        similarity = cosine_similarity(test_vector, test_data)[0][0]
        test_sim.append(similarity)

    
    for i in range(len(course_sim)):
        sum_val = 0.4*course_sim[i] + 0.15*country_sim[i]+ 0.4*stream_sim[i]+0.05*test_sim[i]
        sim.append(sum_val)


def euclidean(x,y):
    # Calculate squared differences of corresponding elements and sum them up
    squared_diff_sum = np.sum((x - y) ** 2)
    # Take the square root of the sum of squared differences
    distance = math.sqrt(squared_diff_sum)
    # Similarity is inversely proportional to the distance
    # Return the similarity, ensuring it's between 0 and 1
    return 1 / (1 + distance)


def euclidean_sim(user_score, user_dur,user_fee):
   for i in range(len(test_score)):
    sim = 0.2*euclidean(test_score[i],user_score) + 0.4*euclidean(duration[i],user_dur) +0.4 * euclidean(fees[i],user_fee)
    euc_sim.append(sim)


@app.route('/top_10_universities',methods=['POST'])
def get_top_10_universities():
    
    user_data = request.get_json()
    user_score = user_data.get('score')
    user_dur = user_data.get('duration')
    user_fee = user_data.get('fee')
    user_course = user_data.get('course')
    user_stream = user_data.get('stream')
    user_country = user_data.get('country')
    user_test = user_data.get('test')

    processed_course = word_tokenize(user_course)
    processed_stream = word_tokenize(user_stream)
    processed_country = word_tokenize(user_country)
    processed_test = word_tokenize(user_test)
    

    # Get vector for user input
    course_vector = np.array([model.wv[processed_course]])
    course_vector = reshape_vector(course_vector)

    stream_vector = np.array([model.wv[processed_stream]])
    stream_vector = reshape_vector(stream_vector)

    country_vector = np.array([model.wv[processed_country]])

    test_vector = np.array([model.wv[processed_test]])
    test_vector = reshape_vector(test_vector)

    text_sim(course_vector,stream_vector,country_vector,test_vector)
    euclidean_sim(user_score, user_dur,user_fee)

    
    top_10 = []
    added_universities = set()
    combined_sim =[]
    for i in range(len(euc_sim)):
        combined_sim.append(0.8*sim[i] + 0.2*euc_sim[i])
    
    zipped_list = list(zip(University, combined_sim))
    sorted_zipped_list = sorted(zipped_list, key=lambda x: x[1], reverse=True)
   


    for university, similarity in sorted_zipped_list:
        if university not in added_universities:
            top_10.append((university, similarity))
            added_universities.add(university)
            if len(top_10) == 10:
                break

    # Return the JSON data
    return jsonify(fetch(added_universities))


if __name__ == '__main__':
    app.run(host="0.0.0.0")




    
