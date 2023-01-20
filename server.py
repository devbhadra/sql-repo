from flask import Flask,request,jsonify,make_response
from flask_cors import CORS
import librosa
import soundfile
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from playsound import playsound 
from pydub import AudioSegment
import wave,random,json
from sklearn.feature_extraction.text import TfidfVectorizer
emotions={
  '01':'neutral' ,
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions=['calm', 'happy', 'neutral', 'sad']
Pkl_Speech_file = "Emotion_Voice_Detection_Model.pkl"

genres_mood = [["Thriller" , "Action" , "Drama" , "Horror" , "Sci-Fi"],["Drama" , "Comedy" , "Romance" , "Fantasy" ,"Adventure", "Family"],[ "Sci-Fi", "Drama", "Romance"]]

def get_genre_movie(genre_mood):
    movie_genre = pd.read_csv('tmdb_5000_movies.csv')
    genre = movie_genre['genres']
    titles = movie_genre['title']
    index = 0
    c= list(zip(genre,titles))
    random.shuffle(c)
    genre,titles = zip(*c)
    found = False
   
    for x in genre:
        obj = json.loads(x)
    for key in obj:
        print(key["name"])
        if key["name"] == genre_mood:
            found = True
            break
        if found:
        	break
        index+=1
    movie = titles[index]
    print(movie) 
    return movie


def give_rec(title):
	# Saving the vectorizer 
    vectorizer_file = "vectorizer.pkl"
    with open(vectorizer_file,'rb') as file:
    	tfv = pickle.load(file)
    
    movies_cleaned_pkl = "movies_cleaned.pkl"
    with open(movies_cleaned_pkl,'rb') as file:
    	movies_cleaned_df = pickle.load(file)

    sigmoid_file = "sigmoid.pkl"
    with open(sigmoid_file,'rb') as file:
    	sig = pickle.load(file)
    
    indices_file = 'indices.pkl'
    indices = pd.read_pickle(indices_file)

    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 4 most similar movies
    sig_scores = sig_scores[1:5]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_cleaned_df['original_title'].iloc[movie_indices]


def clean(y,rate,threshold):
    mask=[]
    y = pd.Series(y).apply(np.abs)#we use rolling window analysis for time-series data 
    y_mean= y.rolling(window = int(rate / 10),min_periods = 1, center = True).mean()
    #each  window is size of frq./ 10 and we check if mean > threshold, we deem it true, else its noise we deem it false.
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32") #convert the sound file to float32 type for our timeseries data
        sample_rate=sound_file.samplerate ##process at the same samplerate as that of param provided audio
        if chroma: #extracting short time fourier transform(stft): time freq values over windows. expected op a matrix.
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            ##y expects the time-series data of audio , sr is sample rate nd n_mfcc are the number of mfcc to return
            result=np.hstack((result, mfccs)) #horizontal columnar store.
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0) #expects abs(TimeSeriesdata)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


def load_speech_model(): ##loading pickle file
	#Loading the same model back from file
    with open(Pkl_Speech_file,'rb') as file:
        model = pickle.load(file)
    return model

def get_genreList(mood):
    
    if mood == 'happy':
        return 0
    elif mood == 'sad':
        return 1
    return 2

app = Flask(__name__)
cors = CORS(app)
@app.route('/api',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        f = open('./file.wav','wb')
        f.write(request.data)
        f.close()
        mood = []
        newA="test1.wav"
        sound = AudioSegment.from_file('./file.wav')
        sound = sound.set_channels(1)
        sound.export(newA, format="wav")
        new_feature = extract_feature(newA,mfcc=True,chroma=True,mel=True)
       
        mood.append(new_feature)
        mood = np.array(mood)
        model = load_speech_model()
        value = model.predict(mood)
        #mood in value. search a movie of same mood.
        ##TODO:map mood to movies.
        genre = genres_mood[get_genreList(value)]
        movie_set=set()
        output = []
        for x in genre:
            print("this is x ",x)
            mov = get_genre_movie(x)
            print("for x"+x+" we get movie "+mov)
            recom_movies = give_rec(mov)
            recom_set = set(recom_movies)
            print(recom_set)
            movie_set = movie_set.union(recom_set)
        print("movie set" , movie_set)    
        movie_list = list(movie_set)
        for x in movie_list:
            output.append(str(x))
        res = make_response(jsonify({'result':output}),200)
  
    return res
### 

if __name__ == '__main__':
    app.run(debug=False)