from flask import Flask, request, render_template
import re, string, pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')  # Add if you use tokenization too



# Load model and vectorizer
RF = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {"not", "no", "bad", "good"}

def cleaning_text(text):
    text = text.lower()
    text = re.sub(r"@user", "", text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/Predict', methods=['POST', 'GET'])  # <-- fixed here
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        cleaned_comment = cleaning_text(comment)
        comment_vector = tfidf.transform([cleaned_comment])
        prediction = RF.predict(comment_vector)[0]
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
