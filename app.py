import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import bs4 as bs
import urllib.request

# Load the NLP model and TF-IDF vectorizer
try:
    with open('nlp_model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
    with open('tranform.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    print(f"Error loading model files: {e}")


# Load and preprocess movie data
def load_movie_data():
    try:
        data = pd.read_csv('Output CSV/main_data.csv')
        return data
    except Exception as e:
        print(f"Error loading movie data: {e}")
        return None


# Create a similarity matrix
def create_similarity():
    data = load_movie_data()
    if data is None:
        return None, None

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)
    return data, similarity


# Movie recommendation function
def recommend_movie(movie_name):
    movie_name = movie_name.lower()
    try:
        data, similarity = create_similarity()
        if data is None or similarity is None:
            return "Error loading data"

        if movie_name not in data['movie_title'].str.lower().values:
            return "Sorry! Try another movie name"

        index = data.loc[data['movie_title'].str.lower() == movie_name].index[0]
        similarity_scores = list(enumerate(similarity[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]

        recommended_movies = [data['movie_title'][i[0]] for i in similarity_scores]
        return recommended_movies
    except Exception as e:
        return f"Error processing request: {e}"


# Convert JSON-like string lists to Python lists
def convert_to_list(my_list):
    my_list = my_list.strip('[]"').split('","')
    return my_list


# Get movie suggestions
def get_suggestions():
    data = load_movie_data()
    if data is not None:
        return list(data['movie_title'].str.capitalize())
    return []


# Flask app setup
app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', suggestions=get_suggestions())


@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form.get('name', '').strip()
    recommendations = recommend_movie(movie)
    if isinstance(recommendations, str):
        return recommendations
    return "---".join(recommendations)


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        # Fetch movie details from request
        details = {key: request.form[key] for key in request.form}

        # Convert relevant fields to lists
        details['rec_movies'] = convert_to_list(details['rec_movies'])
        details['rec_posters'] = convert_to_list(details['rec_posters'])
        details['cast_names'] = convert_to_list(details['cast_names'])
        details['cast_chars'] = convert_to_list(details['cast_chars'])
        details['cast_profiles'] = convert_to_list(details['cast_profiles'])
        details['cast_bdays'] = convert_to_list(details['cast_bdays'])
        details['cast_bios'] = convert_to_list(details['cast_bios'])
        details['cast_places'] = convert_to_list(details['cast_places'])

        # Convert cast IDs to a list
        details['cast_ids'] = details['cast_ids'].strip("[]").split(',')

        # Clean newline characters in bios
        details['cast_bios'] = [bio.replace(r'\n', '\n').replace(r'\"', '\"') for bio in details['cast_bios']]

        # Create dictionaries for easy template rendering
        movie_cards = dict(zip(details['rec_posters'], details['rec_movies']))
        casts = {
            details['cast_names'][i]: [details['cast_ids'][i], details['cast_chars'][i], details['cast_profiles'][i]]
            for i in range(len(details['cast_profiles']))}
        cast_details = {details['cast_names'][i]: [details['cast_ids'][i], details['cast_profiles'][i],
                                                   details['cast_bdays'][i], details['cast_places'][i],
                                                   details['cast_bios'][i]]
                        for i in range(len(details['cast_places']))}

        # Scrape IMDb reviews
        reviews, review_sentiments = scrape_imdb_reviews(details['imdb_id'])

        movie_reviews = dict(zip(reviews, review_sentiments))

        # Render template with data
        return render_template('recommend.html', **details, movie_cards=movie_cards, reviews=movie_reviews,
                               casts=casts, cast_details=cast_details)

    except Exception as e:
        return jsonify({"error": str(e)})


def scrape_imdb_reviews(imdb_id):
    """Scrapes reviews from IMDb and classifies them as Good or Bad."""
    reviews, review_sentiments = [], []
    try:
        url = f'https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt'
        page = urllib.request.urlopen(url).read()
        soup = bs.BeautifulSoup(page, 'lxml')
        review_divs = soup.find_all("div", {"class": "text show-more__control"})

        for review in review_divs:
            if review.string:
                reviews.append(review.string)
                review_vector = vectorizer.transform(np.array([review.string]))
                prediction = clf.predict(review_vector)
                review_sentiments.append('Good' if prediction else 'Bad')
    except Exception as e:
        print(f"Error scraping IMDb reviews: {e}")

    return reviews, review_sentiments


if __name__ == '__main__':
    app.run(debug=True)
