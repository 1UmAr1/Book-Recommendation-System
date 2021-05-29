import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
Data = pd.read_csv("Books_title.csv")
books = pd.read_csv("books.csv")

title_matrix = Data
title_matrix = title_matrix.drop("book_id", axis=1)

indices = pd.Series(books.index, index=books['title'])
titles = books['title']


def book_recommendations(title):
    n = 10
    idx = indices[title]
    cosine_sim = cosine_similarity(title_matrix, title_matrix)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    book_indices = [i[0] for i in sim_scores]
    return titles.iloc[book_indices]


book_index = 10
book_recommendations(books.title[book_index])
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index2.html")


@app.route('/predict', methods=['POST', "GET"])
def predict():
    if request.method == "POST":
        message = request.form['message']
        output = book_recommendations(title=message)
        output = pd.DataFrame(output)
        print(output)

        return render_template("output.html", prediction=output)


if __name__ == "__main__":
    app.run(debug=True)