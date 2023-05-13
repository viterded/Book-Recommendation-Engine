from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the book data from a CSV file
books = pd.read_csv('books.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Retrieve the user's reading history and preferences from the form
    history = request.form['history']
    preferences = request.form['preferences']
    
    # Process the user's reading history and preferences
    history_books = books[books['title'].isin(history)]
    preferences_books = books[books['genre'].isin(preferences)]
    user_profile = pd.concat([history_books, preferences_books]).mean(axis=0)
    
    # Generate book recommendations using cosine similarity
    item_similarities = cosine_similarity(books.iloc[:, 2:], user_profile[2:].values.reshape(1, -1)).flatten()
    recommended_books_indices = np.argsort(item_similarities)[-10:]
    recommended_books = books.iloc[recommended_books_indices]
    
    # Return the recommended books to the user
    return render_template('recommendations.html', books=recommended_books)

if __name__ == '__main__':
    app.run(debug=True)

