import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_imdb_top_movies(num_movies=50):
    """
    Scrape the top IMDb movies and extract title and user review URLs.
    """
    base_url = "https://www.imdb.com/chart/top"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    movie_links = []
    for movie in soup.find_all('td', class_='titleColumn')[:num_movies]:
        movie_title = movie.find('a').text
        movie_url = "https://www.imdb.com" + movie.find('a')['href']
        movie_links.append((movie_title, movie_url))

    return movie_links

def scrape_movie_reviews(movie_url):
    """
    Scrape user reviews for a given movie URL.
    """
    response = requests.get(movie_url + 'reviews')
    soup = BeautifulSoup(response.text, 'html.parser')

    reviews = []
    for review in soup.find_all('div', class_='text show-more__control'):
        reviews.append(review.text.strip())

    return reviews

def generate_dataset(num_movies=50):
    """
    Generate dataset by scraping IMDb top movies and their reviews.
    """
    movie_links = scrape_imdb_top_movies(num_movies)
    data = []
    for title, url in movie_links:
        reviews = scrape_movie_reviews(url)
        for review in reviews:
            data.append({'Title': title, 'Review': review})

    df = pd.DataFrame(data)
    df.to_csv('dataset.csv', index=False)

if __name__ == "__main__":
    generate_dataset()
