import numpy as np
import re
import time
import json
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

def movie_etl(wiki_data, kaggle_data, rating_data):
    start_time = time.time()
    
    # Step 1 - Extract: Load data from files
    try:
        with open(wiki_data, mode='r') as file:
            wiki_movies_raw = json.load(file)
        print(len(wiki_movies_raw))

        kaggle_metadata = pd.read_csv(kaggle_data, low_memory=False)
        print(len(kaggle_metadata))

        ratings = pd.read_csv(rating_data)
        print(len(ratings))
    except Exception as error:
        print(f"Error during loading files: {error}")
    
    
    # Step 2 - Transform
    # clean-up and transform data in movies_df and ratings
    
    
    # Create a DataFrame from the raw data.
    wiki_movies_df = pd.DataFrame(wiki_movies_raw)
    
    # Check if either “Director” or “Directed by” are keys in the current dict. If there is a director listed, we also want to check that the dict has an IMDb link
    wiki_movies=[movie for movie in wiki_movies_raw
             if ('Director' in movie or 'Directed by' in movie) 
                 and 'imdb_link' in movie]

    # Make a DataFrame from wiki_movies
    wiki_movies=pd.DataFrame(wiki_movies)
    
    # Eliminating TV shows ('No. of episodes'):
    wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]
    
    # Function to clean our movie data:
    def clean_movie(movie):
        movie = dict(movie) #create a non-destructive copy
        alt_titles = {}
        # combine alternate titles into one list
        for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                    'Hangul','Hebrew','Hepburn','Japanese','Literally',
                    'Mandarin','McCune-Reischauer','Original title','Polish',
                    'Revised Romanization','Romanized','Russian',
                    'Simplified','Traditional','Yiddish']:
            if key in movie:
                alt_titles[key] = movie[key]
                movie.pop(key)
        if len(alt_titles) > 0:
            movie['alt_titles'] = alt_titles

        # merge column names
        # merge column names
        def change_column_name(old_name, new_name):
            if old_name in movie:
                movie[new_name] = movie.pop(old_name)
                         
        change_column_name('Adaptation by', 'Writer(s)')
        change_column_name('Country of origin', 'Country')
        change_column_name('Directed by', 'Director')
        change_column_name('Distributed by', 'Distributor')
        change_column_name('Edited by', 'Editor(s)')
        change_column_name('Length', 'Running time')
        change_column_name('Original release', 'Release date')
        change_column_name('Music by', 'Composer(s)')
        change_column_name('Produced by', 'Producer(s)')
        change_column_name('Producer', 'Producer(s)')
        change_column_name('Productioncompanies ', 'Production company(s)')
        change_column_name('Productioncompany ', 'Production company(s)')
        change_column_name('Released', 'Release Date')
        change_column_name('Release Date', 'Release date')
        change_column_name('Screen story by', 'Writer(s)')
        change_column_name('Screenplay by', 'Writer(s)')
        change_column_name('Story by', 'Writer(s)')
        change_column_name('Theme music composer', 'Composer(s)')
        change_column_name('Written by', 'Writer(s)')

        return movie
    
    # List of cleaned movies with a list comprehension
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    
    # Set wiki_movies_df to be the DataFrame created from clean_movies, and print out a list of the columns.
    wiki_movies_df = pd.DataFrame(clean_movies)
    
    # Now we can rerun our list comprehension to clean wiki_movies and recreate wiki_movies_df.
    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    wiki_movies_df = pd.DataFrame(clean_movies)
    
    # The code to extract the IMDb ID
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
    
    # drop any duplicates of IMDb IDs by using the drop_duplicates() method.
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
    
    # Remove Mostly Null Columns
    # That will give us the columns that we want to keep, which we can select from our Pandas DataFrame as follows:
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
    
    #Convert and Parse the Data
    # The box office data, which should give us code that we can reuse and tweak for the budget data since they’re both currency.

    # First we’ll make a data series that drops missing values with the following:
    box_office = wiki_movies_df['Box office'].dropna() 

    def is_not_a_string(x):
        return type(x) != str

    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
    
    # Form one “$123.4 million” (or billion)
    form_one = r'\$\d+\.?\d*\s*[mb]illion'

    # Form two “$123,456,789.”
    form_two = r'\$\d{1,3}(?:,\d{3})+'
    
    # Compare Values in Forms
    matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

    # 1. Some values have spaces in between the dollar sign and the number.
    orm_one = r'\$\s*\d+\.?\d*\s*[mb]illion'
    form_two = r'\$\s*\d{1,3}(?:,\d{3})+'

    # 2. Some values use a period as a thousands separator, not a comma.
    form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'    
  
     # 3. Some values are given as a range.
    box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)   
    
    # 4. “Million” is sometimes misspelled as “millon.”
    form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'    
     
    # Extract and Convert the Box Office Values
    matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)

    # Extract and Convert the Box Office Values

    def parse_dollars(s):
        # if s is not a string, return NaN
        if type(s) != str:
            return np.nan

        # if input is of the form $###.# million
        if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " million"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a million
            value = float(s) * 10**6

            # return value
            return value

        # if input is of the form $###.# billion
        elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

            # remove dollar sign and " billion"
            s = re.sub('\$|\s|[a-zA-Z]','', s)

            # convert to float and multiply by a billion
            value = float(s) * 10**9

            # return value
            return value

        # if input is of the form $###,###,###
        elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

            # remove dollar sign and commas
            s = re.sub('\$|,','', s)

            # convert to float
            value = float(s)

            # return value
            return value

        # otherwise, return NaN
        else:
            return np.nan
        
    # we need to extract the values from box_office using str.extract. Then we'll apply parse_dollars to the first column in the DataFrame returned by str.extract,
    wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)   
    
    # We no longer need the Box Office column, so we’ll just drop it:
    wiki_movies_df.drop('Box office', axis=1, inplace=True)

    # Parse Budget Data

    # Create a budget variable
    budget = wiki_movies_df['Budget'].dropna()

    # Convert any lists to strings:
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)

    # Then remove any values between a dollar sign and a hyphen (for budgets given in ranges):
    budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

    # Patern matches
    matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
    matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)

    # Remove the citation references
    budget = budget.str.replace(r'\[\d+\]\s*', '')
    
    # Parse the budget values
    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
    
    # We can also drop the original Budget column
    wiki_movies_df.drop('Budget', axis=1, inplace=True)

    # Parse Release Date
    release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
 
    # parse those forms is with the following:
    date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
    date_form_two = r'\d{4}.[01]\d.[123]\d'
    date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
    date_form_four = r'\d{4}'

    # use the built-in to_datetime() method in Pandas. Since there are different date formats, set the infer_datetime_format option to True.
    wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)
    
     # Parse Running Time
    running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)   
    
    # We only want to extract digits, and we want to allow for both possible patterns. Therefore, we’ll add capture groups around the \d instances as well as add an alternating character. Our code will look like the following.
    running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    
    # this new DataFrame is all strings, we’ll need to convert them to numeric values. Because we may have captured empty strings, we’ll use the to_numeric() method and set the errors argument to 'coerce'. Coercing the errors will turn the empty strings into Not a Number (NaN), then we can use fillna() to change all the NaNs to zeros.
    running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

    # Now we can apply a function that will convert the hour capture groups and minute capture groups to minutes if the pure minutes capture group is zero, and save the output to wiki_movies_df:
    wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)

    # drop Running time from the dataset with the following code:
    wiki_movies_df.drop('Running time', axis=1, inplace=True)

    # Kaggle
    # The following code will keep rows where the adult column is False, and then drop the adult column.
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')

     # Code creates the Boolean column we want. We just need to assign it back to video:
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'   
    
     # Convert numeric columns:
    try:
        kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
        kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
        kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')   
    except Exception as error:
        print(f"Error during numeric conversion: {error}")
     
    # Convert to datetime:
    try:
        kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])   

        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s') 
    except Exception as error:
        print(f"Date time error: {error}")

    # Merge Wikipedia and Kaggle Metadata
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
    
     # Competing data:
    # Wiki                     Movielens                Resolution
    #--------------------------------------------------------------------------
    # title_wiki               title_kaggle             Drop Wikipedia. 
    # running_time             runtime                  Keep Kaggle; fill in zeros with Wikipedia data.
    # budget_wiki              budget_kaggle            Keep Kaggle; fill in zeros with Wikipedia data.
    # box_office               revenue                  Keep Kaggle; fill in zeros with Wikipedia data.
    # release_date_wiki        release_date_kaggle      Drop Wikipedia.
    # Language                 original_language        Drop Wikipedia.
    # Production company(s)    production_companies     Drop Wikipedia.  
    
    # Drop the title_wiki, release_date_wiki, Language, and Production company(s) columns.
    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)

    # Function that fills in missing data for a column pair and then drops the redundant column.
    def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
        df[kaggle_column] = df.apply(
            lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
            , axis=1)
        df.drop(columns=wiki_column, inplace=True)
    
    # Now we can run the function for the three column pairs that we decided to fill in zeros.
    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')

    # Check that there aren’t any columns with only one value, since that doesn’t really provide any information. Don’t forget, we need to convert lists to tuples for value_counts() to work.
    for col in movies_df.columns:
        lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
        value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
        num_values = len(value_counts)
        
    movies_df['video'].value_counts(dropna=False)

    # Reorder the columns:
    movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                           'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                           'genres','original_language','overview','spoken_languages','Country',
                           'production_companies','production_countries','Distributor',
                           'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                          ]]

    # Rename the columns to be consistent.
    movies_df.rename({'id':'kaggle_id',
                      'title_kaggle':'title',
                      'url':'wikipedia_url',
                      'budget_kaggle':'budget',
                      'release_date_kaggle':'release_date',
                      'Country':'country',
                      'Distributor':'distributor',
                      'Producer(s)':'producers',
                      'Director':'director',
                      'Starring':'starring',
                      'Cinematography':'cinematography',
                      'Editor(s)':'editors',
                      'Writer(s)':'writers',
                      'Composer(s)':'composers',
                      'Based on':'based_on'
                     }, axis='columns', inplace=True)
    
    # Transform and Merge Rating Data
    # use a groupby on the “movieId” and “rating” columns and take the count for each group.
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                    .rename({'userId':'count'}, axis=1) \
                    .pivot(index='movieId',columns='rating', values='count')

    # Rename the columns so they’re easier to understand.
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

    # Merge ratings with movie_df
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')

    # Fill missing values with NaN for missing ratings:
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

    # Step 3 - Load
    # Store results in database
    
    try:
        # pasword is stored in a config file
        from config import db_password
        
        # For our local server, the connection string will be as follows:
        db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"
        
        # Create database engine:
        engine = create_engine(db_string)
    except Exception as error:
        print(f"Error while connecting to database: {error}")
        return
    
    try:
        movies_df.to_sql(name='movies', con=engine, if_exists="replace")
        ratings.to_sql(name='ratings', con=engine, if_exists="replace", chunksize=100_000)
    except Exception as error:
        print(f"Error while saving data to database: {error}")
        return
    
    # print that the rows have finished importing
    print(f'Done. {time.time() - start_time} total seconds elapsed')
