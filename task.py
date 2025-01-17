# %%
import pandas as pd
import numpy as np
import datetime
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# %% STEP 1: PREPROCESSING 
# Clean text by converting into lower case, removing non-alpha characters and filtering out non-significant words
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def clean_text(df, col):
    def convert_text(sentences):
        cleaned_texts = []
        for text in sentences:
            text = text.lower()
            text = re.sub(r'\W', ' ', text)       
            text = re.sub(r'\s+', ' ', text)  
            tokens = text.split()
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] 
            cleaned_texts.append(' '.join(tokens))
        return cleaned_texts
    
    df[f"clean_{col}"] = convert_text(df[col].fillna("").tolist())
    return df

# Add new features to train the model as counting how many letters in each abstract and article
def features_engineer(df):
    df['title_length'] = df['title'].apply(len) 
    df['abstract_length'] = df['abstract'].apply(len)  
    
    # Transform the length to log to be less skewed
    df['log_title_length'] = np.log1p(df['title_length'])  
    df['log_abstract_length'] = np.log1p(df['abstract_length'])  
    
    return df
    
# Transform the features into log to reduce skewness 
def transform_to_log(df, col_name):
    if np.issubdtype(df[col_name].dtype, np.number):
        df[f"{col_name}_log"] = np.log1p(df[col_name])
    else:  
        # Handle non-numeric columns (strings or lists)
        df[f"{col_name}_length"] = df[col_name].apply(lambda x: len(x.split()) if isinstance(x, str) else (len(x) if isinstance(x, list) else 0))
        df[f"{col_name}_log"] = np.log1p(df[f"{col_name}_length"])
    return df

# Measure the number of authors per article and also transform the count number into log
def number_of_authors(df):
    df["number_of_authors"] = df["authors"].apply(lambda x: len(x.split(',')))
    df["number_of_authors_log"] = np.log1p(df["number_of_authors"])  
    return df


# %% STEP 2: LOAD DATA AND APPY FUNCTIONS
train = pd.DataFrame.from_records(json.load(open('train.json'))) 
test = pd.DataFrame.from_records(json.load(open('test.json')))

# Check and fill the missing values 
train.isnull()
test.isnull()
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# %% Calculate the time publication of the articles because the articles published long time ago might get more citations compared to recently-published articles
train['year'] = train['year'].astype(int) 
test['year'] = test['year'].astype(int)

current_year = datetime.datetime.now().year
train['time_publish'] = current_year - train['year']
test['time_publish'] = current_year - test['year']

# Convert author, year, and references from train and test set into numeric features 
train = transform_to_log(train,'time_publish')
test = transform_to_log(test, 'time_publish')

train = number_of_authors(train)
test = number_of_authors(test)
    
train = transform_to_log(train, 'references')
test = transform_to_log(test, 'references')
      
# %% Adding features as the abstract and title length
train = features_engineer(train)
test = features_engineer(test)

# %%  Clean the abstracts, titles and venues in train and test sets and convert venue into categorical feature
train = clean_text(train, 'title')
train = clean_text(train, 'abstract')
train = clean_text(train, 'venue')

test = clean_text(test, "title")
test = clean_text(test, "abstract")
test = clean_text(test, "venue")


# %% Drop the unnecessary columns in both train and test sets to make it consistent
col = ['abstract', 'title', 'authors', 'references', 'venue', 'id', 'number_of_authors', 'references_length', 
       'year', 'time_publish', 'title_length', 'abstract_length']
train = train.drop(columns = [c for c in col if c in train.columns], axis = 1)
test= test.drop(columns = [c for c in col if c in test.columns], axis=1)

# %%  STEP 3: EXTRACT A SAMPLE TO TRAIN BY USING THE QUARTITLES FROM THE ORIGINAL TRAIN DATASET 
np.log1p(train['n_citation']).describe()  #check the target distribution for the sample extraction
bins = [0, 3, 22, 50, 25835]  
labels = ['Q1', 'Q2', 'Q3', 'Q4']
train['citation_bin'] = pd.cut(train['n_citation'], bins=bins, labels=labels, include_lowest=True)

sample = train.groupby('citation_bin', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42)) 

# %%  STEP 4: SPLIT AND TRAIN MODEL
X_train, X_validation = train_test_split(sample, test_size=0.2, random_state=123)

# %%  Filter out the words and standardize the numerical features
featurizer = ColumnTransformer(
    transformers=[
        ('time_publish_log', StandardScaler(), ['time_publish_log']),
        ("number_of_authors_log", StandardScaler(), ["number_of_authors_log"]),
        ('clean_venue', TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=500), 'clean_venue'),
        ('clean_abstract', TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0.01, max_df=0.85), 'clean_abstract'),
        ('clean_title', TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0.0005, max_df= 0.8), 'clean_title'),
        ('references_log', StandardScaler(), ['references_log']), 
        ('log_title_length', StandardScaler(), ['log_title_length']),
        ('log_abstract_length', StandardScaler(), ['log_abstract_length']),], remainder='drop')

# %% Make the pipeline for the models
dummy = Pipeline([('featurizer', featurizer), ("dummy", DummyRegressor())])  
elasticnet = Pipeline([('featurizer', featurizer), ('elasticnet', ElasticNet(alpha=0.0001, l1_ratio=0.05))])
gb = Pipeline([('featurizer', featurizer), ('gb', GradientBoostingRegressor(n_estimators=1500, max_depth=12, 
                                                                            learning_rate=0.01,loss='squared_error', random_state=123))])
label = 'n_citation'

# %% Train and fit the models
# R-squared is also used to check how well the model captures the pattern and measure variance explained by the features
for model_name, model in [('dummy', dummy), ('elasticnet', elasticnet), ('gb', gb)]:
    print(f"Fitting model {model_name}")
    model.fit(X_train.drop([label], axis=1), np.log1p(X_train[label].values))
    
    for split_name, split in [("train", X_train),("validation", X_validation)]:
        pred = np.expm1(model.predict(split.drop([label], axis=1)))
        mae = mean_absolute_error(split[label], pred)
        r2 = r2_score(split[label], pred)  
        
        print(f"{model_name} {split_name} MAE: {mae:.2f}")
        print(f"{model_name} {split_name} R2: {r2:.2f}")
        
 # %%  STEP 5: PREDICT THE TEST SET
predicted = np.expm1(gb.predict(test))
test['n_citation'] = predicted
json.dump(test[['n_citation']].to_dict(orient='records'), open('predicted.json', 'w'), indent=2)
        
                     
# %%  HYPERPARAMETER
# This took several hours to tune the parameters because of the large number of trees and large dataset
random_grid = {'gb__n_estimators': [1000, 1200, 1500],
               'gb__learning_rate': [0.01, 0.05, 0.1],
               'gb__max_depth': [10, 12, 15],
               'gb__loss': ['squared_error', 'absolute_error']}

rs = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, 
                        n_iter = 20, cv = 3, random_state = 42, n_jobs=-1)
rs.fit(X_train.drop([label], axis=1), np.log1p(X_train[label].values))
print("Best Hyperparameters:", rs.best_params_)
