import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import mean_squared_error

attributes = pd.read_csv("./data/attributes.csv")
product_descriptions = pd.read_csv("./data/product_descriptions.csv")
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
sample_submission = pd.read_csv("./data/sample_submission.csv")

count = CountVectorizer()
docs = np.array(['The sun is shining',
                  'The weather is sweet',
                  'The sun is shining and the weather is sweet'])


bag = count.fit_transform(docs)

tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

data = 1

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


tokenizer_porter('runners like running and thus they run')

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)



rf_rfidf = Pipeline([('vect', tfidf), ('rf', RandomForestRegressor(random_state=0))])
X_train = train['search_term'].values
y_train = train['relevance'].values

rf_rfidf.fit(X_train, y_train)

X_test =test['search_term'].values

y_hat = rf_rfidf.predict(rf_rfidf.transform(X_test))




X_train, X_validation, y_train, y_validation = train_test_split(train['search_term'].values,
                                                    train['relevance'].values, test_size=0.4,
                                                    random_state=0)


rf_rfidf = Pipeline([('vect', tfidf), ('rf', RandomForestRegressor(random_state=0))])
rf_rfidf.fit(X_train, y_train)
y_hat = rf_rfidf.predict(X_validation)
mean_squared_error(y_true=y_validation, y_pred=y_hat)

X_test = test['search_term'].values

sample_submission['relevance'] = rf_rfidf.predict(X_test)

sample_submission.to_csv('./data/random_forest_submission.csv', index=False)
