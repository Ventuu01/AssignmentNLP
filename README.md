# AssignmantNLP
The task at hand is to divide a text input into "medical" and "non-medical" classifications. In this case, I used Python libraries like:
• Requests: This Python module is used to send HTTP requests to the Wikipedia API.
• NLTK = for text preprocessing 
• scikit-learn = to turn text data into numerical features, I used CountVectorizer. I then divided the dataset into training and testing sets. In the end, I trained the Naïve Bayes classifier and evaluated the model's performance.
• BeautifulSoup: used to remove HTML tags from the text leaving only the content inside of them. 

To train the classifier, I chose a collection of Wikipedia articles from the "medical" and "non-medical" categories. Text from the selected articles was extracted using the Wikipedia API. The article text is obtained in JSON format using the get_wikipedia_text method. To prepare the extracted text for analysis, I performed a number of preprocessing steps (preprocess_text), such as lemmatization, tokenization, removal of stopwords, and removal of HTML elements.
I used scikit-learn's CountVectorizer to apply the Bag of Words approach and turn the preprocessed text into numerical features.
Using the training data, a Multinomial Naive Bayes classifier was trained, and a set of test data was then used to evaluate the model. The classification report and accuracy were two important measures used to evaluate the model's performance.
New data, in the form of a list of Wikipedia page names, were used to test the model. The same preprocessing procedures were applied to this data before the model could make predictions.
