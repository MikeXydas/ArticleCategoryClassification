# Article Classification

A python module which trains different classification models in order to predict the category an article belongs.  

## Introduction
  
The train_set consists of 12k articles which are used to train our models, in order to be able to classify  
articles on a category according to their content.  

**Classification algorithms used:**

- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
- K-Nearest Neighbors (KNN)

## Running:

After having built an environment with the `requirements.txt` installed run the `classify.py` module

## Summary of the module

1. **Importing and preprocessing**  
    We first import the trainset data from the .csv using pandas and storing them in a dataframe  
    and we perform the following preprocessing in the content of each article:
    - Apply Title: Considering that the title of the article has some effect on deciding its category  
    we append it at the end of the text so as the content of the article has a fixed percentage of its content bei the title  
    - Lower Case: We turn all the uppercase letters to lowercase  
    - Stopwords: We remove from the content of the article a set of stopwords which do not offer any useful information  
    Like "the", and", "have" etc.
    - Punctuation: We remove any punctuation
    - Lematisation: determining the lemma of a word based on its intended meaning: better -> good, walked -> walk
    
2. **Vectorising**  
    We will first explain what the simple CountVectorizer does and then what extra the TfidVectorizer offers (which is the one that we use).   
    The countVectorizer will have number of articles rows and number of unique words appearing in any text columns.
    And for each text for each word of that text we will increase by one the corresponding cell.  
    For example:  
       **A: "The cat woke me up to feed her"**   
       **B: "The cat bowl of the cat was full"**   
        
      Texts 2: A, B  
      Unique Words 13: The, cat, woke, me, up, to, feed, her, bowl, of, the, was, full
      
      Result of count vectorizer:  
      <pre>
        The cat woke  me  up  to feed her bowl of the was full  
      A  1   1    1    1   1   1   1   1    0   0   0   0   0  
      B  1   2    0    0   0   0   0   0    1   1   1   1   1    
      </pre>
      
      This is the vector that the classifiers can take as an input in order to do their training(fitting).  
      It is a simple way of turning words into numbers and more specifically vectors which allow math to be done on them.  
      
      However, in the `classify.py` we use the Tfid_vectorizer. In a similar way instead of having a number of appearances of 
      each word we have a number f representing a frequency of that word (0 <= f <= 1).  
      More on the tfid_vectorizer: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
     An important parameter of the tfid_vectorizer which can be tweaked in order to achieve optimum accuracy is the max_fd. Max_fd is an upper limit of the allowed frequencies of each word. For example if we have a max_df of 0.8 and a word appears into 9 out of 10 articles then that specific word won't make it into the vector.  
     
3. **LSI truncating (Latent Semantic Indexing)**  
   As you may expect, the train_set is big and we will end up with a vector that has too many columns. LSI is a way of reducing the column number without losing much of the useful information. This ends up on a great speed up of the training and cross-validation following.  
   
4. **Fitting and cross-validation predicting**  
   The parameters of the models were mostly chosen through repeated executions using grid_search. 
   Then we perform a kFold. kFold slices our X_train and Y_train into k equally sized parts. We then use k - 1 parts for fitting (training the model) and the k part to test the accuracy of our fitted models.  
   This is repeated k times so as each part becomes exactly one time the test part.   
   The metric returned for each model is the accuracy. You can also change that and show more metrics such as precision or f-measure.  
