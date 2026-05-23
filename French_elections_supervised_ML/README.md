# Supervised Learning on Musical Dataset

## Introduction
The goal of this project is to apply supervised learning techniques to a musical dataset to classify and analyze musical tracks. This repository contains the code and resources used to process the data and train the models.

N.B: all the functions are defined in the file `helper.py`

## Phase 1: Data Preparation (data_prep_final.ipynb)

The first phase of the project focused on cleaning, exploring, and transforming the raw data into a format suitable for machine learning algorithms. The process is detailed below.

### 1. Data Exploration
We began by exploring the available datasets to understand their structure and identify quality issues.
- **Datasets Analyzed**: `genres`, `echonest`, `tracks`, and `spectral`.
- **Key Observation**: Significant missing values were identified, particularly in the `top_genre` (approx. 55% missing), `artist_latitude`, and `artist_longitude` columns.

### 2. Handling Missing Values

#### `top_genre` Imputation
To address the high percentage of missing values in the target variable `top_genre`, we implemented a hierarchical imputation strategy:
1.  **ID Conversion**: We converted `top_genre` from titles to IDs using the `genres` dataset to standardize the data.
2.  **Rooted Parent Imputation**: We traversed the genre hierarchy upwards to find a root parent genre.
    - If a dominant root parent was found, it was used to fill the missing value.
    - If multiple root parents existed without a clear dominant one, a parent was selected randomly.
    - *Justification*: This approach maximizes data utility by inferring likely genres based on the dataset's inherent hierarchy.

#### Artist Location
Missing values in `artist_latitude` and `artist_longitude` (approx. 60%) were handled as follows:
- **Imputation**: Missing values were replaced with `0`.
- **Feature Engineering**: A new binary column, `artist_location_unknown`, was created (1 for unknown, 0 for known).
- *Justification*: This preserves the signal that the location information was originally missing, which can be a predictive feature in itself.

### 3. Data Merging
We merged the main `tracks` dataset with the `spectral` dataset.
- **Exclusion of Echonest**: The `echonest` dataset was excluded at this stage because merging it would have reduced the total dataset size by approximately 90%.
- *Note*: We reserve the option to test `echonest` features in future iterations if accuracy improvements are needed.

### 4. Feature Reduction and Engineering
To optimize model performance and reduce multicollinearity, we analyzed feature correlations:
- **Correlation Analysis**: We calculated a correlation matrix for all features.
- **High Correlation (>90%)**: For groups of features exhibiting very high correlation, we retained only the mean column and dropped the others to avoid redundancy.
- **Principal Component Analysis (PCA)**: For the remaining features, we applied PCA to reduce dimensionality while retaining the most significant variance in the data.

### 5. Final Output
The processed and cleaned dataset has been saved as:
- **File**: `data/tracks_spectral_reduced.csv`

## Task 1: Predict the original genre (genre_top) (task1.ipynb)
- The supervised learning was performed on the numerical features of the dataset.
- The dataset has 19 numerical features. None of them was dropped. (F-test scores and variances are not low)
- The target is encoded to insure it is compact.
- 80% of the dataset is dedicated to the training and 20% for the test (with the same class proportions).
- Standardization of the features for the models that require it.
- The training data is split to 5 folds with the same class distribution. In each iteration on fold becomes the validation set and the other four are used as a trining set.

  **Results-1 (tracks_spectral dataset)**
  
  <img width="729" height="299" alt="image" src="https://github.com/user-attachments/assets/76319ccc-35fc-4186-b228-21651645c58c" />

The best model is **LightGBM** with an accuracy of 58% and an overfitting value within an acceptable range.
  
Before the final test :
- We try to find better hyperparameters to improve the best model's accuracy. That improves tha accuracy of 4.4%
- The model is trained again on the whole training/validation set
** Result **
  The final test result are : 65.44% accuracy
** Confusion matrix **
 <img width="708" height="422" alt="image" src="https://github.com/user-attachments/assets/5189d364-bb8e-415c-b0e5-2023e4b6f81e" />


- The diagonale is pretty dominant, which indicates a good class separation (65.44%)
- The best predictes classes are :
     - 11 (originally 38 = Experimental). 75% was well predicted
     - 7  (originally 15 = Electronic). 74% was well predicted
     - 5 (originally 12 = Rock). 69% was well predicte
  This is explained by the fact that thes 3 classes represent over 67% of the dataset. The other 11 classes are a minority.
The class 12 (763 = Holiday) was always mistaken for class 11 (38), and this is predictable as class 12 represents only 0,005% of the data.
All in all, there's a strong correlation between the percentage of the class in the data and the likelihood to predict it right.

We can also notice that the classes are confused the most with the dominant classes (11,7,5)
The results are explained by the imbalanced data.
<img width="257" height="292" alt="image" src="https://github.com/user-attachments/assets/c44236d8-1ac1-4313-934b-3ae372a0b783" />


**Results-2 (tracks_spectral_echonest dataset)**
we are going to work on the 3 datasets that provide more information (features) but less individuals.

<img width="759" height="332" alt="image" src="https://github.com/user-attachments/assets/79cb862e-118e-4d6b-8997-5cc02bd24c81" />

These results show a better accuracy but a higher overfitting.
Again, LightGBM has the best validation accuracy (73%). Even though it has a high overfitting (26%), it still generalizes well.
The difference between the previous results could be explained by the features that were added and that give the model more discriminative cues and more predictive power. Also the dataset is désormais less imbalanced than the data before : 
<img width="274" height="260" alt="image" src="https://github.com/user-attachments/assets/77a7afad-0d54-47be-81ea-7603a09be2c5" />

LightGBM is the model we're using for the final test (after the tuning).

Below, the confusion matrix : 
<img width="693" height="416" alt="image" src="https://github.com/user-attachments/assets/9da7d3f8-c9e3-47c6-8ee1-3aebf8ec18ad" />

The results are way better (accuracy : 79% Vs 65% before)
- The diagonal is dominant. Same, the dominant genres in the dataset are well predicted 5, 7 (55% of the dateset)
- The other classes are well predicted, but when confused, they're mainly confused with 5 and 7 (the dominants)

What I find pretty curious is that class 3 (originally 5 = classical ) is well predicted in results-1 and results-2:
- Results-1 : 67% well predicted and represent only 1,89% of the data
- Results-2 : 95% well predicted and represent only 2,91% of the data
But as class 3 is not very present, very few other classes are mistaken for class 3.
We can suppose that the feauture distribution of classical music (3) is very different from other classes. (A small class but easy to detect)

## Task 2: Predict your coarse-grained genre (3–4 categories) (task_2.ipynb)

# 

In task 2, further feature engineering was applied to build our coarsed genres and supervised prediction of genre_top. the two different merges of datasets are kept to compare the results : 'data/tracks_echonest_titles.tsv’ and 'data/tracks_spectral_titles.tsv’. 

Note : 

- Titles means that the genre top are represented as titles and not ids.

The preprocessing on the existing data is a bit different that the data_prep.ipynb, we chose to :

- Drop the rows (under 3% of data) with missing values in Speechiness, Valence and Danceability
- Drop artist_longitude an artist_latitude because they are not informative for the prediction and introduce a lot of missing values.
- Apply PCA on all spectral related features because they are highly correlated given the plot of correlation in the notebook task_2
- Apply PCA on the popularity related features ['favorites', 'interest', 'listens']

After the steps above, only 5 axis were necessary in the 1st PCA to have nearly 90% of the variance regarding the Spectral Features, and only 1 axis for the Popularity Features. No PCA was applied on the Core Audio Features (['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration']) because the correlation is very small.

Spectral Features and Popularity Features were then substituted with their equivalent PCA components in both datasets. Therefore the datasets were reduced to these dimensions : 

- (97288, 12) for the Spectral dataset
- (10405, 20) for the Echonest dataset

We experimented with embedding on textual data in order to preserve these information on tracks (['artist_name', 'title', 'album_title']) addidng the whole texts in one single string then embedding it using the ranall-MiniLM-L6-v2 Huging Face transformer. Nevertheless the it did not improve performance nor reduce dimensionality as intended : 

- We had +300 dimension after the embeddings
- The PCA was able to gather slightly above 70% of variance with +100 axis, which is not optimal.

Given these results, we excluded contextual features from further modeling.

In order to define the coarsed genre groups, we tried to apply k-means on our data and see how the model would classify the genres we have, the 1st went through auto-encoding the Core Audio Features in order to diminish the dimension but unfortunately the 3 clusters had all genre_top in all of them + nearly the same repartitions of Core Audio Features, which is provided no meaningful separation .

The exact same thing happened when we tried applying the K-means without the auto-encoding.

Conclusion : Attempting a model based coarse genre grouping using the existing datasets is not efficient.

To further refine the dataset for a better model based coarse, a pivot was applied on the Echonest dataset in order to have lines representing genres and where features of all tracks corresponding to that genres were grouped and reduces using the median. Next we removed the features with variance less than 0.01 to improve the efficiency of the K-means clustering. 

We then found a good clustering with good inter-class variability for the echonest features:
<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/c1d4aa4d-2ef5-4840-9e3a-fbb3f6f0dca7" />

The K-means were run many times using 3 or 4 clusters each time, the most relevent clustering is the one we were based on to decide on our final 3 coarse genres :

c1 = Electronic/Hip-Hop/Pop/Rock/International

c2 = Jazz/Blues/Instrumental/Easy Listening

c3 = Classical/Folk/Experimental

Note : *A lot of dataset treatments are only done on the Echonest dataset and that is because it is the dataset with all of the features, it indeed reduce Top Genre diversity, but we well adress that later.*

Note : *Other missing Top Genres will be added when applying the preprocessing and coarse on the Spectral Dataset.*

The next phased consisted of testing models to predict the right coarse-genre, using a proportion of 0.2 for testing, the ML models used are :

- Extreme Gradient Boosting (*XGBoost models*):
    
    The 1st training/testing gave us an accuracy of 0.85, which is good, but the recall on cluster c2 was very low, notably because it is the cluster the least represented in the datasets (+10 times less than c1 and 2 times less than c3). In order to address this issue we set weights for the classes, the best recall, accuracy and F1 error combination was for this weight setting : class_weights = {c1: 1.0, c2: 10, c3: 1.5}
    
- Random Forests :
    
    We kept the same distribution of weights as for XGBoosting in all the other models, the accuracy is the same but this technique proved insufficiency predicting c2 class (recall of 0.09)
    
- Logistic Regression
    
    LogReg is too aggressive on c2, many false positives, hurting overall performance and c3 badly. 
    

Conclusion : XGBoost is clearly the better compromise: higher macro‑F1 and much higher accuracy, with reasonably good c2.

We then applied the same treatment on the spectral dataset since it contanins nearly 10 times the number of tracks of the echonest one and two more genres , we had very poor results on the clustering methods which we interprete the following way:
    -Clustering purely on spectral features gives a rough, timbre-based grouping, often separating electronic vs. acoustic music, but it fails to capture rhythm, melody, vocals that define most genres. For a more robust clustering, we chose to stay with the first approach of  combiing spectral and echonest features even if it has less tracksit generalizes better.

## Task 3: Predict the track duration (task3.ipynb)

**Data Preparation**:  
- **Categorical Encoding:** Instead of dropping high-cardinality columns, we applied **Target Encoding** (with smoothing) to `artist_name` and `album_title`. This allows the model to learn from the average duration associated with specific artists and albums, preserving critical signal.
- **Target Transformation:** The target variable, `duration`, was handled for extreme outliers by **clipping** at 600s and then **log-transformed** (`log(1+x)`) to achieve a near-normal distribution for regression stability.
- **Final Feature Sets:** Two primary feature sets were used for modeling:
    1.  **Set A (Large):** Metadata (Target Encoded) + Reduced Spectral Features ($\approx 100,000$ tracks).
    2.  **Set B (Complete/Small):** Metadata(Target Encoded) + Reduced Spectral + **Echonest** features ($\approx 10,000$ tracks).

**Regression models:**  
The goal is to build the best regression model to predict the log-transformed track duration. We implemented advanced Gradient Boosting methods to handle non-linear relationships and compared them using $\mathbf{R^2}$ score and $\mathbf{MAE}$ (Mean Absolute Error) converted back to seconds.

| Model | Feature Set | Data Size (Tracks) | R² Score | MAE (Seconds) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **Set A** | $\approx 97,000$ | **0.5247** | **73.87 seconds** |
| **LightGBM** | **Set A** | $\approx 97,000$ | **0.5245** | **74.05 seconds** |
| Random Forest | Set A | $\approx 97,000$ | 0.5044 | 74.58 seconds |
| LightGBM | Set B | $\approx 10,000$ | 0.4186 | 60.32 seconds |
| XGBoost | Set B | $\approx 10,000$ | 0.4131 | 60.23 seconds |
| Random Forest | Set B | $\approx 10,000$ | 0.3853 | 61.16 seconds |



The results indicate a shift in the **Quantity vs. Quality Trade-off** due to better Feature Engineering:

* **Best Overall Performance (Set A) :** The XGBoost and LightGBM models on the full dataset achieved the highest predictive power ($R^2 \approx 0.525$). This confirms that leveraging the full dataset with **Target Encoding** for artists and albums provides significantly more signal than using a smaller subset with specialized features.
* **The MAE vs. $R^2$ Trade-off:** Interestingly, Set B has a lower Mean Absolute Error (~60s) compared to Set A (~74s), despite having a much worse $R^2$. It is possibly due to the fact that Set A includes a massive variety of tracks, including long mixes and outliers, which increases the total variance and the average error margin. However, the model on Set A captures the trends much better (higher $R^2$), whereas the model on Set B is "safer" (smaller errors) but fails to explain the variance in the data effectively.

* **Conclusion:** The XGBoost and LightGBM models on Set A are chosen as the champion models. The ability to explain >52% of the variance in song duration using only metadata and spectral features is a strong result for this domain.