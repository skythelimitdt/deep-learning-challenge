# Build a Neural Network to Predict Success of a Company
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received access to a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Run a Model for the Initial Dataset
### Data Preprocessing
- Target: 'IS_SUCCESSFUL' (Binary Classification Target)
- Removed:'EIN', 'NAME'
- Features: APPLICATION TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT
- Feature Transformations:
    - Binned Categories: CLASSIFICATION, APPLICATION_TYPE (Grouped less frequent categories into broader bins)
- Applied One-Hot Encoding: Categorical features were encoded using OneHotEncoder
- Merged Encoded Data: Combined the transformed dataset with the original application_df
- Standardization: Scaled numerical features for optimal model performance

### Build Model
- Features: 78
- 2 Hidden Layers: 78 and 30 neurons respectively. relu activation is applied
- 1 Output layer: 1. sigmoid activation is applied

**model_accuracy is about 0.7269**

**Overfitting Analysis: Training vs. Validation Loss**
The **training and validation loss chart** indicates an overfitting issue.

![chart](https://github.com/skythelimitdt/deep-learning-challenge/blob/main/Resources/training_validation_cost.png)

**Observation:** Training loss continuously decreases, showing the model is learning the training data well. Validation loss does not follow the same trend and remains high, suggesting the model is not generalizing well to unseen data.
The model is memorizing patterns from the training data instead of learning generalized relationships, leading to poor performance on test data.

### Random Forest Model: Features' Importance
- Performed a Random Forest analysis to evaluate feature significance in predicting IS_SUCCESSFUL
- Identified the top 5 most influential features:
    1. ASK_AMT
    2. AFFILIATION
    3. APPLICATION_TYPE
    4. CLASSIFICATION
    5. ORGANIZATION

![random_forest](https://github.com/skythelimitdt/deep-learning-challenge/blob/main/Resources/randomforest.png)

This analysis helps prioritize the most impactful variables, guiding feature selection and potential model optimization.

## Optimization Strategies
To address overfitting and improve model performance, the following optimization techniques were tested:

- Reduced Model Complexity – Decreased the number of neurons per layer to prevent overfitting.
- Dropout Regularization – Applied dropout layers to reduce over-reliance on specific neurons.
- Early Stopping – Stopped training when validation loss stopped improving to prevent overtraining.
- Batch Size Tuning – Experimented with different batch sizes to optimize gradient updates.
- L2 Regularization – Added L2 penalties to weight parameters to control model complexity.
- Activation Function Tuning – Tested different activation functions (ReLU, Tanh, Swish) for improved learning.
- Feature Selection – Selected the most important features based on Random Forest analysis to improve efficiency.

### 1. Reduce Complexity: 
Reduced complexity of the model by reducing the number of neurons. 1st hidden layer: 32, 2nd hidden layer: 16

**Accuracy was about 0.7291**

### 2. Set up keras tuner to test different hyperparameters 
Keras Tuner was used to automate hyperparameter tuning, optimizing the model by testing various configurations for better performance. The following strategies were applied using Keras Tuner:
- Reducing Model Complexity
Keras Tuner tested different neuron counts, reducing overfitting by selecting the optimal architecture.
- Dropout Regularization
Keras Tuner explored different dropout rates (e.g., 0.2 to 0.5) to prevent overfitting.
- Early Stopping
Implementation: Prevented excessive training by stopping when validation loss stopped improving.
- Batch Size Optimization
Keras Tuner evaluated different batch sizes (e.g., 16, 32, 64) to balance training stability and speed.
- L2 Regularization
Implementation: Applied L2 weight regularization to control complexity and avoid overfitting.
- Activation Function Selection
Tested different activation functions (ReLU, Tanh, Swish) to optimize neuron activation behavior.

**Best model was found with 0.7343 accuracy**



### 3. Refining Model Optimization Through Feature Selection1: 
As part of the last optimization phase, less important features were removed before model tuning to enhance efficiency and prevent unnecessary complexity.
- Feature Importance Analysis:
    - Utilized Random Forest feature importance scores to identify the most influential variables.
    - Features with an importance score below 0.05 were excluded to streamline the model.

- Features Included on this model are:
- ASK_AMT
- AFFILIATION
- APPLICATION_TYPE

**Accuracy was about 0.7243**

### 4. Refining Model Optimization Through Feature Selection2:
- Features Removed (based on Feature Importance) if Feature Importance total for that category was less than 0.5:
    - USE_CASE
    - SPECIAL CONSIDERATIONS
    - INCOME_AMT
    - STATUS

**Accuracy was about 0.7282**


### SUMMARY
The best model achieved an accuracy score of 0.7335. Despite multiple optimizations, including feature selection, hyperparameter tuning, and regularization techniques, further improvements were limited.

**Key Observations:**
- The model showed signs of performance saturation, where additional adjustments did not significantly enhance accuracy.
- The dataset and feature transformations may not be fully aligned with the current neural network architecture.
- The complexity of the neural network may not be necessary for this binary classification problem.

**Recommendations:**
Alternative Models:
- A Logistic Regression model could serve as a baseline to compare performance, as it is simple, interpretable, and often effective for binary classification.
- Random Forest or Gradient Boosting (XGBoost, LightGBM) may provide better performance by capturing non-linear relationships.
- Support Vector Machines (SVMs) could be tested if feature scaling is well-applied.

Given the results, testing a simpler and more efficient classification model such as Logistic Regression could provide a strong benchmark while reducing computational overhead. 

**Best Hyperparameters:**
- activation: relu
- reg_strength: 0.0
- first_units: 18
- num_layers: 1
- units_0: 24
- dropout_0: 0.1
- batch_size: 32
- learning_rate: 0.001
- units_1: 32
- dropout_1: 0.1
- units_2: 104
- dropout_2: 0.2
- units_3: 48
- dropout_3: 0.4
- units_4: 56
- dropout_4: 0.4
- units_5: 112
- dropout_5: 0.4
- tuner/epochs: 10
- tuner/initial_epoch: 0
- tuner/bracket: 0
- tuner/round: 0
- Best Activation Function: relu
- Best First Layer Units: 18
- Best Number of Hidden Layers: 1

## Resources
- ASU Bootcamp Classes
- chatGPT for keras tuner setup
