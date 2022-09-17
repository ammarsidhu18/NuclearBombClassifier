# Nuclear Bomb Explosion Classifier
Created a multiclass classification model with **92% accuracy** using imbalanced classification techniques on nuclear bomb explosions data to help third-party inspectors determine the deployment method of a given nuclear bomb.

# Data & Problem
* **Problem**: The goal of this problem is to explore **multiclass classification** (a sample can only be one of many things) on nuclear bomb explosion data.
This is because we're going to be using a number of different **features** about a nuclear bomb to predict how the bombs were deployed: Air, Underground, Surface, and Underwater.
In a statement,
> Given nuclear bomb parameters about an explosion, can we predict how the nuclear bomb was deployed?
* **Data Acquisition**: [Tidytuesday DataSets Repository: Nuclear Explosions Dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-08-20)
* **Sucess Metrics**: If we can reach a **95% accuracy** at predicting/classifying how a nuclear bomb was deployed (e.g., Air, Underground, Surface, etc.) during this proof of concept, we'll pursue this project.

# Code & Resources Used
* **Python Version**: 3.8
* **Environment**: Miniconda, Jupyter Notebook
* **Packages**: Pandas, Scikit-Learn, NumPy, Seaborn, Joblib

# EDA & Feature Engineering
After loading the dataset and inspecting the dataset's features (nuclear bomb explosion attributes), and target variable (deployment method), I needed to clean the data and visualize it to better understand the relationship between the features, and the target as well as prepare the data for modeling. I conducted the following steps with this dataset: 
* Found the number of unique values in the dataset through creating a dictionary
* Cleaned target column by merging deployments types that were of the same category but named differently to end up with **4 deployment types (classes)**:
  1. `Underground`
  2. `Air`
  3. `Surface`
  4. `Underwater`
  * Bar Plot of Original Deployment Methods (Frequency per Deployment Method):
  
  ![rawdeploymentsBarGraph](https://user-images.githubusercontent.com/46492654/190839821-358d3b74-a6ce-4627-b485-b3415581854b.png)
  
  * Bar Plot of Cleaned (Desired) Deployment Methods (Frequency per Deployment Method):
  
  ![cleaneddeploymentBarGraph](https://user-images.githubusercontent.com/46492654/190839833-2ecf8f68-86f0-49d8-a7e9-32d68c992675.png)
  
* Determined a normalized count of the deployment methods for the nuclear bombs (target variable; where 0 implies the nuclear bomb was deployed underground, 1 implies the bomb was deployed in the air, 2 implies that the bomb was deployed on the surface, and 3 impies that the bomb was deployed underwater) to conclude that the target column being dealt with is **imbalanced**.
* The data is consists of more than **5x** as many nuclear bombs that were **deployed underground** than the other **3 deployment methods**, so we have an **imbalanced classification** problem.
* Seperated continuous and categorical columns to acquire summary statistics on the continuous features.
* Checked for total number of missing values in each column and found that there were **9 missing values** in this dataset, which were dropped given that they represented the estimates of the explosion powers, but the dataset contained over 2000 nuclear explosions (imputing these values with a statistic would do little to our predictions given the size of our dataset). 
* Checked for duplicate values in the dataset and found **no duplicate values**.
* Visualized nuclear bomb deployment method by country and found the following:
  * The most common deployment method of nuclear bombs is underground, and the the least common is underwater.
  * Furthermore, the USA has detonated substantially more nuclear bombs than all other countries in every deployment method with exception to deployment via air; where the USSR has detonated almost two times the nuclear bombs than USA did.
  * India and Pakistan have deployed the least amount of nuclear bombs across all deployment methods when compared to the other nations.
  
![deploymentVsCountries](https://user-images.githubusercontent.com/46492654/190839858-d1f03440-e999-496c-9726-82cb12e3db71.png)

* Visualized upper estimate of explosion power and magnitude of explosion by deployment type and found no linear relationships. However, it seems that stronger nuclear bomb explosions in kilotons have lower bodywave mangnitudes. Additionally, the most explosive nuclear bombs were deployed in the air followed by the surface and then underground. The weakest nuclear bombs by explosion power were deployed underwater.

![explosionpowerVsmagnitude](https://user-images.githubusercontent.com/46492654/190839873-44b2a734-af5a-4b37-9cb9-c9b3eb103214.png)

* Created univariate data visualizations of all features through plotting:
  * Bar Plots:
  
  ![bargraphs](https://user-images.githubusercontent.com/46492654/190839893-de765a87-979f-4963-b34e-398f2c627b4c.png)
  
  * Box Plots:
  
  ![boxplots](https://user-images.githubusercontent.com/46492654/190839910-0ab08306-ed69-4f5f-802e-5cddfcbc4516.png)
  
* Created bivariate data visualizations comparing continuous and categorical features to target variable:
  * Distribution Plots of Continuous Features by Target Variable:
  
  ![densityplots](https://user-images.githubusercontent.com/46492654/190839917-4b89f061-41bc-420a-9106-1250b5ac18d7.png)
  
  * Developed correlation matrix of continuous features:
  
  ![correlationmatrix](https://user-images.githubusercontent.com/46492654/190839919-236f029d-c5fc-4109-a762-5b9a5b865a7e.png)
  
  * Heatmap of all features:
  
  ![heatmap](https://user-images.githubusercontent.com/46492654/190839931-9bb10588-07b8-498b-856c-03239187e5c5.png)
  
* Concluded that from correlation matrix: none of the features have a strong correlation with each except for the yield_upper and yield_lower features, which are partically the same features with different estimations of explosion powers for a given nuclear bomb. Their correlation is strong and in the positive direction.
* All the other features have correlation coefficients between -0.6 or 0.6 (not inclusive) indicating that none of the features or targets have strong correlations; rather relatively weak correlations between each other.
* There is **no apparent linear relationship** in the negative or postive direction according the correlation matrix.
* Intuitively, the method of deployment for the nuclear bomb may have a stronger relationship with the purpose of the nuclear bomb. The explosion power of the bomb will ultimately depend upon the purpose and not how it was deployed.

# Feature Engineering
* Dropped `date_long`, `year`, and `id_no` features because they are not relevant features for predicting for nuclear bomb deployment methods (poor predictors). 
* Split data into **X (features)** and **y (target)**; then I split both, the X and y, into training and test sets with **test set size of 20%** for both X and y.
* Label Encoded categorical features (`country`, `region`, `source`, `purpose`) to numerical features.

# Model Building
Based on the [Scikit-Learn Algorithm Selection Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html), I trained 5 classification algorithms and evaluated them using accuracy as the primary evaluation metric. I chose accuracy for checking baseline models because it is to interpret how well the models predict nuclear bomb deployment methods.

The 5 models I selected for the classification:
  1. `Logistic Regression`
  2. `Support Vector Machine Classification`
  3. `Decision Tree Classifier`
  4. `Random Forest Classifier`
  5. `Gradient Boosting Classifier`

* Since this was a **multiclass classification problem** and most machine learning algorithms assume that all classes have an equal number of classes, the algorithms were modified to account for the **imbalance** with **Cost-Sensitive Learning**. 

# Model Performance

From the baseline model perfomance, it is apparent that the ensemble classifiers performed really well; with the **Random Forest Classifier** achieving an **accuracy** of **93%**, and Gradient Boosting Classifier achieving an **accuracy** of **95%** without taking into account the imbalanced classes. To account for the lack of balance between the classes, the baseline algorithms were modified to bias towards the classes that have fewer observations in the training dataset. Therefore, **Cost-Sensitive Learning** was employed. 

![baselinemodelperformance](https://user-images.githubusercontent.com/46492654/190839946-f019b030-4e57-4cc3-96ce-df63ea9a1891.png)

**Cost-sensitive learning** reduced the overall accuracy slightly, but it ensured that the rare classes are taken into consideration with equal weight. The best performing cost-sensitve learner was the **Random Forest Classifier** with an **overall accuracy** of about **90%**. 

![costsensitivemodelperformance](https://user-images.githubusercontent.com/46492654/190839959-f1b8bec4-b6a5-48be-bf2f-c8d30e22d7dc.png)

# Model Improvement
The **Random Forest Classifier** had the highest accuracy while handling the imbalanced data, so I tuned its hyperparameters with **RandomSearchCV** and **GridSearchCV**, and cross-validated the metrics with **5-folds cross-validation**. The best hyperparameters were obtained through **GridSearchCV**, and an overall cross-validated average accuracy of **87%** was obtained by the classifier. 
* A **classification report**, and a **confusion matrix** were created from this model. The following foundings were discovered from these 2 about the classes:
  * There were **5** ocassions where the `Surface` deployment type was not labeled correctly; labeled as `Air` 3 times and `Underwater` twice.
  * There were **17** ocassions where the `Air` deployment type was not labeled correctly; labeled as `Underwater` 7 times and `Surface` 10 times.
  * There were **8** times the `Underwater` deployment type was not labeled correctly; labeled as `Air` 2 times and `Surface` 6 times.
  * Finally, the `Underground` deployment type was labeled incorrectly  once and that was as the `Air` deployment type.

<img width="550" alt="ClassificationReport" src="https://user-images.githubusercontent.com/46492654/190840025-7f06c3d4-4e5c-481d-a8c2-b1ed9fe94521.png">

![confusion matrix](https://user-images.githubusercontent.com/46492654/190839988-e45dbd3b-c389-44db-bad5-cda5d0cd0ccd.png)

# Feature Importance
Which features contribute most to a model predicting/classifying a nuclear deployment type?
* `purpose`, `yield_upper`, `yield_lower`, and `depth` are the most important features for determining/predicting the deployment methods of nuclear bombs.
* `magnitude` and `country` are the least important features for determining/predicting the deployment methods of nuclear bombs.

![featureimportance](https://user-images.githubusercontent.com/46492654/190840033-bac418a1-9f1b-4073-94b8-02c43cf80575.png)

# Conclusion
Did the final and best model achieve the desired accuracy from the problem statement?

> If we can reach a 95% accuracy at predicting/classifying nuclear bomb deployment methods during this proof of concept, we'll pursue this project.

Since the highest accuracy our model achieved was , the target was 92.4 the target was not achieved.

Further experimentation will be required, such as testing different models (CatBoost? XGBoost?), trying to tune different hyperparameters, and selecting the most important features for the prediction process. Through these steps, achieving an accuracy closer to or beyond 95% is certainly possible.

 
