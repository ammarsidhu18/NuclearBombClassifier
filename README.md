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
  * Bar Plot of Original Deployment Methods (Frequency per Deployment Method)
  
  * Bar Plot of Cleaned (Desired) Deployment Methods (Frequency per Deployment Method)
* Determined a normalized count of the deployment methods for the nuclear bombs (target variable; where 0 implies the nuclear bomb was deployed underground, 1 implies the bomb was deployed in the air, 2 implies that the bomb was deployed on the surface, and 3 impies that the bomb was deployed underwater) to conclude that the target column being dealt with is **imbalanced**.
* The data is consists of more than **5x** as many nuclear bombs that were **deployed underground** than the other **3 deployment methods**, so we have an **imbalanced classification** problem.
* Seperated continuous and categorical columns to acquire summary statistics on the continuous features.
* Checked for total number of missing values in each column and found that there were **9 missing values** in this dataset, which were dropped given that they represented the estimates of the explosion powers, but the dataset contained over 2000 nuclear explosions (imputing these values with a statistic would do little to our predictions given the size of our dataset). 
* Checked for duplicate values in the dataset and found **no duplicate values**.
* Visualized nuclear bomb deployment method by country and found the following:
  * The most common deployment method of nuclear bombs is underground, and the the least common is underwater.
  * Furthermore, the USA has detonated substantially more nuclear bombs than all other countries in every deployment method with exception to deployment via air; where the USSR has detonated almost two times the nuclear bombs than USA did.
  * India and Pakistan have deployed the least amount of nuclear bombs across all deployment methods when compared to the other nations.
* Visualized upper estimate of explosion power and magnitude of explosion by deployment type and found no linear relationships. However, it seems that stronger nuclear bomb explosions in kilotons have lower bodywave mangnitudes. Additionally, the most explosive nuclear bombs were deployed in the air followed by the surface and then underground. The weakest nuclear bombs by explosion power were deployed underwater.
* Created univariate data visualizations of all features through plotting:
  * Bar Plots:
  
  * Box Plots:
  
* Created bivariate data visualizations comparing continuous and categorical features to target variable:
  * Distribution Plots of Continuous Features by Target Variable:
  
  * Developed correlation matrix of continuous features:
  
  * Heatmap of all features:
  
* Concluded that from correlation matrix: none of the features have a strong correlation with each except for the yield_upper and yield_lower features, which are partically the same features with different estimations of explosion powers for a given nuclear bomb. Their correlation is strong and in the positive direction.
* All the other features have correlation coefficients between -0.6 or 0.6 (not inclusive) indicating that none of the features or targets have strong correlations; rather relatively weak correlations between each other.
* There is **no apparent linear relationship** in the negative or postive direction according the correlation matrix.
* Intuitively, the method of deployment for the nuclear bomb may have a stronger relationship with the purpose of the nuclear bomb. The explosion power of the bomb will ultimately depend upon the purpose and not how it was deployed.

# Model Building

 
