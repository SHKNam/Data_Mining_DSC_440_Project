# Data_Mining_DSC_440_Project
Comparative Analysis of Classifier Performance on Mental Health and Substance Abuse Data

# Abstract:
This comprehensive report systematically delves into the intricacies of five cutting-edge classifiers, namely Logistic Regression, XGBoost, Random Forest, Catboost, and LightGBM. The evaluation extends beyond the conventional analysis, encompassing both raw data and data preprocessed with FP-Growth frequent pattern mining.
The performance assessment hinges on two pivotal metrics—cross-validation accuracy and cross- validation F-1 score. By scrutinizing these classifiers across diverse datasets, we gain invaluable insights into their effectiveness in discerning the demographic distributions related to Mental Disorders. Furthermore, the report meticulously examines the classifiers' efficacy in predicting tendencies towards Mental Disorders, adding a critical dimension to the analysis.
In essence, this study not only illuminates the nuances of classifier performance but also provides a nuanced understanding of the demographic landscape associated with Mental Disorders. The integration of advanced preprocessing techniques elevates the evaluation, offering a more refined perspective on the classifiers' capabilities.

# Introduction:
The National Survey on Drug Use and Health (NSDUH) dataset stands as a cornerstone in the comprehensive examination of substance use and mental health trends within the United States. With 6,509,025 rows and 40 attributes, this dataset is a critical and expansive resource that plays a pivotal role in shaping public health policies, crafting effective prevention and treatment programs, and supporting academic research initiatives.
In the context of this study, we leverage the rich MHCLD dataset for a classification task aimed at discerning patterns and trends in substance use and mental health. The classification analysis, employing cutting-edge algorithms such as Logistic Regression, XGBoost, Random Forest, Catboost, and LightGBM, seeks to enhance our understanding of the factors influencing substance use and mental health outcomes. By applying machine learning techniques to this vast dataset, we aim to contribute to the refinement of prevention and intervention strategies, ultimately fostering a more targeted and effective approach to addressing mental health challenges in our society.

# Encodings
Within our dataset, we encounter approximately 13 target variables. The table presented below encapsulates the corresponding value encodings. These encodings adhere to the MHCLD codebook [6], meticulously outlined to provide a comprehensive understanding of the representations conveyed in our dataset. This table serves as a key reference, elucidating the mapping between the target variables and their encoded values, as defined by the MHCLD codebook [6]. It plays a pivotal role in facilitating accurate interpretation and analysis of our data by establishing a standardized framework for referencing and comprehending the diverse target variables embedded within our study.


# Methodology:
The project employed two distinct data preprocessing methodologies for performance evaluation:

# Direct Classification Method:
## 1. Raw Data:
### Figure 1:Direct Classification method
The initial phase of the study involves handling raw data, which is stored in an 'rdata' file format. This format necessitates the use of specialized tools for data manipulation. To this end, the pyreadr library is employed, providing an efficient means of file reading and data access. This step is crucial for preparing the data for subsequent processing and analysis.
## 2. Data Pre-Processing:
During the data pre-processing stage, the methodology is guided by the instructions detailed in the codebook. Specifically, columns with -9 values are identified as null and are treated with caution to avoid significant data loss. Since the dataset comprises exclusively categorical variables, null values are replaced with 0. This approach is taken to preserve the integrity and continuity of the data, ensuring that subsequent analysis is based on a complete and coherent dataset.
## 3. Data Splitting:
The data splitting process involves dividing the dataset into training and testing sets, following an 80-20 ratio. This strategic split is crucial for the effective training of classifiers and for ensuring a robust evaluation process. By allocating 80% of the data for training and the remaining 20% for
testing, the study ensures that the classifiers are well-trained on a substantial portion of the data while retaining a significant portion for unbiased evaluation.
## 4. Model Definition:
In the model definition phase, the architecture of the model is carefully designed to accommodate the specific characteristics of the dataset. This step is pivotal as it lays the groundwork for the classifier's training. The chosen architecture is tailored to optimize the processing and analysis of the categorical data, setting a strong foundation for the model's subsequent performance.
## 5. Cross Validation:
To enhance the generalization of the model, stratified cross-validation is implemented. This method ensures that each class is proportionally represented in both the training and validation sets. Such a balanced approach to validation is instrumental in providing a fair and comprehensive evaluation of the model, taking into account the diverse characteristics present in the dataset.
## 6. Data Evaluation:
The final stage, data evaluation, focuses on assessing the model's performance using two key metrics: accuracy and the cross-validation F1 score. These metrics provide a holistic view of the model's effectiveness, encompassing both the overall accuracy and a balance between precision and recall. This dual-metric evaluation offers a thorough and nuanced understanding of the model's capability in handling and analyzing mental health and substance abuse data.


# FP-Growth Augmented Classification Method
## 1. Raw Data Initialization:
The initial stage involves managing unprocessed data stored in an 'rdata' file format. To enable efficient data access and manipulation, the pyreadr library is utilized, owing to its proficiency in reading files. This choice ensures that the raw data is accessible and ready for the forthcoming stages of preprocessing and analysis.
## 2. Data Pre-Processing:
In the pre-processing phase, columns with -9 values, as specified in the codebook, are identified as null. These values are then replaced with the Pythonic representation of null, None. This step is critical in standardizing the data format, thus preparing it for more effective processing and analysis.
## 3. Data Encoding with Abbreviations:
To facilitate interpretability in subsequent frequent mining algorithms, a data encoding scheme is implemented. For example, the variable 'age', originally represented as the integer 1, is transformed into '1AGE', denoting the 0-10 years age group. This method of encoding is applied across various significant variables, enhancing the clarity and usability of the data for pattern mining. 
## 4. Dataset to Transaction Conversion:
Each row in the dataset is converted into a transaction, which is essential for analyzing the data using the FP-Growth algorithm. A transaction encoder is used during this process, and careful attention is given to the removal of null values within transactions to ensure data integrity
## 5. FP-Growth Pattern Mining:
The FP-Growth pattern mining algorithm is employed to uncover meaningful patterns in the dataset. The support parameter is set at 0.1, establishing the minimum frequency threshold for a pattern's significance, thus focusing the analysis on the most relevant patterns.
## 6. Rule Pruning:
Rule pruning is a vital step in refining the relevance of the extracted rules. The lift metric, indicating the strength of association between antecedent and consequent, is used as a criterion for pruning. Rules with a lift greater than 1 are retained, ensuring that only the most meaningful associations are considered in the analysis.
## 7. Redefining the Dataset Based on Antecedent and Consequents:
Post pattern mining and rule pruning, the features are redefined as antecedents, laying the groundwork for the modeling phase. This redefinition is crucial in aligning the dataset with the requirements of the chosen analytical models.
## 8. Under sampling:
To counter potential class imbalances, the RandomUnderSampler technique is employed for strategic under sampling. This approach promotes a more balanced class representation in the dataset, which is essential for unbiased model training and evaluation.
## 9. Model Definition:
A diverse array of models, including Random Forest, Logistic Regression, CatBoost, XGBoost, and LightGBM, is deployed to explore various modeling approaches on the preprocessed dataset. This comprehensive selection allows for an in-depth evaluation of different methods and their suitability for the dataset.
## 10. Model Evaluation:
The models undergo rigorous evaluation using the stratified k-fold algorithm. This methodology provides a thorough assessment of model performance across different folds, enhancing the reliability and robustness of the evaluation process. The stratified k-fold approach ensures that each fold is representative of the overall dataset, allowing for a more accurate evaluation of the models’ capabilities.


# Experiment Section
## 1. Research Objectives
This research aims to improve the efficiency of both traditional classification algorithms and their integration with pattern mining techniques. Focusing on accuracy and speed, our experimentation explores synergies between classification and pattern mining, aiming to enhance interpretability and discriminative power. The study addresses diverse data types, seeking to establish a robust framework for contemporary and future data analysis, with practical applications across domains.
## 2. Pattern Mining Exploration
### 2.1 Algorithm Selection
In the initial stages of pattern mining, we employed widely-used algorithms, including FP-max, FrequentSpan, and Top-K optimal patterns. Surprisingly, these algorithms yielded similar results, prompting a deeper exploration of our methodology.
### 2.2 Data Encoding Strategies
An early challenge emerged when encoding data using string representations, leading to a significant increase in dataset dimensionality. To address this issue, we transitioned to the use of Abbreviated String encoding, aiming to strike a balance between representation efficiency and algorithm performance.
## 3. Under sampling Techniques
Initially, we explored under sampling techniques like NearMiss and OneSided. However, due to computational impracticalities, we shifted to the more efficient Randomized Undersampler to maintain dataset integrity while addressing class imbalance.
## 4. Evaluation and Optimization
Our experimentation involved real-world datasets challenging algorithms and under sampling methods. Performance evaluation included standard metrics—F1 score, accuracy, and K-stratified F1 score—providing nuanced insights into algorithmic strengths and weaknesses.

# Results
In our comparative analysis, algorithms trained with frequent mining patterns exhibited superior results. While the F1 score was notably lower without pattern mining, a synergistic approach—integrating classification algorithms with pattern mining—yielded significantly higher F1 scores. Notably, according to Figure 3, CatBoost demonstrated the highest score in Method 1, while XGBoost outperformed others with the highest score in Method 2. This underscores the nuanced performance variations among different algorithms and highlights the importance of tailoring methods based on specific objectives.
In analyzing the results, Method One, depicted in the figure 3, showcased the consistent dominance of CatBoost with the highest Mean F1 Scores, indicating a commendable balance between precision and recall. While achieving high Mean Accuracy Scores, there was room for improvement, suggesting potential enhancements. In contrast, Method Two, represented in the figure 4, exhibited an overall improvement in classifier performance. High F1 Scores and exceptional Mean Accuracy Scores were observed across diverse classifiers, including Random Forest, XGBoost, and LightGBM. The success of Method Two in handling various mental health aspects, as indicated by high performance across all flags, signifies its effectiveness. Comparatively, Method Two surpassed Method One, with advanced techniques such as FP- Growth pattern mining, rule pruning, and undersampling contributing to its superior performance. The results underscore the potential of sophisticated techniques in elevating mental health classifier performance, with significant implications for accurate diagnostics in clinical practice.
