# KNN 
Summary of KNN
K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple and versatile supervised learning algorithm used for classification and regression. It works based on the idea of similarity: predictions are made by looking at the data points that are closest to a new data point.
How KNN Works

    Store the Training Data:
        KNN does not create a model in the training phase. Instead, it simply stores the dataset.

    Calculate Distance:
        When a new data point needs a prediction, KNN calculates its distance from all points in the training data using a distance metric (e.g., Euclidean distance, Manhattan distance).

    Select Nearest Neighbors:
        KNN selects the k closest data points to the new data point.

    Make a Prediction:
        Classification: Assigns the class of the majority of the k neighbors.
        Regression: Takes the average of the target values of the k neighbors.

Key Concepts

    k Value:
        Determines the number of neighbors to consider.
        A small k (e.g., 1) might be sensitive to noise, while a large k might generalize too much.
        Typically, k is chosen as an odd number to avoid ties in classification.

    Distance Metrics:
       Common Distance Metrics in KNN (Without Formulas)

    Euclidean Distance: Measures straight-line distance between two points.
    Manhattan Distance: Measures distance as the sum of absolute differences along each axis.
    Minkowski Distance: A generalized distance metric that can adapt to Euclidean or Manhattan based on a parameter.
    Cosine Similarity: Measures the angular similarity between two vectors.
    Hamming Distance: Counts differences between categorical values, often used for text or binary data.
    Chebyshev Distance: Measures the maximum absolute difference between dimensions
Advantages

    Simple to Understand: No assumptions about data distribution.
    Versatile: Works for both classification and regression.
    No Training Phase: Stores the data directly.

Disadvantages

    Computational Cost: Predictions require distance calculation for all training points, which can be slow for large datasets.
    Sensitive to Irrelevant Features: Features that do not impact the outcome can reduce accuracy unless normalized or removed.
    Data Scaling Required: Since it relies on distances, features with larger ranges dominate others.

Applications of KNN

    Recommendation Systems: Find similar items or users (e.g., movie or product recommendations).
    Medical Diagnosis: Predict diseases based on symptoms (e.g., diabetes or cancer prediction).
    Pattern Recognition: Handwriting, image, or speech recognition.

Example

For a classification problem:

    Dataset: Points in a 2D space labeled as "Cat" or "Dog".
    New Point: You want to classify a new point.
    KNN Process:
        Calculate distances to all points.
        Select k nearest points.
        Take a majority vote of their labels to classify the new point.

KNN is powerful for intuitive tasks but needs careful parameter tuning and preprocessing for optimal performance.