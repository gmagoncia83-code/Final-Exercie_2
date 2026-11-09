# SAD Mini-Analysis and Reflection

## 1. Problem statement
A wine producer needs a consistent preliminary method to screen wine samples using recorded physicochemical measurements.

## 2. Users / stakeholders
- Quality-control staff
- Production supervisor
- Data analyst

## 3. Inputs
Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.

## 4. Process
Validate required input columns → preprocess numerical data → apply trained classifier → classify the wine sample → display and export the result.

## 5. Outputs
- GOOD QUALITY
- NEEDS IMPROVEMENT
- Model evaluation metrics and downloadable prediction results

## 6. Key limitation
The model is trained on a particular public red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions.

## Reflection answers

### 1. What is the prediction target, and why is the original quality column excluded from input features?
The target is a binary class derived from the original score: `quality >= 6` is GOOD QUALITY and `quality < 6` is NEEDS IMPROVEMENT. The original `quality` column is excluded from input features because it directly determines the target. Including it would cause target leakage and make evaluation misleading.

### 2. What does recall for the GOOD QUALITY class tell you? What could low recall mean for the business?
Recall for GOOD QUALITY measures the proportion of actual good-quality samples that the model correctly identifies as good. Low recall means some good samples are incorrectly classified as needing improvement, which could lead to unnecessary rechecks or delays. Recall should be interpreted together with precision and other metrics.

### 3. Why might a model trained on this red-wine dataset perform differently on another producer's wine samples?
Other producers may use different grape varieties, processes, regions, or measurement conditions. These differences can change the distribution of input measurements, so the model may not generalize well.

### 4. Why should the prediction be treated as preliminary screening rather than a final quality decision?
The dataset and model are limited, and the prediction is only an estimate. Final quality decisions require laboratory testing, sensory evaluation, food-safety procedures, and qualified professional judgment.

### 5. What additional data or testing would help improve reliability?
More samples from different producers, regions, batches, and time periods; consistent laboratory measurement procedures; independent validation data; monitoring of model performance; and review by quality-control specialists would help assess and improve reliability.

## Responsible-use note
This is a classroom prototype based on public data. Predictions are decision-support information only and must not replace laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
