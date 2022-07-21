# [AI Competition for Crop Disease Diagnosis due to Changes in Agricultural Environment](https://dacon.io/competitions/official/235870/overview/description)  
## Goal  
Develop an AI model that diagnoses "type of crop", "type of disease" and "progress of disease" using "crop environment data" and "crop disease image"

## Sample Image
![download-1](https://user-images.githubusercontent.com/87674297/166981077-abd0596f-2876-4f39-94b7-66ffaa2a6d35.png)

## Points of caution  
1. Class imbalance.
2. When using Kfold, valid loss varies up to 0.1 for each fold.  
3. It can be seen that a model with good generalization is required.  

## Solution  
1. Use 3Kfold.
2. Try various augmentation, such as cutmix.
3. Use FocalLoss, and Labelsmoothing.  
4. Use cropped images. (It didn't perform well.)  
5. Ensemble different models(ResNet, EfficientNet_B5, Resnext50_32x4d, Densenet)  

## Model
Model finetunning : feature extractor freeze 40 epoch + unfreeze 30 epoch  

| Model           | ImageSize | BatchSize | LearningRate | Optimizer |LeaderBoard |
|-----------------|-----------|-----------|--------------|-------------|-------------|
| ResNet           | 520       | 32        | 3e-4         | AdamW        |0.892        |
| EfficientNet_B5    | 480       | 32        | 4e-3         | AdamW         |0.924        |
| Resnext50_32x4d | 520       | 64        | 3e-4         |AdamW        |0.921        |
| Above models ensemble |        |         |          |         |0.936        |


## Team Model Ensemble
0.94

## Final Result
Top 8%(29/344)

## What I learned through the competition.
1. After a certain epoch, it is good for model performance not to use techniques such as cutmix.
2. Sufficient EDA for images is needed.
3. Need to study lstm and rnn.
