## Dataset: 
- Prepared from CT scans of **42 patients** of which 21 patients were operated on. Thus, Imbalanced classes are not present.

- The task at hand is a **binary classification problem** to determine whether patient is operated on(1) or not(0).

- Flattened the CT scan videos (raw data) to dataset consisting of 2,343 slices.

## resnet_model_1
<!-- Transfer learning -->
Train acc: 0.8958  |  Val acc: **0.5560**
- Resnet50 Architecture with imagenet weights
- Replaced dense output layer of 1000 classes with a dense layer of 2 classes.
- Only made Fully connected layers trainable (4,098 trainable parameters).

![accuracy vs number of epochs](/media/resnet_model_1.png)

### Observations
- Better than dummy classifier.
- There was no change in validation accuracy over multiple epochs. I tried different learning rates using adam optimizer there was no change in results. 
- Switching to SGD optimizer increased val acc to 0.5800. But soon this also became constant. This constant acc issue could be due to vanishing gradient.
- Model is also overfitting. Possible solution for above problems is to increase data size.

## resnet_model_2
<!-- DropOut/ Finetuning resnet50 -->
Train acc: 0.59  |  Val acc: **0.5250**
- Resnet50 Architecture with imagenet weights
- Replaced fully connected layer of resnet50 with a dense(512), dropout(0.5), dense(256), dropout(0.5), output dense(2)
- Only made Fully connected layers trainable (1,180,930 trainable parameters)

![accuracy vs number of epochs](/media/resnet_model_2.png)

### Observations
- More dropout layers were initially added to prevent overfitting. But I also increased the number of trainable parameters by adding dense layers defeating the purpose of it. 
- I tried different DropOut rates from 0.5-0.8, the acc dropped even further. Overall, not insightful.

## resnet_model_3 
<!-- Training Convolutional layers of resnet50 -->
Train acc: 0.9852  |  Val acc: **0.444**
- I unfroze the parameters for last few convolutional layers in above 2 models and trained them. Only further drop in validation accuracy occured with even higher overfitting. Not insightful

![accuracy vs number of epochs](/media/resnet_model_3.png)

---
---
In the next three models image augmentation has been performed using *ImageDataGenerator* imported from *keras.preprocessing.image*. 

Transformations include horizontal/vertical shifts, upto 45 degree rotations and horizontal flips.

## augmented_resnet_model_1
<!-- First Image augmentation model -->
Train acc: 0.6033  |  Val acc: **0.5960**  
(epoch - 6/10)
- Resnet50 Architecture with imagenet weights
- Replaced dense output layer of 1000 classes with a dense layer of 2 classes.
- Added a dropout(0.5) layer between flatten(2048) and dense output layer.
- Only made Fully connected layers trainable (4,098 trainable parameters).

![accuracy vs number of epochs](/media/augmented_model_1_acc.png)

### Observations
- Observed huge increase in accuracy compared to unaugmented models. The increase was also stable i.e. it was observed in many epochs and val acc was always above 0.52 . 
- Model was not grossly overfitting.
- But, the train acc/ val_acc was not increasing even after running it for 20 epochs. Train_acc was always below 0.68 and val_acc was always below 0.60.
- The hypothesis made was that the model with only 4098 trainable parameters wasn't complex enough to learn more features of the training dataset. Hence, next model introduces another dense layer.

## augmented_resnet_model_2
<!-- Added a dense layer for more model complexity -->
Train acc: 0.7268  |  Val acc: **0.6418**  
(epoch - 4/10)
- Everything same as augmented_resnet_model_1 but also added a dense layer(512) in between the flatten and dropout layer.
- Only made Fully connected layers trainable (1,050,114 trainable parameters).

![accuracy vs number of epochs](/media/augmented_model_2_preloaded_acc.png)
![loss vs number of epochs](/media/augmented_model_2_preloaded_loss.png)

### Observations
- Some improvement to acc was observed. However, the acc vs epochs and loss vs epochs graphs were fluctuating . 
- The model was retrained for 30 epochs, and the **size of validation set was increased from 10% to 20%** to remove the noisy behaviour. The fluctuations were still present. And once again the train acc/ val_acc was not increasing after many epochs.
- **Hypothesis:** This might be due to use of ImageDataGenerator, which generates unique images for every batch i.e. the model does not see the same image twice in training or,
- The model was not able to learn the CT representation by training only dense layers. So in the next model, last few convolutional layers were unfrozen.

## augmented_resnet_model_3
<!-- Unfroze last 3 convolutional layers -->
Train acc: 0.9413  |  Val acc: **0.7505**
(epoch - 18/20)
- Same architecture as augmented_resnet_model_2
- Used weights of above model.
- retrained last 15 layers of model of which 3 were conv layers (5,515,778 trainable parameters).

![accuracy vs number of epochs](/media/augmented_model_3_acc.png)

![loss vs number of epochs](/media/augmented_model_3_loss.png)

### Observations
- Observed huge increase in accuracy. 
- The model is now overfitting. But, the issue of stagnant train_acc was not encountered like in the previous augmented_resnet models. 
- This was due to unfreezing the convolutional layers which atleast enabled the model to learn the training data properly.
- Training the augmented_resnet_model_2 first and then using its weights for model_3 was necessary. As mentioned in a Keras blog entry,
> in order to perform fine-tuning, all layers should start with properly trained weights: for instance you should not slap a randomly initialized fully-connected network on top of a pre-trained convolutional base. This is because the large gradient updates triggered by the randomly initialized weights would wreck the learned weights in the convolutional base. In our case this is why we first train the top-level classifier, and only then start fine-tuning convolutional weights alongside it.
---
## custom_model_1
Train acc: 0.5411  |  Val acc: **0.5650**
- Model: 5x5, max pool, 3x3, mp, 3x3,mp, 3x3, mp, fc

![accuracy vs number of epoch](/media/custom_model_1_acc.png)

![loss vs number of epochs](/media/custom_model_1_loss.png)

### Observations
- More testing needs to be done on this simple architecture. 
- The weights are initialized as 'uniform_random'. Other weight initializations need to be tried. As the model is getting stuck in local minimas.
- Performed Image augmentation on this custom model, the acc didn't improve by much.
  
![accuracy vs number of epoch](/media/custom_model_1_aug_acc.png)

---
### Sidenote
- Verified that : http://www.via.cornell.edu/databases/, http://www.ctisus.com, don't have CT scans for the brain.
- Extracted the data from http://headctstudy.qure.ai. Total size : 26.6 GB. I haven't explored it yet.
