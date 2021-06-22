# Neural_Network_Charity_Analysis
Neural networks (also known as artificial neural networks, or ANN) are a set of algorithms that are modeled after the human brain. They are an advanced form of machine learning that recognizes patterns and features in input data and provides a clear quantitative output. In its simplest form, a neural network contains layers of neurons, which perform individual computations. These computations are connected and weighed against one another until the neurons reach the final layer, which returns a numerical result, or an encoded categorical result.

## Project Overview
The purpose of this analysis is to use a neural network to decide which companies should recieve loans from Alphabet Soup Charity Neural Network Analysis. This analysis uses python's TensorFlow library to create, train, and evaluate data gathered from previous loans.

## Tools
Python, Pandas, Matplotlib, Scikit-learn, TensorFlow

## Results

### Data Preprocessing
- The data was imported, analyzed, and cleaned.

![charity_data](./Resources/charity_data.png)

- After looking at the data, I established that the target variable is the "IS_SUCCESSFUL" column. I then removed the "EIN" and "NAME" columns as they did not offer any relevant data that could help the model perform better. The remaining columns became the features for the model.

![cleaned_data](./Resources/cleaned_data.png)

- Further, data was preprocessed (bucketed, encoded and scaled) before fitting into the deep learning model. 

## Compiling, Training, and Evaluating the Model
- For my first attempt at compiling a neuron network consisted of 80 neurons in the first layer and 30 in the second. 
- Both layers had relu activation functions and the output layer had a sigmoid activation function. 
- I started with these parameters as relu does better with nonlinear data, and two layers allows for a second layer to reweight the inputs from the first layer. 

![compile_trained_model1](./Resources/compile_trained_model1.png)

### Here are the preformance metrics of this model.

![model1_loss_accuracy](./Resources/model1_loss_accuracy.png)

- In my second attempt, I removed the "SPECIAL_CONSIDERATIONS" and "STATUS" column as I thought this might have been confusing the model. 
- I also lowered the threshold for the classification column so that there were more unique values from that column. 

![compile_trained_model2](./Resources/compile_trained_model2.png)

### Here are the preformance metrics of this model.

![model2_loss_accuracy](./Resources/model2_loss_accuracy.png)

- In my third attempt, I reverted back the threshold for the classification column. 
- I also added a third layer with neurons apart of it. B
- By adding a third layer, I wanted to give the model another chance to reweight the inputs from the second layer to the third.
- I also changed the activation function for the three layers to tanh. 
- I did this to see if it would perform better than the relu function. 

![compile_trained_model3](./Resources/compile_trained_model3.png)

### Here are the preformance metrics of this model.

![model3_loss_accuracy](./Resources/model3_loss_accuracy.png)


## Summary
After three attempts, I was unable to create a model that could preform a 75% accuracy rating. This is potential becasue I got rid of too many columns, I did not use the correct activation function, or I did not have hte right amount of layers and neurons. These were the main areas I continued the change with little to no improvement. Next time, I would research more about activation functions to make sure that I am always choosing the right one based on the data.

### Suggestion
- Random forest classifiers can be used to solve this classification problem. 
- The random forest classifier is capable to train on the large dataset and predict values in seconds. 
- It is also capable to achieve comparable predictive accuracy on large tabular data with less code and faster performance.
