# Deep-Learning-FDD

This github repository is part of research where data is protected by the project licence, so the data isn't given. Notheless I wanted to showcase the development of the models I used in research:
- Improving the Efficiency of Fan Coil Units in Hotel Buildings through Deep-Learning-Based Fault Detection 2023
https://www.mdpi.com/1424-8220/23/15/6717
- A Review of Data-Driven Approaches and Techniques for Fault Detection and Diagnosis in HVAC Systems 2022
https://www.mdpi.com/1424-8220/23/1/1


## Use case problem :
-Heating, ventilation, and air conditioning (HVAC) systems are essential for maintaining a comfortable indoor environment in modern buildings. However, HVAC systems are known to consume a lot of energy, which can account for up to 50% of a building's energy consumption. Therefore, it is important to detect and troubleshoot problems in HVAC systems timely. Fault detection and diagnosis (FDD) techniques can help with HVAC monitoring and optimizing system performance for efficient use of energy.

## Data:
- Example of few features in the used data:
![Data_features](https://github.com/IvaMate/Fault-Detection-and-Diagnosis-with-DL-and-ML-algorithms/assets/55032190/4e504cd4-a89c-4ae6-afb5-ab3e411fa792)

## Technology:
The development of the predictive models and the experimental setup was done in Python using Pandas library for data cleaning and preprocessing, Scikit-learn library for Random Forests model and Pytorch model for development of three DL models. 

## Models :
Models used are traditional machine learning model - Random Forests model and deep learning models such as: 1D Convolutional Neural Network, 1D Convolutional Neural Network+Gated Recurrent Network, Long-Short-Term Memory Network.
Architecture of DL models is presented:
![Models](https://github.com/IvaMate/Fault-Detection-and-Diagnosis-with-DL-and-ML-algorithms/assets/55032190/41b2dc8d-d7cd-4781-83df-38ef25d02ae4)

## Results :
Here are some results that show comparison of all mentioned models where the hybrid model (CNN+GRU) has the best detection.

![ex2](https://github.com/IvaMate/Fault-Detection-and-Diagnosis-with-DL-and-ML-algorithms/assets/55032190/74049a47-aba9-426b-94ef-d0a9da460687)
![ex1](https://github.com/IvaMate/Fault-Detection-and-Diagnosis-with-DL-and-ML-algorithms/assets/55032190/63fd3514-97fa-423a-900a-4ce79ce97e17)

## Implementation into production:
- anomaly.py, requirements.txt and models in 'Model' folder should be implemented into the HVAC system software. 
