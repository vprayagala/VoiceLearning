Project to Convert Audio File to Text using Speech API and Derive Sentiment Score.
Bi-Directional LSTM model is trained on IMDB data to predict the text as positive or negative and score for it
The trained model is saved locally and is used for predictions 

The complete source code has been modularized using python packages. The data handling is specific to this project, not generalized. The audio files are converted and saved to vtt files for analysis. The vtt files are the source files for training. 

Services have been exposed using flask web framework. Using the audio conversion needs cloud credentials.
