Project to Convert Audio File to Text using Speech API and Derive Sentiment Score.
Bi-Directional LSTM model is trained on IMDB data to predict the text as positive or negative and score for it
The trained model is saved locally and is used for predictions 

This complete project can be setup to run in your local by following below steps,
    1) Copy the complete folder
    2) Update the config.yml file to use 'local' as data source or use 'cloud' as source. For cloud source you need to specify containe-name, storage account and access keys
    3) For using Audio Conversion , setup the Speech recognition API key in config file
    4) Install the dependencies using requirements.txt
    5) Download the nltk data
    6) Setup Spacy using, python3 -m spacy download en
    7) Setup flask and run the server.py file to test the services as API
    8) Application.py file can be used to check from UI, but this has not been modified for topic modeling predictions
    9) Build Model script has not been exposed as it contains some sensitive information. 
    You can build your own module to build a topic modeling and clustering data
    
  
