# Microsoft Champions League Hackathon

# A Deep learning based Covid-19 Cough Detector

Dataset used : https://github.com/iiscleap/Coswara-Data 

Each folder contains metadata and recordings corresponding to a person. The audio recordings are in wav format (44.1KHz).

Voice samples collected include breathing sounds (fast and slow), cough sounds (deep and shallow), phonation of sustained vowels (/a/ as in made, /i/,/o/), and counting numbers at slow and fast pace. Metadata information collected includes the participant's age, gender, location (country, state/ province), current health status (healthy/ exposed/ cured/ infected) and the presence of comorbidities (pre-existing medical conditions).

# Prerequisites

Make sure that you have the following:

* Python 3.7 and pip (which comes with Python 3+)
* All the mention libraries in Requirements.Txt File.

# Running the App

What each file does:
* ```main.py``` - runs the server and loads the user interface.
* ```templates/layout.html``` - contains the base HTML
* ```templates/index.html``` - contains the body of the HTML
* ```templates/result.html``` - contains the final result of the HTML

To run the app, complete the following steps:

Make sure you have Python 3.7 and a text editor installed.
Clone the repo using ```git clone https://github.com/vivektalwar13071999/Covid_Cough_Detector.git```
Change the directory to ```Covid_Cough_Detector where it is downloaded``` & saved ```cd filepath/Covid_Cough_Detector```
Install the required packages using ```pip install -r requirements.txt```.You can manually install them as you come across them if need be, but this will install them all for you. Note that if you add more packages, run pip freeze > requirements.txt to save them to your requirements file.
In the main directory  Run ```python main.py``` . 
A deployment server will start on local host ```127.0.0.1:5000``` and go to the web-browser and type this ```http://127.0.0.1:5000```
A screen with Covid Cough Detector and a Upload option will be given.
Upload a ```cough.wav``` file from the computer and click on predict the result will appear on the screen in few moments. 
