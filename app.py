#import the flask class from the flask module
from flask import Flask, render_template,request
import pickle

#Flask :import the Flask class to create the app.
#render_template:render HTML templates(index.html)
#request:handles incoming request data(like form input)
#pickle:used to load pre-trained models and tokenizers from .pkl files

tokenizer=pickle.load(open("models/cv.pkl","rb"))
model=pickle.load(open("models/clf.pkl","rb"))
#pickle is a built-in python module ,saving a python object to a file and loading it back
#tokenizer is probably a CountVectorizer or TfidfVectorizer used to convert text into numbers
#model is likely a trained classifier (like Logistic Regression, SVM, etc.) for spam detection
# a saved (pickled) python object file

# //creating an instance of the Flask name is refering to current module
app = Flask(__name__)

#Register a route with the Flask app
@app.route('/')
def home():
    # text = ""
    # if request.method == 'POST':
        # Here you would typically handle the form submission
        # For example, you could get the text from the form and process it
        # text = request.form.get('email-content')
        # Process the text (e.g., classify as spam or not spam)
        # For now, we will just print it to the console
        #print(text)
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    # if request.method == 'POST':
        # Get the email content from the form
        # text = request.form.get('email-content')
        # For now, we will just print it to the console
        # print(text)
     email =request.form.get('content')
     #get the input from the form (email-content)
     tokenized_email=tokenizer.transform([email]) #X
    #tokenize the input using the loaded tokenizer (turns text into numerical vector)
     prediction=model.predict(tokenized_email) #y
    #predict using the model(spam or not spam)
     prediction=1 if prediction == 1 else -1
    #converts the result 1->spam anything else -1->(not spam)
     return render_template('index.html', prediction=prediction,email=email)
    #render html template index.html passes two variables to the template :predictions->the result of the spam check(1 or -1)
    # email_text->the original email content entered by the user
#Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)