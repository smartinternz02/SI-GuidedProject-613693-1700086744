from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

class User:
    def __init__(self, email, password):
        self.email = email
        self.password = password

# In-memory list to store registered users
registered_users = []

#Load the tokenizer and model
with open("working/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("working/best_model.keras")

#Set the upload folder
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

app.use_x_sendfile = False

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

max_length = 35

def idx_to_word(integer, tokenizer):
    return tokenizer.index_word.get(integer, None)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break

    return in_text

def generate_caption(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    features = vgg_model.predict(image)

    # Use the model to generate a caption
    caption = predict_caption(model, features, tokenizer, max_length)

    return caption

@app.route("/signin", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        #Get user input from the registration form
        email = request.form.get("email")
        password = request.form.get("password")

        #Basic validation
        if not email or not password:
            return render_template("signin.html", error="Email and password are required.")

        #Check if the email is already registered
        if any(user.email == email for user in registered_users):
            return render_template("signin.html", error="Email already registered.")

        #Create a new user and add it to the list
        new_user = User(email, password)
        registered_users.append(new_user)

        #Redirect to a success page or login page after registration
        return redirect(url_for("index"))

    #Render the registration form template for GET requests
    return render_template("signin.html")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]

        # Check if the file is empty
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Generate caption for the uploaded image
            caption = generate_caption(filepath)

            return render_template("index.html", caption=caption, image_path=filename)

        else:
            return render_template("index.html", error="Invalid file type. Please upload a valid image file.")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

