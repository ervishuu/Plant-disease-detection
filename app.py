from flask import Flask, render_template, request, send_from_directory
import cv2
import pickle
import joblib
import numpy as np
from sklearn.svm import SVC
from keras.models import load_model

#load model
model_corn =load_model("AG_Corn_Plant_VGG19 .h5")
model_cotton =load_model("AG_COTTON_plant_VGG19.h5")
model_grape= load_model("AI_Grape.h5")
model_potato= load_model("AI_Potato_VGG19.h5")
model_tomato= load_model("AI_Tomato_model_inception.h5")


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/leaf_detection')
def leaf_detection():
    return render_template('leaf_detection.html')

@app.route('/inputcotton')
def inputcotton():
    return render_template('prediction_cotton.html')


@app.route('/inputcorn')
def inputcorn():
    return render_template('prediction_Corn.html')

@app.route('/inputgrape')
def inputgrape():
    return render_template('prediction_Grape.html')

@app.route('/inputpotato')
def inputpotato():
    return render_template('prediction_potato.html')

@app.route('/inputtomato')
def inputtomato():
    return render_template('prediction_tomato.html')

@app.route('/input_crop_recommendation')
def input_crop_recommendation():
    return render_template('crop_recomdation.html')



@app.route('/data' , methods = ['POST','GET'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        phone = int(request.form['phone'])
        email = request.form['email']
        subject =request.form['subject']
        message =request.form['message']

        print("Name Of User:",name)
        print("Phone no:",phone)
        print("Email:",email)
        print("subject:",subject)
        print("message:",message)

        return render_template('index.html')
    
    else :
        return render_template('index.html')


@app.route('/predictioncotton',methods = ['POST'])
def predictioncotton():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_cotton.predict(img_arr)
    prediction=np.argmax(predictions, axis=1)
    print(prediction[0])
    #
    # x = round(prediction[0])
    # # y = round(prediction[0, 1], 2)
    # preds = np.array([x])
    COUNT += 1
    if prediction[0] == 0:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["diseased cotton leaf", 'green'])
    elif prediction[0] == 1:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["diseased cotton plant", 'red'])
    elif prediction[0] == 2:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["fresh cotton leaf", 'red'])
    else:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["fresh cotton plant", 'red'])


@app.route('/predictioncorn', methods=['POST'])
def predictioncorn():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_corn.predict(img_arr)
    prediction = np.argmax(predictions, axis=1)
    print(prediction[0])

    COUNT += 1
    if prediction[0] == 0:
        return render_template('Output.html', data=["Blight"])
    elif prediction[0] == 1:
        return render_template('Output.html', data=["Common_Rust"])
    elif prediction[0] == 2:
        return render_template('Output.html', data=["Gray_Leaf_Spot"])
    else:
        return render_template('Output.html', data=["Healthy"])



@app.route('/predictiongrape',methods = ['POST'])
def predictiongrape():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_grape.predict(img_arr)
    prediction=np.argmax(predictions, axis=1)
    print(prediction[0])
    #
    # x = round(prediction[0])
    # # y = round(prediction[0, 1], 2)
    # preds = np.array([x])
    COUNT += 1
    if prediction[0] == 0:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___Black_rot'", 'green'])
    elif prediction[0] == 1:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___Esca_(Black_Measles)", 'red'])
    elif prediction[0] == 2:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 'red'])
    else:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___healthy", 'red'])






@app.route('/predictionpotato',methods = ['POST'])
def predictionpotato():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_potato.predict(img_arr)
    prediction=np.argmax(predictions, axis=1)
    print(prediction[0])
    #
    # x = round(prediction[0])
    # # y = round(prediction[0, 1], 2)
    # preds = np.array([x])
    COUNT += 1
    if prediction[0] == 0:

        return render_template('Output.html', data=["Potato_Early_blight", 'red'])
    elif prediction[0] == 1:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Potato_Late_blight", 'red'])

    else:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Potato_healthy ", 'red'])




@app.route('/predictiontomato', methods=['POST'])
def predictiontomato():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_tomato.predict(img_arr)
    prediction = np.argmax(predictions, axis=1)
    print(prediction[0])

    COUNT += 1
    if prediction[0] == 0:
        return render_template('Output.html', data=["Bacterial_spot"])
    elif prediction[0] == 1:
        return render_template('Output.html', data=["Early_blight"])
    elif prediction[0] == 2:
        return render_template('Output.html', data=["Late_blight"])
    elif prediction[0] == 3:
        return render_template('Output.html', data=["Leaf_Mold"])
    elif prediction[0] == 4:
        return render_template('Output.html', data=["Septoria_leaf_spot"])
    elif prediction[0] == 5:
        return render_template('Output.html', data=["Spider_mites Two-spotted_spider_mite"])
    elif prediction[0] == 6:
        return render_template('Output.html', data=["Target_Spot"])
    elif prediction[0] == 7:
        return render_template('Output.html', data=["Tomato_Yellow_Leaf_Curl_Virus"])
    elif prediction[0] == 8:
        return render_template('Output.html', data=["Tomato_mosaic_virus"])
    else:
        return render_template('Output.html', data=["Healthy"])



@app.route('/crop_recommendation' , methods = ['POST','GET'])
def crop_recommendation():
    if request.method == 'POST':
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        temperature =float(request.form['temperature'])
        humidity =float(request.form['humidity'])
        rainfall =float(request.form['rainfall'])
        ph =float(request.form['ph'])
        # State =request.form['State']
        print(Nitrogen,Phosphorus,Potassium,temperature,humidity,rainfall,ph)

        # Load the Model back from file
        with open("Crop_Recomandation_RF.pkl", 'rb') as file:
            Pickled_RF_Model = pickle.load(file)
        result = Pickled_RF_Model.predict([[Nitrogen,Phosphorus,Potassium,temperature,humidity,ph,rainfall]])
        if result[0] == 20:
            return render_template('crop_recomdation.html', data=["rice",'green'])
        elif result[0] == 11:
            return render_template('crop_recomdation.html', data=["maize",'green'])
        elif result[0] == 3:
            return render_template('crop_recomdation.html', data=["chickpea",'green'])
        elif result[0] == 9:
            return render_template('crop_recomdation.html', data=["kidneybeans",'green'])
        elif result[0] == 18:
            return render_template('crop_recomdation.html', data=["pigeonpeas",'green'])
        elif result[0] == 13:
            return render_template('crop_recomdation.html', data=["mothbeans",'green'])
        elif result[0] == 14:
            return render_template('crop_recomdation.html', data=["mungbean",'green'])
        elif result[0] == 2:
            return render_template('crop_recomdation.html', data=["blackgram",'green'])
        elif result[0] == 10:
            return render_template('crop_recomdation.html', data=["lentil",'green'])
        elif result[0] == 19:
            return render_template('crop_recomdation.html', data=["pomegranate",'green'])
        elif result[0] == 1:
            return render_template('crop_recomdation.html', data=["banana",'green'])
        elif result[0] == 12:
            return render_template('crop_recomdation.html', data=["mango",'green'])
        elif result[0] == 7:
            return render_template('crop_recomdation.html', data=["grapes",'green'])
        elif result[0] == 21:
            return render_template('crop_recomdation.html', data=["watermelon",'green'])
        elif result[0] == 15:
            return render_template('crop_recomdation.html', data=["muskmelon",'green'])
        elif result[0] == 0:
            return render_template('crop_recomdation.html', data=["apple",'green'])
        elif result[0] == 16:
            return render_template('crop_recomdation.html', data=["orange",'green'])
        elif result[0] == 17:
            return render_template('crop_recomdation.html', data=["papaya",'green'])
        elif result[0] == 4:
            return render_template('crop_recomdation.html', data=["coconut",'green'])
        elif result[0] == 6:
            return render_template('crop_recomdation.html', data=["cotton",'green'])
        elif result[0] == 8:
            return render_template('crop_recomdation.html', data=["jute",'green'])

        else:
            return render_template('crop_recomdation.html', data=['coffee','green'])


    else :
        return render_template('crop_recomdation.html')


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static/img', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)

