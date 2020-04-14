# from turtle import Turtle
import turtle
import csv
import numpy as np
# temp imports
# for splitting data to train ans test parts
from sklearn.model_selection import train_test_split
from keras.utils import np_utils  # neaiskus ar reikia
from keras.models import load_model
from keras.models import Sequential  # training model
from keras.layers import Dense, Dropout  # activation functions in layers
from keras.wrappers.scikit_learn import KerasClassifier
# decoding strings to categorias "one hot encoding"
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from pickle import load
from pickle import dump

# TODO sutvarkyti importus , ne visu gali reiketi


def draw_grid(game):
    grid = turtle.Turtle()
    lenght = game.grid_size * game.move_step
    for i in range(0, lenght + game.move_step, game.move_step):
        grid.penup()
        # go to start hoz line possition
        grid.setpos(-lenght/2, (lenght/2 - i))
        grid.pendown()
        grid.setpos(lenght/2, (lenght/2 - i))  # draw horizontal line
        grid.penup()
        # go to start vertical line possition
        grid.setpos((-lenght/2 + i), lenght/2)
        grid.pendown()
        grid.setpos((-lenght/2 + i), -lenght/2)
    grid.hideturtle()

# -------------------------Matrix of cords-----------------------------------------------


def create_cords_map(g):
    '''
    0,3     3,3
    _________
    |__|__|__|
    |__|__|__|
    |__|__|__|
    0,0     3,0
    '''
  # create matrix map with x and y coords ex: (0,0) = [-60,-60] this is bottom left corner
    cords_map = {}
    for key_x in range(g.max_grid_size):  # from 1 to 5 if size 3x3
        for key_y in range(g.grid_size + 2):  # the same
            x = g.move_step * key_x + g.first_cell_pos_x
            y = g.move_step * key_y + g.first_cell_pos_y
            cords_map[(key_x, key_y)] = [x, y]
    return cords_map

# ----------------------------------------------------------------------------------------

# update score
    def updateScore(snake):
        f_score = turtle.Turtle()
        f_score.setpos(0, 300)
        f_score.penup()
        f_score.color('black')
        style = ('Courier', 30, 'italic')
        f_score.hideturtle()
        f_score.write(f'Found: {snake.food_count}', font=style, align='center')


def write(xy, text, color="black"):
    turtle.color(color)
    turtle.hideturtle()
    turtle.penup()
    turtle.setpos(xy)
    style = ('Courier', 12, 'bold')
    turtle.write(text, font=style, align='center')


def placeText(text, xy, size=12, color="black"):

    win_text = turtle.Turtle()
    win_text.color(color)
    win_text.hideturtle()
    win_text.penup()
    win_text.setpos(xy)
    style = ('Courier', size, 'bold')
    win_text.write(text, font=style, align='center')


def placeButton(shape, color, xy, shapesize=1.5):

    # draw button
    btn_record = turtle.Turtle()
    btn_record.penup()
    btn_record.goto(xy)
    btn_record.shape(shape)
    btn_record.color(color)
    btn_record.shapesize(shapesize)


def logRecordToFile(data):

    # print(data)
    # print(len(data))

    # data with snake inputs 27 about
    mydict = [data]

    # field names
    fields = ['Right_Food',
              'Right_Body',
              'Right_Wall',
              'RightDown_Food',
              'RightDown_Body',
              'RightDown_Wall',
              'Down_Food',
              'Down_Body',
              'Down_Wall',
              'LeftDown_Food',
              'LeftDown_Body',
              'LeftDown_Wall',
              'Left_Food',
              'Left_Body',
              'Left_Wall',
              'UpLeft_Food',
              'UpLeft_Body',
              'UpLeft_Wall',
              'Up_Food',
              'Up_Body',
              'Up_Wall',
              'UpRight_Food',
              'UpRight_Body',
              'UpRight_Wall',
              'Dist_to_Food',
              'Body_Lenght',
              'Direction'
              ]

    # write headers
    # name of csv file
    filename = "./logs/headers_aisnake_records_log.csv"

    # writing to csv file headers
    with open(filename, 'w', newline='') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

    # write rows
    # name of csv file
    filename = "./logs/inputs_aisnake.csv"

    # writing to csv file headers
    with open(filename, 'a', newline='') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing data rows
        writer.writerows(mydict)


def loadTrainedModel(game):

    try:

        # del model

        # load saved  model
        game.train_model = load_model("deep_learning_models/best_weights.hdf5")
        # model = load(open('deep_learning_models/model.pkl', 'rb'))
        # model = load_model("deep_learning_models/model.h5")
        # https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/

        # load saved scaler
        game.train_scaler = load(open('deep_learning_models/scaler.pkl', 'rb'))

        print('Trained model and scaler sucefuly loaded.')

    except:
        print('Error acured when loading Trained model and scaler!')

    return


def trainSnake():

    # load dataset using pandas
    dataframe = pd.read_csv(
        "logs\inputs_aisnake.csv", header=None)
    dataset = dataframe.values

    # split into input and output columns
    X = dataset[:, :-1].astype(float)
    Y = dataset[:, -1]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    # Normalizing the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # split train and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, dummy_y, test_size=0.25, random_state=42)

    print(f"Total: {len(X)}\nX train: {len(X_train)}, Y train: {len(Y_train)}\nX test:  {len(X_test)}, Y test:  {len(Y_test)}")

    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=26, activation='relu',
                    kernel_initializer='random_normal'))
    # model.add(Dropout(0.5))
    # model.add(Dense(25, activation='relu', kernel_initializer='random_normal'))

    model.add(Dense(4, activation='softmax',
                    kernel_initializer='random_normal'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # simple early stopping for overfitting eliminate
    # stop learning if not inprufed by patience step and save best weights and model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=6)
    mc = ModelCheckpoint(filepath='deep_learning_models/best_weights.hdf5',
                         monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

    # model fit
    history = model.fit(X_train, Y_train, validation_data=(
        X_test, Y_test), epochs=1000, batch_size=1000, verbose=0, callbacks=[es, mc])

    # evaluate the model
    _, train_acc = model.evaluate(X_train, Y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Train acc: %.2f%%, Test acc: %.2f%%' %
          (train_acc*100, test_acc*100))

    # # save model and architecture to single file
    # model.save("deep_learning_models/model.h5")
    # # save the model
    # dump(model, open('deep_learning_models/model.pkl', 'wb'))

    # save the scaler
    dump(scaler, open('deep_learning_models/scaler.pkl', 'wb'))
    print("Scaler and model saved to disk")

    # plot training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def checkTrainedModel():

    # load saved  model
    model = load_model("deep_learning_models/best_weights.hdf5")
    # model = load(open('deep_learning_models/model.pkl', 'rb'))
    # model = load_model("deep_learning_models/model.h5")
    # https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/

    # load saved scaler
    scaler = load(open('deep_learning_models/scaler.pkl', 'rb'))

    # load dataset using pandas
    dataframe = pd.read_csv("logs\inputs_aisnake.csv", header=None)
    dataset = dataframe.values

    # split into input and output columns
    X = dataset[:, :-1].astype(float)
    Y = dataset[:, -1]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    # Normalizing the data
    # scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # split train and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, dummy_y, test_size=0.25, random_state=42)

    print(f"Total: {len(X)}\nX train: {len(X_train)}, Y train: {len(Y_train)}\nX test:  {len(X_test)}, Y test:  {len(Y_test)}")

    # Make predictions
    predictions = model.predict(X_test)  # model.predict_proba(X_test[:6,:])
    predictions = predictions.argmax(1)
    expected = Y_test.argmax(1)  # convert from one hot encoded to integers

    # # summarize the first 6 cases
    for i in range(6):
        print(f'{predictions[i]} expected {expected[i]} ')

    # summarize model.
    model.summary()

    # evaluate the model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    # Compute confusion matrix
    results = confusion_matrix(expected, predictions)
    print(results)
    outputs = ['Right', 'Down', 'Left', 'Up']
    df_cm = pd.DataFrame(results, index=[i for i in outputs],
                         columns=[i for i in outputs])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def predict(inputs, game):
    # print(inputs)

    # # load saved  model
    model = game.train_model
    # # load saved scaler
    scaler = game.train_scaler

    # model = load_model("deep_learning_models/best_weights.hdf5")
    # # model = load(open('deep_learning_models/model.pkl', 'rb'))
    # # model = load_model("deep_learning_models/model.h5")
    # # https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/

    # scaler = load(open('deep_learning_models/scaler.pkl', 'rb'))

    # load dataset using pandas
    dataframe = pd.DataFrame([inputs.values()])
    dataset = dataframe.values

    # split into input and output columns
    X = dataset[:, :-1].astype(float)
    Y = dataset[:, -1]

    # Normalizing the data
    X = scaler.transform(X)

    # Make prediction
    pred = model.predict(X)
    pred = pred.argmax(1)[0]

    # print(prediction)
    dir_names = {
        0: 'Down',
        1: 'Left',
        2: 'Right',
        3: 'Up',
    }
    prediction = dir_names[pred]

    # print(f'My prediction: {prediction}')
    return prediction
