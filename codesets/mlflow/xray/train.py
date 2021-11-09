import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import sys
import shutil
import joblib
import pickle
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from os import mkdir
from mlflow import  tensorflow as mtf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras.models import Sequential,load_model,save_model, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,deserialize, serialize
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau, Callback
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.saving import saving_utils
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report,confusion_matrix
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient




def messages(msg):
    message = print(msg)
    return message

#Define Constants
train_path = 'data/train'
test_path = 'data/test'
valid_path = 'data/val'

#Define standard parameter values
img_height = 500
img_width = 500



def main():
    msg="""
    MIT License
    Original Work:
    Copyright (c) 2020 Hardik Deshmukh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Re-written by Alessandro Festa 2021
    """
    messages(msg)
    #Let's find MLFlow arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size")
    parser.add_argument("--epochs")
    args = parser.parse_args()
    batch_size=int(args.batch_size)
    epochs=int(args.epochs)


    train,test,valid = dataset_config(batch_size,epochs)
    plotSamples(train)
    # Run the function
    make_keras_picklable()
    cnn = model_config(train,test,valid,epochs)
    validate_model(cnn)
    final_validation(cnn,test)

#Let's make the model pickable
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__




def dataset_config(batch_size,epochs):

    msg="""
    ====================================================
    Let's evaluate classes and images for each classes...
    ====================================================
    """
    messages(msg)
    BATCH_SIZE=batch_size
    EPOCHS=epochs
    image_gen = ImageDataGenerator(
                                    rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                
                                )  

    test_data_gen = ImageDataGenerator(rescale = 1./255)



    train = image_gen.flow_from_directory(
        train_path,
        target_size=(img_height, img_width),
        color_mode='grayscale',
        class_mode='binary',
        batch_size=BATCH_SIZE
                                        )

    test = test_data_gen.flow_from_directory(
        test_path,
        target_size=(img_height, img_width),
        color_mode='grayscale',shuffle=False,
        class_mode='binary',
        batch_size=BATCH_SIZE
        )
    valid = test_data_gen.flow_from_directory(
        valid_path,
        target_size=(img_height, img_width),
        color_mode='grayscale',
        class_mode='binary', 
        batch_size=BATCH_SIZE
        )
    # type(train)
    msg="""
    ====================================================
    done...move to next section
    ====================================================
    """
    messages(msg)
    return train,test,valid

def plotSamples(train):
    msg="""
    ====================================================
    Saving a sample in MLFlow to later visaully validate results...
    ====================================================
    """
    messages(msg)
    fig = plt.figure(figsize=(20, 12))
    for i in range(0, 10):
        plt.subplot(2, 5, i+1)
        for X_batch, Y_batch in train:
            image = X_batch[0]        
            dic = {0:'NORMAL', 1:'PNEUMONIA'}
            plt.title(dic.get(Y_batch[0]))
            plt.axis('off')
            plt.imshow(np.squeeze(image),cmap='gray',interpolation='nearest')
            break
    plt.tight_layout()
    # plt.savefig("data/batch_images.pdf")
    # mlflow.log_artifact("data/batch_images.pdf","data")
    mlflow.log_figure(fig,"samples/sample.png")
    msg="""
    ====================================================
    done...move to next section
    ====================================================
    """
    messages(msg)

def model_config(train,test,valid,epochs):
    EPOCHS = epochs
    # mlflow.log_metric("loss", ['val_loss'],step=EPOCHS)
    # mlflow.log_metric("accuracy", ['val_acc'])
    
    # mlflow.log_metric("loss", round(['loss'],2), step=EPOCHS)
    msg = """
        ----------------------------------------------------------------
        Creating the CNN model...
        ----------------------------------------------------------------
        """
    messages(msg)


    cnn = Sequential()

    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))

    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))

    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))

    cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))

    cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))

    cnn.add(Flatten())

    cnn.add(Dense(activation = 'relu', units = 128))
    cnn.add(Dense(activation = 'relu', units = 64))
    cnn.add(Dense(activation = 'sigmoid', units = 1))

    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    early = EarlyStopping(monitor="val_loss", mode="min",patience=3)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

    weights = compute_class_weight('balanced',classes=np.unique(train.classes), y=train.classes)
    cw = dict(zip( np.unique(train.classes), weights))
    history = cnn.fit(train,epochs=epochs, validation_data=valid, class_weight=cw,callbacks=early)
    mlflow_tracking(history)
    return cnn
def validate_model(cnn):
    # fp = "h5/saved_model.h5"
    # cnn.save("h5/saved_model.h5")
    
    # Save

    tf.saved_model.save(cnn, "h5/")
    mlflow.tensorflow.log_model(tf_saved_model_dir="h5", tf_meta_graph_tags=None, tf_signature_def_key='serving_default', artifact_path="model", conda_env=None)
    if not os.path.exists('model/1'):
        os.makedirs("model/1")
    # modelName = os.path.join("1/", "model.pkl")
    # pickle.dump(cnn, open(modelName, 'wb'))
    cnn = load_model("h5/")
    #Copy model to dir "1" so kfserving may pick it up
    # Save the model as an MLflow Model
    # cnn = pickle.load(open(modelName,'rb'))
  

def mlflow_tracking(history):
    mlflow_client = MlflowClient()
    all_metrics = []
    mlflow.tensorflow.autolog()
    for metric_name in history.history:
        for i in history.epoch:
            metric = Metric(
                key=metric_name,
                value=history.history[metric_name][i],
                timestamp=0,
                step=i,
            )
            all_metrics.append(metric)
    mlflow_client.log_batch(run_id=mlflow.active_run().info.run_id, metrics=all_metrics)
def final_validation(cnn,test):
        test_accu = cnn.evaluate(test)
        print('The testing accuracy is :',test_accu[1]*100, '%')
        preds = cnn.predict(test,verbose=1)
        predictions = preds.copy()
        predictions[predictions <= 0.5] = 0
        predictions[predictions > 0.5] = 1

        #Confusion Matrix
        cm = confusion_matrix(test.classes, predictions)
        t_n, f_p, f_n, t_p = cm.ravel()
        mlflow.log_metric("true negative", t_n)
        mlflow.log_metric("false positives", f_p)
        mlflow.log_metric("false negatives", f_n)
        mlflow.log_metric("true positives", t_p)
        cm = pd.DataFrame(data=confusion_matrix(test.classes, predictions, labels=[0, 1]),
                        index=["Actual Normal", "Actual Pneumonia"],
                        columns=["Predicted Normal", "Predicted Pneumonia"])
        fig = plt.figure()
        ax = sns.heatmap(cm,annot=True,fmt="d")
        plt.savefig("data/confusion_matrix.png")
 
        # store confusion matrix alongside model on mlflow
        client = mlflow.tracking.MlflowClient()
        client.log_artifact(mlflow.active_run().info.run_id, "data/confusion_matrix.png")

        results=classification_report(y_true=test.classes, y_pred=predictions,target_names =['NORMAL','PNEUMONIA'] )
        print(classification_report(y_true=test.classes, y_pred=predictions,target_names =['NORMAL','PNEUMONIA'] ))
        res_file = os.path.join("data", "results.txt")
        res_result = open(res_file, "w")
        res_result.write(str(results))
        res_result.close()
        mlflow.log_artifact("data/results.txt","data")
        

        test.reset()
        x=np.concatenate([test.next()[0] for i in range(test.__len__())])
        y=np.concatenate([test.next()[1] for i in range(test.__len__())])
        print(x.shape)
        print(y.shape)

        dic = {0:'NORMAL', 1:'PNEUMONIA'}
        fig = plt.figure(figsize=(20,20))
        for i in range(0+87, 9+87):
            plt.subplot(1,1,1)
        if preds[i, 0] >= 0.5: 
            out = ('{:.2%} probability of being Pneumonia case'.format(preds[i][0]))
            
            
        else: 
            out = ('{:.2%} probability of being Normal case'.format(1-preds[i][0]))     

        plt.title(out+"\n Actual case : "+ dic.get(y[i]))    
        plt.imshow(np.squeeze(x[i]))
        plt.axis('off')
        plt.savefig("validation.png")
        # store confusion matrix alongside model on mlflow
        client = mlflow.tracking.MlflowClient()
        client.log_artifact(mlflow.active_run().info.run_id, "validation.png")

        mlflow.end_run()
#Let's retrive MLProjec arguments and init the training
if __name__ == "__main__":
    main()

