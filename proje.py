def vgg19():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.vgg19 import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
    from glob import glob
    import cv2
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    #import scikitplot.metrics as splt
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,BatchNormalization,GlobalAveragePooling2D,MaxPooling2D
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    import numpy as np
    import os
    import keras
    from tensorflow.keras.applications import DenseNet201
    from collections import Counter
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import confusion_matrix
    import itertools  
     
    
    sonuclar=['Covid','Lung_Opacity','Normal','Viral_Pneumonia']  
    anaklasor='/content/drive/MyDrive/Uygulama/covidveriseti/'
    targetsizeX=224
    targetsizeY=224
    batchsize=20
    epochsayisi=15
    valsplit=0.2
    lrate = 0.01
    decay = lrate/epochsayisi
    sgd = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    #adag=keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    ada=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    #rsm=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    #adamm=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    optimizers=ada
    IMAGE_SIZE = [224, 224]
    vgg =InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    for layer in vgg.layers:
      layer.trainable = False
    folders = glob('/content/drive/MyDrive/Uygulama/covidveriseti/*')  
    x = Flatten()(vgg.output)
    x= Dropout(0.5)(x)
    x= BatchNormalization()(x)
    x= Dropout(0.5)(x)
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=vgg.input, outputs=prediction) 
    model.summary()
    # klasör isimlerini alma  
    #sonuclar=[]
    #klasorismi='/content/drive/MyDrive/Uygulama/vs/'
    #directories=os.listdir(klasorismi)
    #for directory in directories:
    #    sonuclar.append(directory)      
    #print(sonuclar)    
    
    #modelin derlenmesi       
    model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers,
    metrics=['accuracy']
    )
    data_generator = ImageDataGenerator(
                                          rescale = 1. / 255, #normalizasyon işlemi
                                          validation_split = float (valsplit))#hold-out yöntemiyle veri setini ayırır
          
    Train=data_generator.flow_from_directory(anaklasor,
                                        target_size=(int (targetsizeX), int (targetsizeY)),
                                        shuffle=True,
                                        batch_size=int (batchsize),
                                        class_mode='categorical',
                                        subset="training")    
                        
    Test=data_generator.flow_from_directory(anaklasor,
                                        target_size=(int (targetsizeX), int (targetsizeY)),
                                        shuffle=False,
                                        batch_size=int (batchsize),
                                        class_mode='categorical',
                                        subset="validation")
    
    #karşılaştırılacak resimi yükleme               
    img=image.load_img('/content/drive/MyDrive/Uygulama/covidveriseti/Viral_Pneumonia/Viral Pneumonia-48.png',target_size=(224,224))
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    resim=np.vstack([X])
    # fit model
    #for i in range (0,len(folders)):
    model_fit =model.fit_generator(
      Train,
      validation_data=Test,
      epochs=int (epochsayisi),
      steps_per_epoch=len(Train),
      validation_steps=len(Test)
    )    
    
    scores = model.evaluate(Test, verbose=0)#verbose=ayrıntılı
    
    print(scores[1])
    print(" Accuracy: %.2f%%"% (scores[1]*100))
        
    #başarı grafiği
    plt.figure(figsize=(5,3))
    plt.plot(model_fit.history['accuracy'])
    plt.plot(model_fit.history['val_accuracy'])
    plt.title('Model Accuracy Grafiği')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("/content/drive/MyDrive/projesonuc/InceptionV3basari.png")
    #loss grafiği                 
    plt.figure(figsize=(5,3))
    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.title('Model Loss Grafiği')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("/content/drive/MyDrive/projesonuc/InceptionV3loss.png")
    #confusion matrix
    classes = []
    for i in Train.class_indices:
          classes.append(i)
    tahmin = model.predict_generator(Test)
    y_pred = np.argmax(tahmin, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(Test.classes, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    print("thresh",thresh)
    for i in range (cm.shape[0]):
      for j in range (cm.shape[1]):
          plt.text(j, i, cm[i, j],
          horizontalalignment="center",
          color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("/content/drive/MyDrive/projesonuc/InceptionV3confusionmatrix.png")
    X_train, y_train = next(Train)
    X_test, y_test = next(Test)            
    tahmin=model.predict_generator(Train)
    model.save('/content/drive/MyDrive/covidVGG19_model.h5')
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    print("tn",str(TN))
    print("tp",str(TP))
    print("fn",str(FN))
    print("fp",str(FP))
    sensitivity=round(float(TP)/(TP+FN)*100,4)
    specificity=round(float(TN)/(FP+TN)*100,4)
    precision=round(float(TP)/(TP+FP)*100,4)
    recall=round(float(TP)/(TP+FN)*100,4)
    f1=round(2*(float(precision*recall)/(precision+recall)))
    print("sensitivity "+str(sensitivity))
    print("specificity "+str(specificity))
    print("precision "+str(precision))
    print("recall "+str(recall))
    print("f1 "+str(f1))
    
    output = model.predict(resim)
    enbuyuk=tahmin[0][0]
    k=0
    for i in range(4):
        if enbuyuk<output[0][i]:
            enbuyuk=output[0][i]
            k=i       
    print("Sonuc : %.2f%%" % (enbuyuk*100)+" "+sonuclar[k])

def densenet201():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.vgg19 import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
    from glob import glob
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from tensorflow.keras.optimizers import Adam
    #from tensorflow.keras.utils import multi_gpu_model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,BatchNormalization
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.applications import DenseNet201
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import confusion_matrix
    import itertools  
    
    anaklasor='/content/drive/MyDrive/Uygulama/covidveriseti/'
    targetsizeX=224
    targetsizeY=224
    batchsize=20
    epochsayisi=15
    valsplit=0.2
    lrate = 0.001
    decay = lrate/epochsayisi
    sgd = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    optimizers='adam'
    optimizerler=['Adadelta','adam','Adamax','SGD']   
    IMAGE_SIZE = [224, 224]
    densenet201 =VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    for layer in densenet201.layers:
      layer.trainable = False
    folders = glob('/content/drive/MyDrive/Uygulama/covidveriseti/*')
    x = Flatten()(densenet201.output)
    x= Dropout(0.5)(x)
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=densenet201.input, outputs=prediction) 
    model.summary()
    print(optimizers)  
    #modelin derlenmesi       
    model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers,
    metrics=['accuracy']
    )
    data_generator = ImageDataGenerator(
                                        rescale = 1. / 255, #normalizasyon işlemi
                                        validation_split = float(valsplit))#hold-out yöntemiyle veri setini ayırır
        
    Train=data_generator.flow_from_directory(anaklasor,
                                      target_size=(int(targetsizeX),int(targetsizeY)),
                                      shuffle=True,
                                      batch_size=int(batchsize),
                                      class_mode='categorical',
                                      subset="training")    
                      
    Test=data_generator.flow_from_directory(anaklasor,
                                      target_size=(int(targetsizeX),int(targetsizeY)),
                                      shuffle=False,
                                      batch_size=int(batchsize),
                                      class_mode='categorical',
                                      subset="validation")
                  
    # fit model
    model_fit =model.fit_generator(
    Train,
    validation_data=Test,
    epochs=int(epochsayisi),
    steps_per_epoch=len(Train),
    validation_steps=len(Test)
    )    
    scores = model.evaluate(Test, verbose=0)#verbose=ayrıntılı
    print(optimizers+" "+"Accuracy: %.2f%%" % (scores[1]*100))
    X_train, y_train = next(Train)
    X_test, y_test = next(Test)            
    tahmin=model.predict_generator(Train)
    
    #başarı grafiği
    plt.figure(figsize=(5,3))
    plt.plot(model_fit.history['accuracy'])
    plt.plot(model_fit.history['val_accuracy'])
    plt.title(optimizers+' '+'Model Accuracy Grafiği')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("/content/drive/MyDrive/sonuc/ResNet50V2"+optimizers+"basari.png")
    #loss grafiği                 
    plt.figure(figsize=(5,3))
    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.title(optimizers+' '+'Model Loss Grafiği')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("/content/drive/MyDrive/sonuc/ResNet50V2"+optimizers+"loss.png")
    #confusion matrix
    classes = []
    for i in Train.class_indices:
          classes.append(i)
    tahmin = model.predict_generator(Test)
    y_pred = np.argmax(tahmin, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(Test.classes, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(str(i)+'Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    print("thresh",thresh)
    for i in range (cm.shape[0]):
      for j in range (cm.shape[1]):
          plt.text(j, i, cm[i, j],
          horizontalalignment="center",
          color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("/content/drive/MyDrive/sonuc/ResNet50V2"+optimizers+"confusionmatrix.png")
    X_train, y_train = next(Train)
    X_test, y_test = next(Test)            
    tahmin=model.predict_generator(Train)
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    print("tn",str(TN))
    print("tp",str(TP))
    print("fn",str(FN))
    print("fp",str(FP))
    sensitivity=round(float(TP)/(TP+FN)*100,4)
    specificity=round(float(TN)/(FP+TN)*100,4)
    precision=round(float(TP)/(TP+FP)*100,4)
    recall=round(float(TP)/(TP+FN)*100,4)
    f1=round(2*(float(precision*recall)/(precision+recall)))
    print("sensitivity "+str(sensitivity))
    print("specificity "+str(specificity))
    print("precision "+str(precision))
    print("recall "+str(recall))
    print("f1 "+str(f1))
    #h5 dosyası kaydetme        
    model.save('/content/drive/MyDrive/sonuc/DNNDenseNet201'+optimizers+'model.h5')  
    
def resnet50v2():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from glob import glob
import scikitplot.metrics as splt
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,BatchNormalization
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import numpy as np
import os
from collections import Counter
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import itertools  
import matplotlib.pyplot as plt

sonuclar=['Covid','Lung_Opacity','Normal','Viral_Pneumonia']
anaklasor='/content/drive/MyDrive/Uygulama/covidveriseti/'
targetsizeX=224
targetsizeY=224
batchsize=16
epochsayisi=30
valsplit=0.2
#lrate = 0.001
#decay = lrate/epochsayisi
#sgd = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
optimizers='hepsi'
optimizerler=['Adadelta','adam','Adamax','SGD','sgd']
if (optimizers=='Hepsi'):
  for i in range (len(optimizerler)):
      optimizers=optimizerler[i]
      print(optimizers)  
      IMAGE_SIZE = [220, 220]
      resnet50v2 =ResNet50V2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
      for layer in resnet50v2.layers:
        layer.trainable = False
      folders = glob('/content/drive/MyDrive/Uygulama/covidveriseti/*')
      x = Flatten()(resnet50v2.output)
      x= Dense(500,activation='relu')(x) 
      x= BatchNormalization()(x)
      x= Dropout(0.5)(x)
      prediction = Dense(len(folders), activation='softmax')(x)
      model = Model(inputs=resnet50v2.input, outputs=prediction) 
      model.summary()

      #modelin derlenmesi       
      model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers,
        metrics=['accuracy']
      )
      data_generator = ImageDataGenerator(
                                              rescale = 1. / 255, #normalizasyon işlemi
                                              validation_split = float (valsplit))#hold-out yöntemiyle veri setini ayırır
              
      Train=data_generator.flow_from_directory(anaklasor,
                                            target_size=(int (targetsizeX),int (targetsizeY)),
                                            shuffle=True,
                                            batch_size=int(batchsize),
                                            class_mode='categorical',
                                            subset="training")    
                            
      Test=data_generator.flow_from_directory(anaklasor,
                                            target_size=(int (targetsizeX),int (targetsizeY)),
                                            shuffle=False,
                                            batch_size=int (batchsize),
                                            class_mode='categorical',
                                            subset="validation")
                  
      #karşılaştırılacak resimi yükleme               
      """img=image.load_img('/content/drive/MyDrive/Uygulama/Viral Pneumonia-48.png',target_size=(int(targetsizeX),int(targetsizeY)))
      X=image.img_to_array(img)
      X=np.expand_dims(X,axis=0)
      resim=np.vstack([X])     """  
      # fit model
      model_fit =model.fit_generator(
        Train,
        validation_data=Test,
        epochs=int (epochsayisi),
        steps_per_epoch=len(Train),
        validation_steps=len(Test)
      )    
      scores = model.evaluate(Test, verbose=0)#verbose=ayrıntılı
      print("Accuracy: %.2f%%" % (scores[1]*100))

      X_train, y_train = next(Train)
      X_test, y_test = next(Test)            
      tahmin=model.predict_generator(Train)
      #print(tahmin)
      model.save('/content/drive/MyDrive/sonuc/DNNResNet50V2'+optimizers+'.h5')
#başarı grafiği
      plt.figure(figsize=(5,3))
      plt.plot(model_fit.history['accuracy'])
      plt.plot(model_fit.history['val_accuracy'])
      plt.title(optimizers+' '+'Model Accuracy Grafiği')
      plt.ylabel('Accuracy')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Test'], loc='upper left')
      plt.savefig("/content/drive/MyDrive/sonuc/DNNResNet50V2"+optimizers+"basari.png")
#loss grafiği                 
      plt.figure(figsize=(5,3))
      plt.plot(model_fit.history['loss'])
      plt.plot(model_fit.history['val_loss'])
      plt.title(optimizers+' '+'Model Loss Grafiği')
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.legend(['Train', 'Test'], loc='upper left')
      plt.savefig("/content/drive/MyDrive/sonuc/DNNResNet50V2"+optimizers+"loss.png")

    #confusion matrix
      classes = []
      for i in Train.class_indices:
            classes.append(i)
      tahmin = model.predict_generator(Test)
      y_pred = np.argmax(tahmin, axis=1)
      print('Confusion Matrix')
      cm = confusion_matrix(Test.classes, y_pred)
      plt.figure(figsize=(4,4))
      plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
      plt.title(str(i)+'Confusion matrix')
      plt.colorbar()
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)
      thresh = cm.max() / 2.
      print("thresh",thresh)
      for i in range (cm.shape[0]):
        for j in range (cm.shape[1]):
            plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
      plt.tight_layout()
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
      plt.savefig("/content/drive/MyDrive/sonuc/DNNResNet50V2confusionmatrix.png")         
      TN=cm[0][0]
      FP=cm[0][1]
      FN=cm[1][0]
      TP=cm[1][1]
      print("tn",str(TN))
      print("tp",str(TP))
      print("fn",str(FN))
      print("fp",str(FP))
      sensitivity=round(float(TP)/(TP+FN)*100,4)
      specificity=round(float(TN)/(FP+TN)*100,4)
      precision=round(float(TP)/(TP+FP)*100,4)
      recall=round(float(TP)/(TP+FN)*100,4)
      f1=round(2*(float(precision*recall)/(precision+recall)))
      print("sensitivity "+str(sensitivity))
      print("specificity "+str(specificity))
      print("precision "+str(precision))
      print("recall "+str(recall))
      print("f1 "+str(f1))  
      #hangi sınıfa ait tahmin işemi  
def inceptionv3():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.vgg19 import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
    from glob import glob
    #import scikitplot.metrics as splt
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from tensorflow.keras.optimizers import Adam
    #from tensorflow.keras.utils import multi_gpu_model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D,BatchNormalization
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    import numpy as np
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import confusion_matrix
    import itertools  
    import matplotlib.pyplot as plt
    
    sonuclar=['Covid','Lung_Opacity','Normal','Viral_Pneumonia']  
    anaklasor='/content/drive/MyDrive/Uygulama/covidveriseti/'
    targetsizeX=224
    targetsizeY=224
    batchsize=16
    epochsayisi=30
    valsplit=0.2
    lrate = 0.001
    decay = lrate/epochsayisi
    sgd = tf.keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    optimizers='adadelta'
    
    IMAGE_SIZE = [224, 224]
    inception =InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    #inception=tf.keras.applications.InceptionV3(
    #include_top=False,
    #weights="imagenet",
    #input_tensor=None,
    #input_shape=(229, 229, 3),
    #pooling=None,
    #classes=1000,
    #)
    for layer in inception.layers:
      layer.trainable = False
    folders = glob('/content/drive/MyDrive/Uygulama/covidveriseti/*')
    x = Flatten()(inception.output)
    x= Dense(500,activation='relu')(x) 
    x= BatchNormalization()(x)
    x= Dropout(0.5)(x)
    prediction = Dense(len(folders), activation='softmax')(x)
    model = Model(inputs=inception.input, outputs=prediction) 
    model.summary()
    
    #modelin derlenmesi       
    model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizers,
      metrics=['accuracy']
    )
    data_generator = ImageDataGenerator(
                                            rescale = 1. / 255, #normalizasyon işlemi
                                            validation_split = float (valsplit))#hold-out yöntemiyle veri setini ayırır
            
    Train=data_generator.flow_from_directory(anaklasor,
                                          target_size=(int (targetsizeX), int (targetsizeY)),
                                          shuffle=True,
                                          batch_size=int (batchsize),
                                          class_mode='categorical',
                                          subset="training")    
                          
    Test=data_generator.flow_from_directory(anaklasor,
                                          target_size=(int (targetsizeX), int (targetsizeY)),
                                          shuffle=False,
                                          batch_size=int (batchsize),
                                          class_mode='categorical',
                                          subset="validation")
                
    #karşılaştırılacak resimi yükleme               
    img=image.load_img('/content/drive/MyDrive/Uygulama/covidveriseti/Covid/COVID-30.png',target_size=(224,224))
    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    resim=np.vstack([X])
                       
    # fit model
    model_fit =model.fit_generator(
      Train,
      validation_data=Test,
      epochs=int (epochsayisi),
      steps_per_epoch=len(Train),
      validation_steps=len(Test)
    )    
    scores = model.evaluate(Test, verbose=0)#verbose=ayrıntılı
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    X_train, y_train = next(Train)
    X_test, y_test = next(Test)            
    tahmin=model.predict_generator(Train)
    #print(tahmin)
    model.save('/content/drive/MyDrive/sonuc/InceptionV3_model.h5')
    
    #hangi sınıfa ait tahmin işemi  
    output = model.predict(resim)
    enbuyuk=tahmin[0][0]
    k=0
    for i in range(4):
        if enbuyuk<tahmin[0][i]:
            enbuyuk=tahmin[0][i]
            k=i       
    print("Sonuc : %.2f%%" % (enbuyuk*100)+" "+sonuclar[k])
    #başarı grafiği
    plt.figure(figsize=(5,3))
    plt.plot(model_fit.history['accuracy'])
    plt.plot(model_fit.history['val_accuracy'])
    plt.title(optimizers+' '+'Model Accuracy Grafiği')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("/content/drive/MyDrive/sonuc/InceptionV3"+optimizers+"basari.png")
    #loss grafiği                 
    plt.figure(figsize=(5,3))
    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.title(optimizers+' '+'Model Loss Grafiği')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("/content/drive/MyDrive/sonuc/InceptionV3"+optimizers+"loss.png")
    #confusion matrix
    classes = []
    for i in Train.class_indices:
          classes.append(i)
    tahmin = model.predict_generator(Test)
    y_pred = np.argmax(tahmin, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(Test.classes, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(str(i)+'Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    print("thresh",thresh)
    for i in range (cm.shape[0]):
      for j in range (cm.shape[1]):
          plt.text(j, i, cm[i, j],
          horizontalalignment="center",
          color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("/content/drive/MyDrive/sonuc/InceptionV3confusionmatrix.png")         
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    print("tn",str(TN))
    print("tp",str(TP))
    print("fn",str(FN))
    print("fp",str(FP))
    sensitivity=round(float(TP)/(TP+FN)*100,4)
    specificity=round(float(TN)/(FP+TN)*100,4)
    precision=round(float(TP)/(TP+FP)*100,4)
    recall=round(float(TP)/(TP+FN)*100,4)
    f1=round(2*(float(precision*recall)/(precision+recall)))
    print("sensitivity "+str(sensitivity))
    print("specificity "+str(specificity))
    print("precision "+str(precision))
    print("recall "+str(recall))
    print("f1 "+str(f1)) 