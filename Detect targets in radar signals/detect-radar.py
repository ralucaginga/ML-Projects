import glob
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import ttach
from keras.layers import GlobalAveragePooling2D, Dense
from keras_preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow.python.keras import Model
from sklearn.metrics import classification_report
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn import Linear, ReLU, CrossEntropyLoss
from timm import create_model
from torch.optim import AdamW
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets import ImageFolder
import albumentations as augment ## albumentations should have version 1.1.0
from albumentations.pytorch import ToTensorV2

def put_images_in_folders(labels, class_names):
    for label in class_names:
        os.makedirs(os.path.join(TRAIN_PATH, str(label)))

    for c in class_names:
        for i in list(labels[labels.label == c]['id']):
            get_image = os.path.join(TRAIN_PATH, i)
            move_image_to_folder = shutil.move(get_image, TRAIN_PATH + str(c))

def feature_extraction(model_chosen, classes, TRAINING_PATH, both = "false"):
    extracted_features = list()
    extracted_labels = list()

    if both:
        xception = tf.keras.applications.xception.Xception()
        model1 = Model(xception.input, xception.get_layer('avg_pool').output)
        size1 = (299, 299)

        vgg16 = tf.keras.applications.vgg16.VGG16()
        model2 = Model(vgg16.input, vgg16.get_layer('fc1').output)
        size2 = (224, 224)

        for index, predictor in enumerate(classes):
            # Going to path corresponding to the class name
            temporary_path = TRAINING_PATH + str(predictor)
            print(temporary_path)
            count = 0
            for path_of_the_image in glob.glob(temporary_path + "/*.png"):
                count += 1
                xception_image = tf.keras.preprocessing.image.load_img(path_of_the_image, target_size = size1)
                feature_xception = tf.keras.applications.xception.preprocess_input(np.expand_dims(tf.keras.preprocessing.image.img_to_array(xception_image), axis = 0))
                extracted_xception_feature = model1.predict(feature_xception)
                feature_xception_flattened = extracted_xception_feature.flatten()

                vgg16_image = tf.keras.preprocessing.image.load_img(path_of_the_image, target_size = size2)
                feature_vgg16 = tf.keras.applications.vgg16.preprocess_input(np.expand_dims(tf.keras.preprocessing.image.img_to_array(vgg16_image), axis = 0))
                extracted_vgg16_feature = model2.predict(feature_vgg16)
                feature_vgg16_flattened = extracted_vgg16_feature.flatten()

                features_concatenated = np.concatenate((feature_xception_flattened, feature_vgg16_flattened), axis = 0)
                print(count)
                extracted_features.append(features_concatenated)
                extracted_labels.append(predictor)
    else:
        if model_chosen == 'vgg16':
            vgg16 = tf.keras.applications.vgg16.VGG16()
            model = Model(vgg16.input, vgg16.get_layer('fc1').output)
            size = (224, 224)

            for index, predictor in enumerate(classes):
                # Going to path corresponding to the class name
                temporary_path = TRAINING_PATH + str(predictor)
                print(temporary_path)
                count = 0
                for path_of_the_image in glob.glob(temporary_path + "/*.png"):
                    count += 1
                    vgg16_image = tf.keras.preprocessing.image.load_img(path_of_the_image, target_size = size)
                    feature_vgg16 = tf.keras.applications.vgg16.preprocess_input(np.expand_dims(tf.keras.preprocessing.image.img_to_array(vgg16_image), axis = 0))
                    extracted_vgg16_feature = model.predict(feature_vgg16)
                    feature_vgg16_flattened = extracted_vgg16_feature.flatten()
                    print(count)
                    extracted_features.append(feature_vgg16_flattened)
                    extracted_labels.append(predictor)

        if model_chosen == "xception":
            xception = tf.keras.applications.xception.Xception()
            model = Model(xception.input, xception.get_layer('avg_pool').output)
            size = (299, 299)

            for index, predictor in enumerate(classes):
                # Going to path corresponding to the class name
                temporary_path = TRAINING_PATH + str(predictor)
                print(temporary_path)
                count = 0
                for path_of_the_image in glob.glob(temporary_path + "/*.png"):
                    count += 1
                    xception_image = tf.keras.preprocessing.image.load_img(path_of_the_image, target_size = size)
                    feature_xception = tf.keras.applications.xception.preprocess_input(np.expand_dims(tf.keras.preprocessing.image.img_to_array(xception_image), axis = 0))
                    extracted_xception_feature = model.predict(feature_xception)
                    print(count)
                    feature_xception_flattened = extracted_xception_feature.flatten()
                    extracted_features.append(feature_xception_flattened)
                    extracted_labels.append(predictor)

    return extracted_features, extracted_labels

def execute_augmentation(PATH):
    size_required = (224, 224)

    # Defining a slight augmentation form of data augmentation
    train_augmentation = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=30,
        validation_split=0.15)

    # Train-val splitting the dataset and augmenting the images, defining the labels as well
    training_samples = train_augmentation.flow_from_directory(
        PATH,
        target_size = size_required,
        classes=['1', '2', '3', '4', '5'], # the classes are corresponding to the 5 folders created in the dataset directory
        batch_size=64,
        subset='training')

    validation_samples = train_augmentation.flow_from_directory(
        PATH,
        target_size = size_required,
        classes=['1', '2', '3', '4', '5'],
        batch_size=64,
        subset='validation') # specifying either it's training or validation

    return training_samples, validation_samples

def choose_model(model_name, models_path, train, valid, neurons = 4096, no_of_classes = 5):
    image_shape_of_input = (224, 224, 3)
    if model_name == 'Xception':
        learning_rate = 0.001
        model_chosen = tf.keras.applications.xception.Xception(include_top=False, input_shape = image_shape_of_input)
        input = model_chosen.output
        input = GlobalAveragePooling2D()(input)
        input = Dense(neurons, activation = 'relu')(input)
        final_output = Dense(no_of_classes, activation = 'softmax')(input)
        model_final = tf.keras.Model(inputs = model_chosen.input, outputs = final_output)
        model_final.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                             loss = tf.keras.losses.categorical_crossentropy, metrics = ['accuracy'])
        history = model_final.fit(train,
                            validation_data = valid,
                            epochs = 15)
        model_final.save(models_path + model_name + ".h5")
    elif model_name == 'DenseNet':
        learning_rate = 0.0001
        model_chosen = tf.keras.applications.densenet.DenseNet169(include_top=False, input_shape=image_shape_of_input)
        input = model_chosen.output
        input = GlobalAveragePooling2D()(input)
        final_output = Dense(no_of_classes, activation='softmax')(input)
        model_final = tf.keras.Model(inputs = model_chosen.input, outputs = final_output)
        # Freezing first 55 layers
        for layer_in_model in model_final.layers[0:55]:
            layer_in_model.trainable = False
        model_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                            loss = tf.keras.losses.categorical_crossentropy, metrics = ['accuracy'])
        history = model_final.fit(train,
                                  validation_data=valid,
                                  epochs=15)
        model_final.save(models_path + model_name + ".h5")
    else:
        return "Choose one of the 2 models: Xception/DenseNet"

    return history

def plot_historic_model(list_of_history):
    # Plotting historic of the model in terms of accuracy and loss
    plt.plot(list_of_history.history['accuracy'])
    plt.plot(list_of_history.history['val_accuracy'])
    plt.xlabel('Number of epoch')
    plt.ylabel('Accuracy obtained')
    plt.show()

    plt.plot(list_of_history.history['loss'])
    plt.plot(list_of_history.history['val_loss'])
    plt.xlabel('Number of epoch')
    plt.ylabel('Loss obtained')
    plt.show()

SIZE_OF_IMAGE = 224
augmentation_transformation = augment.Compose([
        augment.Resize(SIZE_OF_IMAGE, SIZE_OF_IMAGE),
        augment.HorizontalFlip(0.5),
        augment.VerticalFlip(),
        augment.RandomRotate90(),
        augment.Normalize(),
        ToTensorV2(),
    ])

class ReadingData(Dataset):
    def __init__(self, data, transformation = None):
        self.data = data
        self.transformation = transformation

    def __getitem__(self, i):
        image = self.data[i][0]
        label = self.data[i][1]
        if self.transformation:
            image = np.array(image)
            image = self.transformation(image = image)['image']
        return image, label

    def __len__(self):
        return len(self.data)

input_feats = 1000
output_feats = 500
no_of_classes = 5

class CustomModel(pl.LightningModule):
    def __init__(self):
        super(CustomModel, self).__init__()
        # First of all, we define the model using Timm library
        self.model = create_model(name_of_model, pretrained = False)
        self.fc1 = Linear(input_feats, output_feats)
        self.relu = ReLU()
        self.fc2 = Linear(output_feats, no_of_classes)
        # Configuration of the params of the model
        self.learning_rate = 0.001
        self.batch_size = 64
        self.num_of_workers = 2
        self.metric = Accuracy()
        self.loss_value = CrossEntropyLoss()
        # Storing accuracy and loss values for both train and validation dataa
        self.train_accuracy_list, self.val_accuracy_list = [], []
        self.train_loss_list, self.val_loss_list = [], []
        # Data Loading
        self.dataset = ImageFolder('dataset/')
        # split data
        self.training_images, self.validation_images = random_split(self.dataset,
                                                    [int(len(self.dataset) * 0.7), int(len(self.dataset) * 0.3)],
                                                    generator = torch.Generator().manual_seed(42))

    # Overriding some of the Pytorch Lightning methods
    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def augmentation_using_mixup(self, sampleX, sampleY, lamda = 0.3):
        '''This function is implementing the formulas present in Zhang paper
        we take some samples of datapoints and give us the combination between them'''
        if lamda > 0:
            l = np.random.beta(lamda, lamda)
        else:
            l = 1
        batch_size = sampleX.size()[0]
        indice = torch.randperm(batch_size)
        combined_sampleX = l * sampleX + (1 - l) * sampleX[indice, :]
        label_1, label_2 = sampleY, sampleY[indice]
        return combined_sampleX, label_1, label_2, l

    def combine_loss_mixup(self, predition, label_1, label_2, lamda):
        # taking the loss corresponding to the mixup formula from Zhang paper
        return lamda * self.loss_value(predition, label_1) + (1 - lamda) * self.loss_value(predition, label_2)

    def configure_optimizers(self):
        # setting the params for the Optimizers and scheduler
        t_0 = 10
        t_mult = 1
        eta_min = 0.000001
        optimizator = AdamW(params = self.parameters(), lr = self.learning_rate)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizator, T_0 = t_0, T_mult = t_mult, eta_min = eta_min)
        return {'optimizer': optimizator, 'lr_scheduler': lr_scheduler}

    def train_dataloader(self):
        return DataLoader(ReadingData(self.training_images, augmentation_transformation), shuffle = True,
                          batch_size = self.batch_size,
                          num_workers = self.num_of_workers,
                          pin_memory = True)

    def val_dataloader(self):
        return DataLoader(ReadingData(self.validation_images, augmentation_transformation),
                          batch_size = self.batch_size, num_workers = self.num_of_workers,
                          pin_memory = True)

    def training_step(self, batch, batch_idx):
        training_image, image_label = batch
        combined_sampleX, label_1, label_2, lamda = self.augmentation_using_mixup(training_image, image_label)
        prediction = self(combined_sampleX)
        loss_value = self.combine_loss_mixup(prediction, label_1, label_2, lamda)
        accuracy_value = self.metric(prediction, image_label)
        return {'loss': loss_value, 'acc': accuracy_value}

    def validation_step(self, batch, batch_idx):
        validation_image, validation_label = batch
        prediction = self(validation_image)
        loss_value = self.loss_value(prediction, validation_label)
        accuracy_value = self.metric(prediction, validation_label)
        return {'loss': loss_value, 'acc': accuracy_value}

    def training_epoch_end(self, outs):
        loss_value = torch.stack([history["loss"] for history in outs]).mean().detach().cpu().numpy().round(3)
        accuracy_value = torch.stack([history["acc"] for history in outs]).mean().detach().cpu().numpy().round(3)
        self.train_accuracy_list.append(accuracy_value)
        self.train_loss_list.append(loss_value)

    def validation_epoch_end(self, outs):
        loss_value = torch.stack([history["loss"] for history in outs]).mean().detach().cpu().numpy().round(3)
        accuracy_value = torch.stack([history["acc"] for history in outs]).mean().detach().cpu().numpy().round(3)
        print('Epoch {}, Loss Value {}, Accuracy Value {}:'.format(self.current_epoch, loss_value, accuracy_value))

def evaluate_model(model_chosen, tta = False):

    batches = model_chosen.val_dataloader()
    model_chosen.cuda.eval()
    actual_values, predicted_values = [], []

    if tta == True:
        transformation = ttach.Compose(
            [
                ttach.HorizontalFlip(),
                ttach.VerticalFlip(),
                ttach.Rotate90(angles=[0, 90, 180, 270])
            ]
        )
        test_time_augmentation_wrapper = ttach.ClassificationTTAWrapper(model_chosen, transformation)
        with torch.no_grad():
            for batch in batches:
                validation_image, validation_label = batch
                predicted_value = test_time_augmentation_wrapper(validation_image.cuda())
                predicted_value = torch.argmax(predicted_value, dim=1).detach().cpu().numpy()
                actual_values.append(validation_label.cpu().numpy())
                predicted_values.append(predicted_value)
    else:
        with torch.no_grad():
            for batch in batches:
                validation_image, validation_label = batch
                predicted_value = model_chosen(validation_image.cuda())
                predicted_value = torch.argmax(predicted_value, dim=1).detach().cpu().numpy()
                actual_values.append(validation_label.cpu().numpy())
                predicted_values.append(predicted_value)

    return actual_values, predicted_values

def get_predictions(model_chosen, tta = False):

    model_chosen.cuda.eval()
    actual_values, predicted_values = [], []

    if tta == True:
        transformation = ttach.Compose(
            [
                ttach.HorizontalFlip(),
                ttach.VerticalFlip(),
                ttach.Rotate90(angles=[0, 90, 180, 270])
            ]
        )
        test_time_augmentation_wrapper = ttach.ClassificationTTAWrapper(model_chosen, transformation)
        with torch.no_grad():
            for batch in loader_of_test:
                test_image, test_label = batch
                predicted_value = test_time_augmentation_wrapper(test_image.cuda())
                predicted_value = torch.argmax(predicted_value, dim=1).detach().cpu().numpy()
                actual_values.append(test_label.cpu().numpy())
                predicted_values.append(predicted_value)
    else:
        with torch.no_grad():
            for batch in loader_of_test:
                test_image, test_label = batch
                predicted_value = model_chosen(test_image.cuda())
                predicted_value = torch.argmax(predicted_value, dim=1).detach().cpu().numpy()
                actual_values.append(test_label.cpu().numpy())
                predicted_values.append(predicted_value)

    return predicted_values

if __name__ == "__main__":
    print(tf.config.list_physical_devices("GPU"))
    # Ensuring our environment will use GPU
    print(tf.test.is_built_with_cuda())
    print(torch.cuda.is_available())

    # Path of the dataset.
    TRAIN_PATH = 'dataset/train/'
    TEST_PATH = 'dataset/test/'
    MODELS_PATH = 'models'

    train = pd.read_csv("train.csv")
    labels = train.sort_values('label')
    classes = list(labels.label.unique())

    # Data Preparation
    '''
    Our dataset contains:
    - two .csv files that contain the images names and one of them (train.csv) contains the label as well, whereas test.csv contains only the name of the image
    - two folders (corresponding to train and test datasets) with the images

    In order to work on such a project, I should split the training set (based on the labels present in train.csv file) into 5 different folders corresponding to each class.
    '''

    # Uncomment the following line of code if the 5 folders are not created and all of the train images are in "train" folder
    #put_images_in_folders(labels, class_names)

    # Approach 1 - Feature Extraction using Pretrained Models and using Classical ML Algorithms for predicting the labels
    extracted_feats, labels = feature_extraction('xception', classes, TRAIN_PATH, both = "True")

    (train_features, test_features, train_classes, test_classes) = train_test_split(np.array(extracted_feats),
                                                                              np.array(labels),
                                                                              test_size = 0.15,
                                                                              random_state = 9)

    xgboost = xgb.XGBClassifier()
    xgboost.fit(train_features, train_classes)
    predictions_of_xgboost = xgboost.predict(test_features)
    print(classification_report(test_classes, predictions_of_xgboost))

    logistic_reg = LogisticRegression()
    logistic_reg.fit(train_features, train_classes)
    predictions_of_logisticRegression = logistic_reg.predict(test_features)

    print(classification_report(test_classes, predictions_of_logisticRegression))

    svc_linearity = LinearSVC()
    svc_linearity.fit(train_features, train_classes)
    predictions_of_linearSVC = svc_linearity.predict(test_features)

    print(classification_report(test_classes, predictions_of_linearSVC))

    # Approach 2 - Pretrained Models for Predicting the Labels

    training_dataset, validation_dataset = execute_augmentation(TRAIN_PATH)
    model_chosen = choose_model("Xception", MODELS_PATH, training_dataset, validation_dataset)
    plot_historic_model(model_chosen)

    # Approach 3 - Mixup
    # Creating a data augmentation transformation
    name_of_model = 'mobilenetv2_100'
    mobilenet = CustomModel()
    name_of_model = 'efficientnetv2_rw_s'
    efficientnet = CustomModel()

    history_mobilenet = Trainer(max_epochs = 50, gpus = -1, num_sanity_val_steps = 0, precision = 16, accumulate_grad_batches = 1)
    history_mobilenet.fit(mobilenet)
    torch.save(mobilenet.state_dict(), MODELS_PATH + 'model_mobilenet.pt')
    history_efficientnet = Trainer(max_epochs = 50, gpus = -1, num_sanity_val_steps = 0, precision = 16, accumulate_grad_batches = 1)
    history_efficientnet.fit(efficientnet)
    torch.save(efficientnet.state_dict(), MODELS_PATH + 'model_efficientnet.pt')

    #mobilenet.load_state_dict(torch.load(MODELS_PATH + 'model_mobilenet.pt'))
    #efficientnet.load_state_dict(torch.load(MODELS_PATH + 'model_efficientnet.pt'))

    # Evaluation of the model against the validation and extracting the classification report for this
    actuals_mobilenet, predictions_mobilenet = evaluate_model(mobilenet)
    #predictions_efficientnet = evaluate_model(efficientnet)
    #predictions_mobilenet_tta = evaluate_model(mobilenet, True)
    #predictions_efficientnet_tta = evaluate_model(efficientnet, True)
    ## An example of classification report
    print(classification_report(np.hstack(actuals_mobilenet), np.hstack(predictions_mobilenet)))

    # Preparing the submission file
    test_dataset = ImageFolder(TEST_PATH)
    loader_of_test = DataLoader(ReadingData(test_dataset, augmentation_transformation), batch_size = mobilenet.batch_size,
                                num_workers = mobilenet.num_of_workers, pin_memory = True)
    classes = mobilenet.dataset.class_to_idx
    classes = {index:label for label, index in classes.items()}
    # Getting the mapping of the classes
    print(classes)
    predictions_submission = get_predictions(mobilenet, True)
    ## Creating a dataframe with the predictions corresponding to each image path
    img_path = [image[0] for image in test_dataset.imgs]
    predictions_labels = [classes[i] for i in np.hstack(predictions_submission)]
    image_path = [i.split('test/', 1)[1] for i in img_path]
    results = pd.DataFrame({'id': image_path, 'label': predictions_labels})

    test = pd.read_csv('test.csv')
    test = pd.merge(test, results, how = 'left', on = 'id')
    test.to_csv(MODELS_PATH + "submission_mobilenet_tta.csv", index = False)










