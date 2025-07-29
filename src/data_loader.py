from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, img_size=(300,300), batch_size=32):

    #argument for train

    train_datagen=ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2 # 20% for validation
    )

    #Generator of train

    train_generator= train_datagen.flow_from_directory(
        data_dir,
        target_size = img_size,
        batch_size= batch_size,
        class_mode= 'binary',
        shuffle=True
    )

    #generator of validation

    val_generator= train_datagen.flow_from_directory(
        data_dir,
        target_size = img_size,
        batch_size= batch_size,
        class_mode= 'binary',
        subset='validation',
        shuffle=True
    )

    return train_generator, val_generator