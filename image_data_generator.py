from keras.preprocessing.image import ImageDataGenerator

train_data_dir = "/home/gaurav/codiecon-2018/flipkart-dataset/train"
validation_data_dir = "/home/gaurav/codiecon-2018/flipkart-dataset/validation"
test_data_dir = "/home/gaurav/codiecon-2018/flipkart-dataset/test"

target_size = (224,224)
batch_size_train=20
batch_size_validation=10
batch_size_test=10

classes = ["Bath and Spa","Containers & Bottle","Living Room Furniture","Office Supplie","Team Support","Women's Clothing", "Women's Footwear"]
train_datagen = ImageDataGenerator().flow_from_directory(train_data_dir,target_size=target_size,classes=classes)
validation_datagen = ImageDataGenerator().flow_from_directory(validation_data_dir,target_size=target_size,classes=classes)
# test_datagen = ImageDataGenerator().flow_from_directory(test_data_dir,target_size=target_size,classes=classes)

def get_image_generator():
    return train_datagen,validation_datagen

def get_train_image_generator():
    return train_datagen
