import create_model
import image_data_generator
from keras.callbacks import Callback

class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            print("saving model", name)
            self.model.save_weights(name)
        self.batch += 1

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

batch_size=20
print("Loadind model")
# load model
model = create_model.get_base_model()
print("Model loaded successfully")

print("Get data generator")
# get train data generator
train_generator,validation_generator = image_data_generator.get_image_generator()

print("Start fitting model")
# fit model with train_generator
model.fit_generator(
        train_generator,
        steps_per_epoch=100 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=60 // batch_size)

print("save model to h5 file")
model.save_weights('first_try.h5')
