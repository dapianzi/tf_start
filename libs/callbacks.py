"""
控制model训练时的控制台打印回调
"""
from tensorflow import keras


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    # def on_train_batch_end(self, batch, logs=None):
    #     print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))
    #
    # def on_test_batch_end(self, batch, logs=None):
    #     print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 50 == 0:
            print(logs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 50 == 0:
            print(
                "Epoch {}/ is {:7.2f} "
                "and mean absolute error is .".format(
                    epoch, logs["loss"]
                )
            )
            print(logs)
