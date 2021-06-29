import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import utils

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

EPOCHS=25

class BASEMODEL:
    def __init__(self, dataset, class_num, batch_size, input_size, save_dir, valid_dataset=None):
        self.class_num = class_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))
        self.dataset = dataset
        self.valid_dataset = valid_dataset
        self.save_dir = save_dir
        #self.model = self.__create_model()
        
    def loss_angle(self, y_true, y_pred, alpha=0.5):
        #tf.print(y_pred)
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]
        #cls_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.keras.utils.to_categorical(bin_true, 66), logits=y_pred)
        cls_loss = tf.keras.metrics.sparse_categorical_crossentropy(bin_true, y_pred, from_logits=True)[0]

        # MSE loss
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * self.idx_tensor, 1) * 3 - 99
        mse_loss = tf.keras.metrics.mean_squared_error(cont_true, pred_cont)
        # Total loss
        total_loss = cls_loss + alpha * mse_loss
        #tf.print(cls_loss, bin_true, cont_true)
        return total_loss

    def train(self, model_path, max_epoches=EPOCHS, load_weight=True):
        self.model.summary()

        if load_weight:
            self.model.load_weights(model_path)
        else:
            csv_logger = tf.keras.callbacks.CSVLogger('training.log', append=True)
            tb_cb = tf.keras.callbacks.TensorBoard(log_dir=self.save_dir, histogram_freq=1)
            best_model = model_path.replace('.h5', '_best.h5')
            modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath = best_model,
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='min',
                                  period=1)
            early_stopping =  tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.0,
                            patience=10,
                            )
            def lr_schedul(epoch):
                x = 0.0001
                if epoch >= 30:
                    x = 0.00005
                return x
            lr_decay = tf.keras.callbacks.LearningRateScheduler( lr_schedul, verbose=1,)
            callbacks=[ tf.keras.callbacks.TerminateOnNaN(), 
                        modelCheckpoint,
                        csv_logger, 
                        lr_decay,
                        early_stopping,
                        tb_cb]
        
            if self.valid_dataset != None:
                self.model.fit_generator(generator=self.dataset.data_generator(test=False),
                                    epochs=max_epoches,
                                    steps_per_epoch=self.dataset.train_num // self.batch_size,
                                    callbacks=callbacks,
                                    max_queue_size=10,
                                    workers=1,
                                    validation_data=self.valid_dataset.data_generator(test=False, mode='valid'),
                                    validation_steps=self.valid_dataset.test_num // self.batch_size,
                                    verbose=1)
            else:
                self.model.fit_generator(generator=self.dataset.data_generator(test=False),
                                    epochs=max_epoches,
                                    steps_per_epoch=self.dataset.train_num // self.batch_size,
                                    callbacks=callbacks,
                                    max_queue_size=10,
                                    workers=1,
                                    verbose=1)

            last_model = model_path.replace('.h5', '_last.h5')
            self.model.save(last_model)
            
    def test(self):
        num_test = self.dataset.test_num
        for i, (images, [batch_yaw, batch_pitch, batch_roll], names) in enumerate(self.dataset.data_generator(test=True)):
            if i >= num_test: break
            predictions = self.model.predict(images, batch_size=self.batch_size, verbose=1)
            predictions = np.asarray(predictions)
            pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0,:,:]) * self.idx_tensor, 1) * 3 - 99
            pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1,:,:]) * self.idx_tensor, 1) * 3 - 99
            pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2,:,:]) * self.idx_tensor, 1) * 3 - 99
            # print(pred_cont_yaw.shape)
            
            self.dataset.save_test(names[0], self.save_dir, [pred_cont_yaw[0], pred_cont_pitch[0], pred_cont_roll[0]])

    def test_online(self, face_imgs):
        batch_x = np.array(face_imgs, dtype=np.float32)
        predictions = self.model.predict(batch_x, batch_size=1, verbose=1)
        predictions = np.asarray(predictions)
        # print(predictions)
        pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2, :, :]) * self.idx_tensor, 1) * 3 - 99
        
        return pred_cont_yaw[0], pred_cont_pitch[0], pred_cont_roll[0]
    
    def test_save_predicted(self, save_dir, num_test = 0):
        if num_test == 0:
            num_test = self.dataset.test_num
        name_list = []
        pred_list = []
        truth_list = []
        for i, (images, [batch_yaw, batch_pitch, batch_roll], names) in enumerate(self.dataset.data_generator(test=True)):
            if i >= num_test: break
            predictions = self.model.predict(images, batch_size=self.batch_size, verbose=1)
            pred_list.append(np.asarray(predictions))
            name_list.append(names)
            truth_list.append(np.array([batch_yaw, batch_pitch, batch_roll]))
            
        np.savez_compressed(save_dir+'/predicted.npz', 
                    name=name_list,
                    predictions=np.array(pred_list),
                    truth = np.array(truth_list)
                    )


class AlexNet(BASEMODEL):
    def __init__(self, dataset, class_num, batch_size, input_size, save_dir, valid_dataset='None'):
        super().__init__(dataset, class_num, batch_size, input_size, save_dir, valid_dataset=valid_dataset)
        self.model = self.__create_model()

    def preprocess(self, img):
        crop_img = np.asarray(cv2.resize(img, (self.input_size, self.input_size)))
        return (crop_img - crop_img.mean()) / crop_img.std()

    def __create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_size, self.input_size, 3))
        
        feature = tf.keras.layers.Conv2D(filters=64, kernel_size=(11, 11), strides=4, padding='same', activation=tf.nn.relu)(inputs)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Conv2D(filters=192, kernel_size=(5, 5), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = tf.keras.layers.Flatten()(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        feature = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(feature)
        
        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(feature)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(feature)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(feature)
    
        model = tf.keras.Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll])
        
        losses = {
            'yaw':self.loss_angle,
            'pitch':self.loss_angle,
            'roll':self.loss_angle,
        }
        
        model.compile(
                #optimizer=tf.train.AdamOptimizer(),
                optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                loss=losses)
       
        with open("model_alexnet.txt", "w") as fp:
            model.summary(print_fn=lambda x: fp.write(x + "\r\n"))
        return model
        
class Mobilenetv2(BASEMODEL):
    def __init__(self, dataset, class_num, batch_size, input_size, save_dir, valid_dataset='None'):
        super().__init__(dataset, class_num, batch_size, input_size, save_dir, valid_dataset=valid_dataset)
        self.model = self.__create_model()

    def __create_model(self):
        backbone = tf.keras.applications.MobileNetV2(
            weights="imagenet", 
            include_top=False, input_shape=(self.input_size, self.input_size, 3)
            )
        #backbone.trainable = False

        inputs = tf.keras.layers.Input(shape=(self.input_size, self.input_size, 3))
        feature = backbone(inputs)

        #feature = keras.layers.SeparableConv2D(1280 , kernel_size=3, strides=1, activation="relu")(feature)
        feature = tf.keras.layers.GlobalAveragePooling2D()(feature)

        #feature = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        #feature = tf.keras.layers.Flatten()(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        #feature = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(feature)
        
        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(feature)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(feature)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(feature)
    
        model = tf.keras.Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll])
        
        losses = {
            'yaw':self.loss_angle,
            'pitch':self.loss_angle,
            'roll':self.loss_angle,
        }
        
        model.compile(  optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                        loss=losses)
       
        with open("model_movilenetv2.txt", "w") as fp:
            model.summary(print_fn=lambda x: fp.write(x + "\r\n"))
        return model
        
