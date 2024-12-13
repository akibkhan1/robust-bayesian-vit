import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib
matplotlib.use('Agg')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For multiple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt 
import numpy as np
import math
import time, sys
import pickle
import timeit
import wandb
import shutil
import pandas as pd
import keras.backend as K 
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
import keras
os.environ["WANDB_API_KEY"] = "a7b3bca989f1cf6c97c2cbf57f77de63403a5fe5"
plt.ioff() 
#from keras.constraints import Constraint
from utils_v1 import VDP_ViT
        
# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0: 
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


#def min_max_scaling(x):
#    min_val = tf.reduce_min(x)
#    max_val = tf.reduce_max(x)
#    scaled_x = (x - min_val) / (max_val - min_val + 1e-7)
#    return scaled_x 
class CustomLoss(Loss):
    def __init__(self, weight=None, name="custom_cross_entropy_loss"):
        """
        Initializes the custom cross-entropy loss.
        
        Args:
            weight (float or None): Optional weight to scale the loss. If None, no scaling is applied.
            name (str): Name of the loss function.
        """
        super().__init__(name=name)
        self.weight = weight

    def call(self, y_true, y_pred_with_sigma):
        """
        Computes the custom cross-entropy loss.
        
        Args:
            y_true (Tensor): True labels (one-hot encoded or categorical).
            y_pred (Tensor): softmax output.
        
        Returns:
            Tensor: Computed loss.
        """
#        # Apply softmax to predicted logits
#        y_pred = tf.nn.softmax(y_pred)
        
        # Compute the cross-entropy loss
       # loss = -tf.reduce_sum(y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()), axis=-1)
        y_pred, sigma = y_pred_with_sigma
        # Compute the cross-entropy loss
        #y_pred_sd = tf.math.softplus(sigma )
        s = tf.math.divide_no_nan(1., sigma)
        
        loss1 = tf.math.reduce_mean( tf.math.multiply((y_true - y_pred)**2 , s ), axis=-1)
        #loss1 = -tf.math.reduce_mean( tf.math.multiply( y_true * tf.math.log(y_pred +1e-4), s ), axis=-1)
        #loss1 = l1/tf.math.reduce_mean(l1)
#        mu = y_test - y_pred_mean #loss_fn(y_test, y_pred_mean) 
#        mu_2 = mu ** 2
        
      #  loss1 = tf.math.reduce_sum(tf.math.multiply(loss, s), axis=-1) #with scaling with sigma
        loss2 = tf.math.reduce_mean(tf.math.log(sigma), axis=-1) 
        #loss2 = l2 /tf.math.reduce_mean(l2)
        loss = 0.5*tf.math.reduce_mean(tf.math.add(loss1, loss2))#/( tf.reduce_max(tf.math.add(loss1, loss2))+ tf.keras.backend.epsilon() )
#        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
#        loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)

#        mu = y_test - y_pred_mean #loss_fn(y_test, y_pred_mean) 
#        mu_2 = mu ** 2
#        y_pred_sd = y_pred_sd + 1e-5
#        s = tf.math.divide_no_nan(1., y_pred_sd)
#        loss1 = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.multiply(mu_2, s), axis=-1)) #with scaling with sigma
#        #loss1 = tf.math.reduce_mean(tf.math.reduce_sum(mu)) #without scaling with sigma
#        loss2 = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.log(y_pred_sd), axis=-1)) 
#        loss = tf.math.reduce_mean(tf.math.add(loss1, loss2))
#        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
#        loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
        
        # Apply weighting if specified 
        if self.weight is not None:
            loss *= self.weight    
        return tf.reduce_mean(loss)     
##############################################################     
def main_function(image_size=28, patch_size=8, num_layers=2, num_classes=10, embed_dim=32, num_heads=4, mlp_dim=32,
                  channels=1, drop_prob=0.1, batch_size=100, epochs=200, lr=0.001, lr_end=0.00001, kl_factor=0.00001, 
                  Targeted=False, Random_noise=False, gaussain_noise_std=0.3, epsilon=0.1, Training=False, Testing=False,
                  Adversarial_noise=False, adversary_target_cls=3, PGD_Adversarial_noise=False, stepSize=1, power = 0.5, 
                  maxAdvStep=20, continue_training=False, saved_model_epochs=800):  
 
    PATH = './saved_models_Mnist_NLL_latest/loss_VDP_trans_mnist_epoch_{}_lr_{}_kl_{}_layers_{}_drop{}_power{}/'.format(epochs,lr,kl_factor, num_layers,drop_prob,power)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  
    x_train, x_test = x_train / 255.0, x_test / 255.0  
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')  
    x_train = tf.expand_dims(x_train, -1) 
    x_test = tf.expand_dims(x_test, -1)
    one_hot_y_train = tf.one_hot(y_train.astype(np.float32), depth=num_classes)
    one_hot_y_test = tf.one_hot(y_test.astype(np.float32), depth=num_classes)
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, one_hot_y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)

    trans_model = VDP_ViT(image_size=image_size, patch_size=patch_size, num_layers=num_layers, num_classes=num_classes,
                          embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim,
                          channels=channels, dropout=drop_prob, name='vdp_trans')
 
    num_train_steps = epochs * int(x_train.shape[0] / batch_size)
    #    step = min(step, decay_steps) 
    #    ((initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ^ (power) ) + end_learning_rate
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr,
                                                                     decay_steps=num_train_steps,
                                                                     end_learning_rate=lr_end, power=power)
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate_fn)#, clipvalue=1.0)
    #loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    loss_fn = CustomLoss()#    tf.keras.losses.CategoricalCrossentropy(from_logits=False)
     
    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            mu_out, sigma, kl = trans_model(x, training=True)
            trans_model.trainable = True
            #epsilon = tf.random.normal(shape=mu_out.shape, mean=0.0, stddev=1., dtype=x.dtype)
            #y_out = mu_out + sigma*epsilon
            loss_final1 = loss_fn(y, (mu_out, sigma))#/init_loss1 #+ tf.math.reduce_mean(sigma)
#            loss_final2 = nll_gaussian(y, mu_out, tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-8),
#                                                                  clip_value_max=tf.constant(1e+2)))/init_loss2
            
            #regularization_loss= kl#/init_kl #+ tf.reduce_mean(tf.norm(tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+5)), ord='euclidean', axis=-1)) 
            loss = (loss_final1 + kl_factor* kl) #regularization_loss )#/tf.cast( tf.shape( tf.reshape(y, [-1])), tf.float32)
        gradients = tape.gradient(loss, trans_model.trainable_weights)
        gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        optimizer.apply_gradients(zip(gradients, trans_model.trainable_weights))
        return loss, mu_out, sigma, gradients, kl, loss_final1
    @tf.function
    def validation_on_batch(x, y):
        mu_out, sigma, kl = trans_model(x, training=False)
     #   epsilon = epsilon = tf.random.normal(shape=mu_out.shape, mean=0.0, stddev=1., dtype=x.dtype)
      #  y_out = mu_out + sigma*epsilon
       # trans_model.trainable = False
        vloss1 = loss_fn(y, (mu_out, sigma))#/init_loss1 #+  tf.math.reduce_mean(sigma)
#        vloss2 = nll_gaussian(y, mu_out, tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-8),
#                                                         clip_value_max=tf.constant(1e+2)))/init_loss2
        
       # regularization_loss= kl #/init_kl #+ tf.reduce_mean(tf.norm(tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+5)), ord='euclidean', axis=-1))
        total_vloss = (vloss1 + kl_factor*kl)#/tf.cast( tf.shape( tf.reshape(y, [-1])), tf.float32)
        return total_vloss, mu_out, sigma, vloss1, kl
    @tf.function
    def test_on_batch(x, y):
        trans_model.trainable = False
        mu_out, sigma, kl = trans_model(x, training=False)
        return mu_out, sigma
    @tf.function
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            trans_model.trainable = False
            prediction, sigma, kl = trans_model(input_image, training=True) 
          #  epsilon = epsilon = tf.random.normal(shape=prediction.shape, mean=0.0, stddev=1., dtype=x.dtype)
           # y_out = prediction + sigma*epsilon
            loss_final = loss_fn(input_label, (prediction, sigma)) 
            #loss_final = 0.01*nll_gaussian(input_label, prediction,
            #                          tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-8),   clip_value_max=tf.constant(1e+2)))
           # regularization_loss= kl  #+ tf.reduce_mean(tf.norm(tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+5)), ord='euclidean', axis=-1))
            loss = (loss_final)# + kl_factor*regularization_loss)
            
            #loss =  loss_final
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
          return signed_grad
    if Training:
        if os.path.exists(PATH):
            shutil.rmtree(PATH) 
        os.makedirs(PATH)        
        wandb.init(entity="bayes-transformer",
                   project="Akib_MNIST_NLL_epochs_{}_layer_{}_lr_{}_kl_factor_{}_batchsize_{}_dimension_{}_patch_{}_head_{}_drop_{}".format(
                       epochs, num_layers, lr, kl_factor, batch_size, embed_dim, patch_size, num_heads,drop_prob))        
#        if continue_training:
#            saved_model_path = './Old_loss_VDPtransformer_saved_models_new/VDP_trans_epoch_{}_kl_{}_lr_{}latestv11/'.format(saved_model_epochs, kl_factor,lr)
#            trans_model.load_weights(saved_model_path + 'vdp_trans_model')
        train_acc = np.zeros(epochs)
        valid_acc = np.zeros(epochs)
        valid_acc_n = np.zeros(epochs)
        valid_acc_adv = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        start = timeit.default_timer()
        for epoch in range(epochs):
            print('Epoch: ', epoch+1, '/' , epochs)
            acc1 = 0
            acc_valid1 = 0
            acc_valid1_n = 0
            acc_valid1_adv = 0
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0
            #-------------Training--------------------
            acc_training = np.zeros(int(x_train.shape[0] / (batch_size)))
            err_training = np.zeros(int(x_train.shape[0] / (batch_size)))
#            if epoch == int(epochs/6):
#               lambda_factor=0.2  
#            if epoch == int(epochs/4):
#               lambda_factor=0.1  
#            if epoch == int(epochs/2):
#               lambda_factor=0.01   
            for step, (x, y) in enumerate(tr_dataset):
                update_progress(step/int(x_train.shape[0]/(batch_size)) )
                
#                if (epoch==0 and tr_no_steps==0):
#                   loss, mu_out, sigma, gradients, regularization_loss_0, loss_final1_0 = train_on_batch(x, y)
#                   loss, mu_out, sigma, gradients, regularization_loss, loss_final1 = train_on_batch(x, y, init_loss1=loss_final1_0, init_kl=regularization_loss_0)
#                else:
#                   loss, mu_out, sigma, gradients, regularization_loss, loss_final1 = train_on_batch(x, y, init_loss1=loss_final1_0, init_kl=regularization_loss_0)
#                              
                loss, mu_out, sigma, gradients, regularization_loss, loss_final1 = train_on_batch(x, y)#, init_loss1=loss_final1_0, init_kl=regularization_loss_0)     
                err1+= loss.numpy()
                corr = tf.equal(tf.math.argmax(mu_out, axis=-1),tf.math.argmax(y,axis=-1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc1+=accuracy.numpy()
                if step % 50 == 0:
                    print('\n gradient', np.mean(gradients[0].numpy()))
                    print('\n Matrix Norm', np.mean(sigma))
                    print("\n Step:", step, "Loss:" , float(err1/(tr_no_steps + 1.)))
                    print("Total Training accuracy so far: %.3f" % float(acc1/(tr_no_steps + 1.)))
                tr_no_steps+=1
                wandb.log({"Average Variance value": tf.reduce_mean(sigma).numpy(),
                            "Total Training Loss": loss.numpy() ,
                            "Training Accuracy per minibatch": accuracy.numpy() ,
                            "gradient per minibatch": np.mean(gradients[0]),
                            'epoch': epoch,
                            "Regularization_loss": regularization_loss.numpy(),
                            "Cross-Entropy Loss": np.mean(loss_final1.numpy())           
                   })
            train_acc[epoch] = acc1/tr_no_steps
            train_err[epoch] = err1/tr_no_steps
            print('Training Acc  ', train_acc[epoch])
            print('Training error  ', train_err[epoch])
            #---------------Validation----------------------
            for step, (x, y) in enumerate(val_dataset):
                update_progress(step / int(x_test.shape[0] / (batch_size)) )
                
                noise = tf.random.normal(shape=[batch_size, image_size, image_size, 1], mean=0.0,
                                         stddev=gaussain_noise_std, dtype=x.dtype)
                noisy_x = x+ noise                        
                #total_vloss, mu_out, sigma, vloss   = validation_on_batch(x, y)
                
#                if (epoch==0 and va_no_steps==0):
#                   total_vloss, mu_out, sigma,vloss1_0,v_regularization_loss10=validation_on_batch(x, y)
#                   total_vloss_n, mu_out_n,sigma_n,vloss2_0,v_regularization_loss20=validation_on_batch(noisy_x, y)
#                   
#                   total_vloss, mu_out, sigma,vloss1,v_regularization_loss1=validation_on_batch(x, y, init_loss1=vloss1_0, init_kl=v_regularization_loss10)
#                   total_vloss_n, mu_out_n,sigma_n,vloss2,v_regularization_loss2=validation_on_batch(noisy_x, y, init_loss1=vloss2_0, init_kl=v_regularization_loss20)
#                else:
#                   total_vloss, mu_out, sigma,vloss1,v_regularization_loss1=validation_on_batch(x, y, init_loss1=vloss1_0,  init_kl=v_regularization_loss10)
#                   total_vloss_n, mu_out_n,sigma_n,vloss2,v_regularization_loss2=validation_on_batch(noisy_x, y, init_loss1=vloss2_0, init_kl=v_regularization_loss20)
#               
                          
                adv_x = x + epsilon * create_adversarial_pattern(x, y)
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
                
                total_vloss, mu_out, sigma,vloss1,v_regularization_loss1=validation_on_batch(x, y)
                total_vloss_n, mu_out_n,sigma_n,vloss2,v_regularization_loss2=validation_on_batch(noisy_x, y)   
                total_vloss_adv, mu_out_adv,sigma_adv,vloss3,v_regularization_loss3=validation_on_batch(adv_x, y)  
                                        
                err_valid1+= total_vloss.numpy()
                corr_n = tf.equal(tf.math.argmax(mu_out_n, axis=-1),tf.math.argmax(y,axis=-1))
                va_accuracy_n = tf.reduce_mean(tf.cast(corr_n,tf.float32))
                acc_valid1_n +=va_accuracy_n.numpy()
                
                corr_adv = tf.equal(tf.math.argmax(mu_out_adv, axis=-1),tf.math.argmax(y,axis=-1))
                va_accuracy_adv = tf.reduce_mean(tf.cast(corr_adv,tf.float32))
                acc_valid1_adv +=va_accuracy_adv.numpy()
                
                corr = tf.equal(tf.math.argmax(mu_out, axis=-1),tf.math.argmax(y,axis=-1))
                va_accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_valid1+=va_accuracy.numpy()
                
                if step % 50 == 0:
                    print("Step:", step, "Loss:", float(total_vloss))
                    print("Total validation accuracy so far: %.3f" % va_accuracy)
                va_no_steps+=1
                wandb.log({"Average Variance value (validation)": tf.reduce_mean(sigma).numpy(),
                              "Total Validation Loss": total_vloss.numpy() ,
                              "Validation Acuracy per minibatch": va_accuracy.numpy(),
                              "Validation Acuracy per minibatch with noise": va_accuracy_n.numpy(),
                              "Validation Acuracy per minibatch with Adv": va_accuracy_adv.numpy(),
                              "Average Variance value (validation) Noise": tf.reduce_mean(sigma_n).numpy(),
                              "Average Variance value (validation) Adv": tf.reduce_mean(sigma_adv).numpy(),
                              "Valid Regularization_loss": v_regularization_loss1.numpy(),
                              "Valid Cross-Entropy Loss": np.mean(vloss1.numpy()),
                              "Valid Cross-Entropy (noise) Loss": np.mean(vloss2.numpy()),
                              "Valid Cross-Entropy (Adv) Loss": np.mean(vloss3.numpy())
                               })
            valid_acc[epoch] = acc_valid1/va_no_steps
            valid_error[epoch] = err_valid1/va_no_steps 
            valid_acc_n[epoch] = acc_valid1_n/va_no_steps
            valid_acc_adv[epoch] = acc_valid1_adv/va_no_steps
            stop = timeit.default_timer()
            if np.max(valid_acc) == valid_acc[epoch]:
                trans_model.save_weights(PATH+'vdp_model_best.weights.h5')
                
            wandb.log({"Average Training Loss":  train_err[epoch],
                       "Average Training Accuracy": train_acc[epoch],
                       "Average Validation Loss": valid_error[epoch],
                       "Average Validation Accuracy": valid_acc[epoch],
                       "Average Validation Accuracy Noise": valid_acc_n[epoch],
                       "Average Validation Accuracy Adv": valid_acc_adv[epoch],
                       'epoch': epoch
                      })
            print('Total Training Time: ', stop - start)
            print('Training Acc  ', train_acc[epoch])
            print('Validation Acc  ', valid_acc[epoch])
            print('------------------------------------')
            print('Training error  ', train_err[epoch])
            print('Validation error  ', valid_error[epoch])
        #-----------------End Training--------------------------
        #trans_model.save_weights(PATH + 'vdp_trans_model.weights.h5')
        #trans_model.save(PATH+'vdp_cnn_model_last.keras')
        trans_model.save_weights(PATH+'vdp_model_last.weights.h5')
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            plt.ylim(0, 1.1)
            plt.title("Density Propagation Transformer on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VDP_Trans_on_MNIST_Data_acc.png')
            plt.close(fig)

            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')
            plt.plot(valid_error,'r' , label='Validation error')
            plt.title("Density Propagation Transformer on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_Trans_on_MNIST_Data_error.png')
            plt.close(fig)

        f = open(PATH + 'training_validation_acc_error.pkl', 'wb')
        pickle.dump([train_acc, valid_acc, train_err, valid_error], f)
        f.close()

        textfile = open(PATH + 'Related_hyperparameters.txt','w')
        textfile.write(' Input Dimension : ' + str(image_size))
        textfile.write('\n Hidden units : ' + str(mlp_dim))
        textfile.write('\n Number of Classes : ' + str(num_classes))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' + str(lr_end))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        if Training:
            textfile.write('\n Total run time in sec : ' + str(stop - start))
            if (epochs == 1):
                textfile.write("\n Averaged Training  Accuracy : " + str(train_acc))
                textfile.write("\n Averaged Validation Accuracy : " + str(valid_acc))

                textfile.write("\n Averaged Training  error : " + str(train_err))
                textfile.write("\n Averaged Validation error : " + str(valid_error))
            else:
                textfile.write("\n Averaged Training  Accuracy : " + str(np.mean(train_acc[epoch])))
                textfile.write("\n Averaged Validation Accuracy : " + str(np.mean(valid_acc[epoch])))

                textfile.write("\n Averaged Training  error : " + str(np.mean(train_err[epoch])))
                textfile.write("\n Averaged Validation error : " + str(np.mean(valid_error[epoch])))
        textfile.write("\n---------------------------------")
        textfile.write("\n--------------------------------")
        textfile.close()

    #-------------------------Testing-----------------------------
    if Testing:
        test_path = 'test_results/'
        if Random_noise:
            print(f'Random Noise: {gaussain_noise_std}')
            test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
        full_test_path = PATH + test_path
        if os.path.exists(full_test_path):
            # Remove the existing test path and its contents
            shutil.rmtree(full_test_path)
        os.makedirs(PATH + test_path)
        #trans_model.load_weights(PATH + 'vdp_trans_model.weights.h5')
        #trans_model = tf.keras.models.load_model(PATH + 'vdp_model_best.keras', custom_objects={'VDP_ViT': VDP_ViT})
        trans_model.load_weights_safely(PATH+'vdp_model_best.weights.h5')

        test_no_steps = 0
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, image_size, image_size, 1])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)))
            true_x[test_no_steps, :, :, :, :] = x
            true_y[test_no_steps, :, :] = y
            if Random_noise:
                noise = tf.random.normal(shape=[batch_size, image_size, image_size, 1], mean=0.0,
                                         stddev=gaussain_noise_std, dtype=x.dtype)
                x = x + noise
            mu_out, sigma = test_on_batch(x, y)
            mu_out_[test_no_steps, :, :] = mu_out
            sigma_[test_no_steps, :, :] = sigma
            corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            acc_test[test_no_steps] = accuracy.numpy()
            if step % 100 == 0:
                print("Total running accuracy so far: %.3f" % acc_test[test_no_steps])
            test_no_steps += 1           

        test_acc = np.mean(acc_test)
        print('Test accuracy : ', test_acc)
        ave_uncer = np.mean(sigma_)
      
        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')
        pickle.dump([mu_out_, sigma_, true_x, true_y, test_acc], pf)
        pf.close()

        var = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])
        if Random_noise:
            snr_signal = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])
        for i in range(int(x_test.shape[0] / (batch_size))):
            for j in range(batch_size):
                if Random_noise:
                    noise = tf.random.normal(shape=[image_size, image_size, 1], mean=0.0, stddev=gaussain_noise_std, dtype=x.dtype)
                    snr_signal[i, j] = 10 * np.log10(np.sum(np.square(true_x[i, j, :, :, :])) / np.sum(np.square(noise)))

                predicted_out = np.argmax(mu_out_[i, j, :])
                var[i, j] = sigma_[i, j, int(predicted_out)]
        if Random_noise:        
            print('SNR', np.mean(snr_signal))

        print('Average Output Variance', np.mean(var))
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(image_size))
        textfile.write('\n Number of Classes : ' + str(num_classes))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' + str(lr_end))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n Output Variance: " + str(np.mean(np.abs(var))))
        textfile.write("\n Average Uncertainty: " + str(np.mean( ave_uncer)))
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: ' + str(gaussain_noise_std))
            textfile.write("\n SNR: " + str(np.mean(snr_signal)))
        textfile.write("\n---------------------------------")
        textfile.close()

#####
    def divide_no_nan(x, y):
        # Create a mask for non-zero elements of y
        non_zero_mask = y != 0
        # Perform the division, setting elements to 0 where y is zero
        result = np.where(non_zero_mask, np.divide(x, y, where=non_zero_mask), 0)
        return result
    if (Adversarial_noise):
        if Targeted:
            test_path = 'test_results_targeted_adversarial_noise_{}/'.format(epsilon)
            full_test_path = PATH + test_path
            if os.path.exists(full_test_path):
            # Remove the existing test path and its contents
              shutil.rmtree(full_test_path)
            os.makedirs(PATH + test_path)
        else:
            test_path = 'test_results_non_targeted_adversarial_noise_{}/'.format(epsilon)
            full_test_path = PATH + test_path
            if os.path.exists(full_test_path):
            # Remove the existing test path and its contents
              shutil.rmtree(full_test_path)
            os.makedirs(PATH + test_path)
        #trans_model.load_weights(PATH + 'vdp_trans_model')
       # trans_model = tf.keras.models.load_model(PATH + 'vdp_model_best.keras', custom_objects={'VDP_ViT': VDP_ViT})
        trans_model.load_weights_safely(PATH+'vdp_model_best.weights.h5')
        
        test_no_steps = 0
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, image_size, image_size, 1])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, image_size, image_size, 1])
        adv_perturbations_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, image_size, image_size, 1])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])

        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)))
            true_x[test_no_steps, :, :, :, :] = x
            true_y[test_no_steps, :, :] = y

            if Targeted:
                y_true_batch = np.zeros_like(y)
                y_true_batch[:, adversary_target_cls] = 1.0
                adv_perturbations[test_no_steps, :, :, :, :] = create_adversarial_pattern(x, y_true_batch)
            else:
                adv_perturbations[test_no_steps, :, :, :, :] = create_adversarial_pattern(x, y)
            #print(np.sum(adv_perturbations[test_no_steps,:, :, :,:]))
#            mean_ = tf.reduce_mean(adv_perturbations[test_no_steps, :, :, :, :])
#            std_ = tf.math.reduce_std(adv_perturbations[test_no_steps, :, :, :, :])
#            adv_perturbations[test_no_steps, :, :, :, :] = (adv_perturbations[test_no_steps, :, :, :, :] - mean_) / (std_ + 1e-8)  
            adv_x = x + epsilon * adv_perturbations[test_no_steps, :, :, :, :]
            adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
            adv_perturbations_[test_no_steps, :, :, :, :] = adv_x
            mu_out, sigma = test_on_batch(adv_x, y)
            mu_out_[test_no_steps, :, :] = mu_out
            sigma_[test_no_steps, :, :] = sigma
            corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            acc_test[test_no_steps] = accuracy.numpy()
            if step % 10 == 0:
                print("Total running accuracy so far: %.3f" % accuracy.numpy())
            test_no_steps += 1

        test_acc = np.mean(acc_test)
        print('Test accuracy : ', test_acc)
        ave_uncer = np.mean(sigma_)

        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')
        pickle.dump([mu_out_, sigma_, true_x,true_y, adv_perturbations_, test_acc], pf)
        pf.close()

        var = np.zeros([int(x_test.shape[0] / batch_size), batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] / batch_size), batch_size])
        for i in range(int(x_test.shape[0] / batch_size)):
            for j in range(batch_size):
                predicted_out = np.argmax(mu_out_[i, j, :])
                var[i, j] = sigma_[i, j, int(predicted_out)]
               # print(np.sum(adv_perturbations[i, j, :, :,:]))
             #   print(np.sum(true_x[i,j,:, :,:]))
                snr_signal[i,j] = np.nan_to_num(10*np.log10( divide_no_nan(np.sum(np.square(true_x[i,j,:, :,:])),(np.sum( np.square(true_x[i,j,:, :,:] - adv_perturbations_[i, j, :, :,:] ) ))) ), copy=True, nan=0.0, posinf=0.0, neginf=0.0)
               # snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :,:]))/np.sum( np.square(true_x[i,j,:,:, :] - np.clip(true_x[i,j,:,:, :]+epsilon*adv_perturbations[i, j, :, :,:], 0.0001, 1.0)  ) )+1e-10)   
                #snr_signal[i,j] = 10*np.log10( np.sum(np.square(true_x[i,j,:, :,:]))/(np.sum( np.square( epsilon*adv_perturbations[i, j, :, :,:]     ) )+1e-6))                
                #print(snr_signal[i, j])
        print('Output Variance', np.mean(var))
        print('SNR', np.mean(snr_signal))
        

        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(image_size))
        textfile.write('\n Number of Classes : ' + str(num_classes))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' + str(lr_end))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Accuracy : " + str(test_acc))
        textfile.write("\n Output Variance: " + str(np.mean(np.abs(var))))
        textfile.write("\n Average Uncertainty: " + str(np.mean( ave_uncer)))
        textfile.write("\n---------------------------------")
        if Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))
            else:
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: ' + str(epsilon))
            textfile.write("\n SNR: " + str(np.mean(snr_signal)))
        textfile.write("\n---------------------------------")
        textfile.close()

    if (PGD_Adversarial_noise):
        if Targeted:
          test_path = 'test_results_targeted_PGDadversarial_noise_{}_max_iter_{}_{}/'.format(epsilon, maxAdvStep, stepSize)
          full_test_path = PATH + test_path
          if os.path.exists(full_test_path):
            # Remove the existing test path and its contents
              shutil.rmtree(full_test_path)
          os.makedirs(PATH + test_path)
        else:
          test_path = 'test_results_non_targeted_PGDadversarial_noise_{}/'.format(epsilon)
          full_test_path = PATH + test_path
          if os.path.exists(full_test_path):
            # Remove the existing test path and its contents
              shutil.rmtree(full_test_path)
          os.makedirs(PATH + test_path)


        #trans_model.load_weights(PATH + 'vdp_trans_model')
       # trans_model = tf.keras.models.load_model(PATH + 'vdp_model_best.keras', custom_objects={'VDP_ViT': VDP_ViT})
        trans_model.load_weights_safely(PATH+'vdp_model_best.weights.h5')

        test_no_steps = 0
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, image_size, image_size, channels])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, image_size, image_size, channels])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size,  num_classes])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size,  num_classes])

        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)))
            true_x[test_no_steps, :, :, :,:] = x
            true_y[test_no_steps, :, :] = y

            adv_x = x + tf.random.uniform(x.shape, minval=-epsilon, maxval=epsilon)
            adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
            for advStep in range(maxAdvStep):
                if Targeted:
                    y_true_batch = np.zeros_like(y)
                    y_true_batch[:, adversary_target_cls] = 1.0
                    adv_perturbations_ = create_adversarial_pattern(adv_x, y_true_batch)
                else:
                    adv_perturbations_ = create_adversarial_pattern(adv_x, y)
                adv_x = adv_x + stepSize * adv_perturbations_
                adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
                    
#                pgdTotalNoise = tf.clip_by_value(adv_x - x, -epsilon, epsilon)
#                adv_x = tf.clip_by_value(x + pgdTotalNoise, 0.0, 1.0)
                adv_perturbations[test_no_steps, :, :, :,:] = adv_x
            mu_out, sigma = test_on_batch(adv_x, y)
            mu_out_[test_no_steps, :, :] = mu_out
            sigma_[test_no_steps, :, :] = sigma
            corr = tf.equal(tf.math.argmax(mu_out, axis=-1), tf.math.argmax(y, axis=-1))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            acc_test[test_no_steps] = accuracy.numpy()
            if step % 50 == 0:
                print("Total running accuracy so far: %.4f" % acc_test[test_no_steps])
            test_no_steps += 1
        test_acc = np.mean(acc_test)
        print('Test accuracy : ', test_acc)
        print('Best Test accuracy : ', np.amax(acc_test))
        ave_uncer = np.mean(sigma_)

        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')
        pickle.dump([mu_out_, sigma_, true_x, true_y, adv_perturbations, test_acc], pf)
        pf.close()

        var = np.zeros([int(x_test.shape[0] / batch_size), batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] / batch_size), batch_size])
        for i in range(int(x_test.shape[0] / batch_size)):
            for j in range(batch_size):
                predicted_out = np.argmax(mu_out_[i, j, :])
                var[i, j] = sigma_[i, j, int(predicted_out)]
                snr_signal[i, j] = 10 * np.log10(np.sum(np.square(true_x[i, j, :, :, :])) / (np.sum(np.square(true_x[i, j, :, :, :] -   adv_perturbations[i, j, :, :, :] ) )+1e-6))
                    
        print('Output Variance', np.mean(var))
        print('SNR', np.mean(snr_signal))
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(image_size))
        textfile.write('\n Number of Classes : ' + str(num_classes))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' + str(lr_end))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n Output Variance: " + str(np.mean(np.abs(var))))
        textfile.write("\n Average Uncertainty: " + str(np.mean( ave_uncer)))
        textfile.write("\n---------------------------------")
        if PGD_Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))
            else:
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: ' + str(epsilon))
            textfile.write("\n SNR: " + str(np.mean(snr_signal)))
            textfile.write("\n stepSize: " + str(stepSize))
            textfile.write("\n Maximum number of iterations: " + str(maxAdvStep))
        textfile.write("\n---------------------------------")
        textfile.close()
if __name__ == '__main__':   
    main_function(Training=True) 
    
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.001,Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)                  
    main_function(Random_noise=True, gaussain_noise_std=0.001, epsilon=0.005,Training=False, Testing=True,
                  Adversarial_noise=False, PGD_Adversarial_noise=False)                 
    main_function(Random_noise=True, gaussain_noise_std=0.01, epsilon=0.01,Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.05, epsilon=0.05,Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.1, epsilon=0.07,Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.2, epsilon=0.08,Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.3, epsilon=0.1,Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.4, epsilon=0.15,Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
    main_function(Random_noise=True, gaussain_noise_std=0.5, epsilon=0.2,Training=False, Testing=True,
                  Adversarial_noise=False,  PGD_Adversarial_noise=False)
###                   
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.001,Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)                 
    main_function(Random_noise=False, gaussain_noise_std=0.001, epsilon=0.005,Training=False, Testing=False,
                  Adversarial_noise=True, PGD_Adversarial_noise=False)               
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.01,Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.05, epsilon=0.05,Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.1, epsilon=0.07,Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.2, epsilon=0.08,Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.3, epsilon=0.1,Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.4, epsilon=0.15,Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
    main_function(Random_noise=False, gaussain_noise_std=0.5, epsilon=0.2,Training=False, Testing=False,
                  Adversarial_noise=True,  PGD_Adversarial_noise=False)
#    main_function(Random_noise=False, gaussain_noise_std=0.5, epsilon=0.3,Training=False, Testing=False,
#                  Adversarial_noise=True,  PGD_Adversarial_noise=False) 
####                   
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.001,Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)                
    main_function(Random_noise=False, gaussain_noise_std=0.001, epsilon=0.005,Training=False, Testing=False,
                  Adversarial_noise=False, PGD_Adversarial_noise=True)                  
    main_function(Random_noise=False, gaussain_noise_std=0.01, epsilon=0.01,Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.05, epsilon=0.05,Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.1, epsilon=0.07,Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.2, epsilon=0.08,Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.3, epsilon=0.1,Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.4, epsilon=0.15,Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
    main_function(Random_noise=False, gaussain_noise_std=0.5, epsilon=0.2,Training=False, Testing=False,
                  Adversarial_noise=False,  PGD_Adversarial_noise=True)
                  