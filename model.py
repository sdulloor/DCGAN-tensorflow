from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         data_dir=None, input_fname_labels='labels.txt',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None,
         conditional=False, loss_type=0):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess

    self.data_dir = data_dir
    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.input_fname_labels = input_fname_labels
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y, self.y_dim, self.c_dim = self.load_mnist()
    else:
      self.img_data, self.img_labels, self.y_dim, self.c_dim = self.load_image_dataset()

    self.grayscale = (self.c_dim == 1)
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.loss_type = loss_type
    self.conditional = conditional
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.build_model()

  def build_model(self):
    self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    if self.dataset_name == 'mnist':
      image_dims = [self.input_height, self.input_width, self.c_dim]
    else:
      image_dims = [self.output_height, self.output_width, self.c_dim]

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    #loss_type = 0 -> cross entropy
    #loss_type = 1 -> vanilla logloss
    #loss_type = 2 -> wasserstein

    if self.loss_type == 0:
      #cross entropy loss
      self.d_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
      self.d_loss_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
      self.g_loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
      self.d_loss = self.d_loss_real + self.d_loss_fake
    elif self.loss_type == 1:
      #vanilla logloss
      self.d_loss_real = -tf.reduce_mean(tf.log(self.D))
      self.d_loss_fake = -tf.reduce_mean(tf.log(1-self.D_))
      self.d_loss = self.d_loss_real+self.d_loss_fake
      self.g_loss = -tf.reduce_mean(tf.log(self.D_))
    elif self.loss_type == 2:
      #wasserstein
      self.d_loss_real = tf.reduce_mean(self.D_logits)
      self.d_loss_fake = -tf.reduce_mean(self.D_logits_)
      self.d_loss = self.d_loss_real+self.d_loss_fake
      self.g_loss = -tf.reduce_mean(self.D_logits_)

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def optimizer(self, config):
    if config.loss_type == 2:
      d_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(-self.d_loss, var_list=self.d_vars)
      g_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.g_loss, var_list=self.g_vars)
    else:
      d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                        .minimize(self.d_loss, var_list=self.d_vars)
      g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                        .minimize(self.g_loss, var_list=self.g_vars)
    return d_optim, g_optim

  def read_images(self, image_files):
    # read images
    image = [
        get_image(image_file,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  crop=self.crop,
                  grayscale=self.grayscale) for image_file in image_files]
    if (self.grayscale):
      image_inputs = np.array(image).astype(np.float32)[:, :, :, None]
    else:
      image_inputs = np.array(image).astype(np.float32)

    # read labels
    image_basename = [os.path.basename(x) for x in image_files]
    y_labels = self.img_labels[self.img_labels['image'].isin(image_basename)]
    y_labels = y_labels['identity'].tolist()

    # image labels (one-hot vector)
    y = np.array(y_labels)
    image_labels = np.zeros((y.shape[0], self.y_dim), dtype=np.float)
    image_labels[np.arange(y.shape[0]), y] = 1.0

    return image_inputs, image_labels

  def read_dataset(self, config):
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      sample_files = self.img_data[0:self.sample_num]
      sample_inputs, sample_labels = self.read_images(sample_files)
    return sample_inputs, sample_labels

  def read_next_batch(self, config, idx):
    if config.dataset == 'mnist':
      batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
      batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
    else:
      batch_files = self.img_data[idx*config.batch_size:(idx+1)*config.batch_size]
      batch_images, batch_labels = self.read_images(batch_files)
    return batch_images, batch_labels

  def train(self, config):
    d_optim, g_optim = self.optimizer(config)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs/{}".format(int(time.time())), self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

    sample_inputs, sample_labels = self.read_dataset(config)

    counter = 1
    start_time = time.time()

    ## Load from checkpoint
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter

    ## start training
    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:      
        self.data = glob(os.path.join(
          config.data_dir, config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch_images, batch_labels = self.read_next_batch(config, idx)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        ### Training always assumes labels.
        # If the network is not conditional, discriminator and generator
        # will simply ignore the labels
        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict={
            self.inputs: batch_images,
            self.z: batch_z,
            self.y:batch_labels,
          })
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={
            self.z: batch_z,
            self.y:batch_labels,
          })
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict={ self.z: batch_z, self.y:batch_labels })
        self.writer.add_summary(summary_str, counter)

        errD_fake = self.d_loss_fake.eval({
            self.z: batch_z,
            self.y:batch_labels
        })
        errD_real = self.d_loss_real.eval({
            self.inputs: batch_images,
            self.y:batch_labels
        })
        errG = self.g_loss.eval({
            self.z: batch_z,
            self.y: batch_labels
        })
        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 100) == 1:
          try:
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
              },
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
          except:
            print("one pic error!...")

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def dcgan_discriminator(self, image, y, reuse):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      x = tf.reshape(image, [self.batch_size, -1])

      h0 = lrelu(linear(x, self.dfc_dim, 'd_h0_lin'))
      h1 = linear(h0, 1, 'd_h1_lin')

      return tf.nn.sigmoid(h1), h1

  def dcgan_cond_discriminator(self, image, y, reuse):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      x = tf.reshape(image, [self.batch_size, -1])
      xy = concat([x, y], 1)

      h0 = lrelu(linear(xy, self.dfc_dim, 'd_h0_lin'))
      h1 = linear(h0, 1, 'd_h1_lin')

      return tf.nn.sigmoid(h1), h1

  def dcgan_generator(self, z, y):
    with tf.variable_scope("generator") as scope:

      s_h, s_w = self.output_height, self.output_width

      h0 = tf.nn.relu(linear(z, self.gfc_dim, 'g_h0_lin'))
      h1 = linear(h0, s_h * s_w * self.c_dim, 'g_h1_lin')
      h1 = tf.reshape(h1, [self.batch_size, s_h, s_w, self.c_dim])
      return tf.nn.tanh(h1)

  def dcgan_cond_generator(self, z, y):
    with tf.variable_scope("generator") as scope:
      self.g_bn0 = batch_norm(name='g_bn0')
      self.g_bn1 = batch_norm(name='g_bn1')
      self.g_bn2 = batch_norm(name='g_bn2')

      s_h, s_w = self.output_height, self.output_width

      zy = concat([z, y], 1)
      h0 = tf.nn.relu(linear(zy, self.gfc_dim, 'g_h0_lin'))
      h1 = linear(h0, s_h * s_w * self.c_dim, 'g_h1_lin')
      h1 = tf.reshape(h1, [self.batch_size, s_h, s_w, self.c_dim])

      return tf.nn.tanh(h1)

  def dcgan_sampler(self, z, y):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      self.g_bn0 = batch_norm(name='g_bn0')
      self.g_bn1 = batch_norm(name='g_bn1')
      self.g_bn2 = batch_norm(name='g_bn2')
      self.g_bn3 = batch_norm(name='g_bn3')

      s_h, s_w = self.output_height, self.output_width
      h0 = tf.nn.relu(linear(z, self.gfc_dim, 'g_h0_lin'))
      h1 = linear(h0, s_h * s_w * self.c_dim, 'g_h1_lin')
      h1 = tf.reshape(h1, [self.batch_size, s_h, s_w, self.c_dim])

      return tf.nn.tanh(h1)

  def dcgan_cond_sampler(self, z, y):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      self.g_bn0 = batch_norm(name='g_bn0')
      self.g_bn1 = batch_norm(name='g_bn1')
      self.g_bn2 = batch_norm(name='g_bn2')

      s_h, s_w = self.output_height, self.output_width

      zy = concat([z, y], 1)
      h0 = tf.nn.relu(linear(zy, self.gfc_dim, 'g_h0_lin'))
      h1 = linear(h0, s_h * s_w * self.c_dim, 'g_h1_lin')
      h1 = tf.reshape(h1, [self.batch_size, s_h, s_w, self.c_dim])

      return tf.nn.tanh(h1)

  def discriminator(self, image, y=None, reuse=False):
    if self.conditional:
      return self.dcgan_cond_discriminator(image, y, reuse)
    else:
      return self.dcgan_discriminator(image, y, reuse)

  def generator(self, z, y=None):
    if self.conditional:
      return self.dcgan_cond_generator(z, y)
    else:
      return self.dcgan_generator(z, y)

  def sampler(self, z, y):
    if self.conditional:
      return self.dcgan_cond_sampler(z, y)
    else:
      return self.dcgan_sampler(z, y)

  def load_mnist(self):
    data_dir = os.path.join(self.data_dir, self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_dim = 10
    y_vec = np.zeros((len(y), y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0

    return X/255., y_vec, y_dim, X[0].shape[-1]

  # load images with the specified fname pattern
  def load_image_dataset(self):
    # load image filenames
    images = glob(os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern))

    # find the number of channels
    imreadImg = imread(images[0])
    print(imreadImg.shape)
    # check if image is a non-grayscale image by checking channel number
    if len(imreadImg.shape) >= 3:
      c_dim = imread(images[0]).shape[-1]
    else:
      c_dim = 1

    # one-hot encoding of labels
    labels = pd.read_csv(os.path.join(self.data_dir, self.dataset_name, self.input_fname_labels), sep=' ')
    # 0-index
    y_dim = labels['identity'].max()+1
    print('images.length: {}, y_dim: {}, c_dim: {}, labels.length: {}'.format(len(images), y_dim, c_dim, len(labels)))
    return images, labels, y_dim, c_dim

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      print(" [*] Load SUCCESS")
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      print(" [!] Load failed")
      return False, 0
