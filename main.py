import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("data_dir", "data", "Directory with image datasets [data]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_labels", "labels.txt", "The mapping between images and identities [*]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
flags.DEFINE_boolean("conditional", False, "Model and train conditional GAN")
flags.DEFINE_boolean("dense", False, "Model and train dense GAN")
flags.DEFINE_integer("loss_type", 0, "Loss type [0=cross entropy] 1=logloss 2=wasserstein")
flags.DEFINE_boolean("generate", False, "Generate 100 sample images for testing. Defaults to [False]")
flags.DEFINE_integer("exp_num", 0,
        "[0=original DCGAN],"
        "1=Sigmoid y in DIS,"
        "2=Y at dense layer in DIS,"
        "3=linearizing x & y to concat in DIS,"
        "4=Extra discriminator,"
        "5=Y concat only once before h0 in DIS and GEN,"
        "6=Y concat only once before h0 in DIS and GEN. Invoke unconditional after.")

FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  # y_dim is inferred from the dataset name or the labels file
  if FLAGS.conditional and FLAGS.dataset != 'mnist':
    labels_fname = os.path.join(FLAGS.data_dir, FLAGS.dataset, FLAGS.input_fname_labels)
    if not os.path.exists(labels_fname):
      raise Exception("[!] conditional requires image<->identity labels")

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        z_dim=FLAGS.generate_test_images,
        data_dir=FLAGS.data_dir,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        conditional = FLAGS.conditional,
        dense=FLAGS.dense,
        loss_type=FLAGS.loss_type,
        exp_num=FLAGS.exp_num)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

      if FLAGS.generate:
        generate_samples(sess, dcgan, FLAGS)
        exit()
    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    OPTION = 1
    visualize(sess, dcgan, FLAGS, OPTION)

def generate_samples(sess, dcgan, FLAGS):
    print('Generating samples...')
    n_samples = 2
    labels = []
    eval_classifier_dir = './eval_classifier/testdata'
    ## Generate 2 x 64 (num of batches the model accept) sample images:
    for i in range(n_samples):
        z_sample = np.random.uniform(-1, 1, size=(FLAGS.batch_size , FLAGS.generate_test_images))
        y = np.random.choice(10, FLAGS.batch_size)
        y_one_hot = np.zeros((FLAGS.batch_size, 10))
        y_one_hot[np.arange(FLAGS.batch_size), y] = 1
        labels.append(y)
        ## Generate the samples
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})

        ## Save the samples
        for idx in range(len(samples)):
            sample = samples[idx]
            label = np.where(y_one_hot[idx] > 0)[0][0]
            if not os.path.exists(eval_classifier_dir):
                os.makedirs(eval_classifier_dir)
            image_fname = os.path.join(eval_classifier_dir, str(label) + '_' + str(i) + '_' + str(idx) + '.png')
            scipy.misc.imsave(image_fname, sample.reshape(FLAGS.output_height, FLAGS.output_height, 3))
if __name__ == '__main__':
  tf.app.run()
