""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)


logs_path = '/tmp/tensorflow_logs/example/'




def print_array_properties(input_array):
		input_array = np.array(input_array)
		print("input array shape is:",input_array.shape)
		print("input array type is:",input_array.dtype)
		print("input array mean is:",np.mean(input_array))
		print("input array max is:",np.max(input_array))
		print("input array min is:",np.min(input_array))
		print("input array variance is:",np.var(input_array))



# def visualize_labeled_images(images, labels=None, max_outputs=3, name="image"):
#     def _visualize_image(image):
#         # Do the actual drawing in python
#         fig = plt.figure(figsize=(3, 3), dpi=80)
#         ax = fig.add_subplot(111)
#         ax.imshow(image[::-1,...])
#         fig.canvas.draw()
#
#         # Write the plot as a memory file.
#         buf = io.BytesIO()
#         data = fig.savefig(buf, format="png")
#         buf.seek(0)
#
#         # Read the image and convert to numpy array
#         img = PIL.Image.open(buf)
#         return np.array(img.getdata()).reshape(img.size[0], img.size[1], -1)
#
#     def _visualize_images(images, labels):
#         # Only display the given number of examples in the batch
#         outputs = []
#         for i in range(images.shape[0]):
#             output = _visualize_image(images[i], labels[i])
#             outputs.append(output)
#         return np.array(outputs, dtype=np.uint8)
#
#     # Run the python op.
#     figs = tf.py_func(_visualize_images, [images], tf.uint8)
#     return tf.summary.image(name, figs)
#



# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
with tf.name_scope('RMS'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(y_pred, y_true)
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))





# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)


# Create a summary to monitor accuracy tensor
image_simple = tf.summary.image('sample', tf.reshape(y_pred, [-1, 28, 28, 1]), 3)
# img_place = tf.placeholder(tf.uint8,[2,784],name='image')
# image = tf.summary.image('sample', tf.reshape(img_place,[-1, 28, 28, 1]), 3)


# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()


# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l, summary = sess.run([optimizer, loss, merged_summary_op], feed_dict={X: batch_x})

        # Display logs per step
        if i % display_step == 0 or i == 1:
            sess.run([image_simple],feed_dict={X: batch_x[0:1]} )
            # image = np.reshape(batch_x[0:2], [2, 28, 28, 1])
            # image = image * 255
            # image  = image.astype(int)
            #sess.run([image],feed_dict={img_place:batch_x[0:2]})
            print('Step %i: Minibatch Loss: %f' % (i, l))

        # Write logs at every iteration
        summary_writer.add_summary(summary, i)


    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
