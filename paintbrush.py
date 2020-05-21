import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import imageio
import skimage.transform

from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras import backend as K

CONFIG_MEANS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3)).astype(np.float32)

def content_cost(content, gen):
    m, n_H, n_W, n_C = gen.get_shape().as_list()

    content_reshaped = tf.reshape(content, [m, -1, n_C])
    gen_reshaped = tf.reshape(gen, [m, -1, n_C])

    return (1/(4*n_H*n_W*n_C)) * tf.reduce_sum(tf.square(tf.subtract(content_reshaped, gen_reshaped)))

def style_cost(style, gen):
    n = style.shape[3]
    m = style.shape[1] * style.shape[2]

    # flatten activations to shape (# of channels, height * width)
    flat_style = tf.transpose(tf.reshape(style, [m, n]))
    flat_gen = tf.transpose(tf.reshape(gen, [m, n]))

    gram_style = tf.matmul(flat_style, tf.transpose(flat_style))
    gram_gen = tf.matmul(flat_gen, tf.transpose(flat_gen))

    return 1/(2*n*m)**2 * tf.reduce_sum(tf.square(gram_gen - gram_style))

def cost(output_c, output_s, gen, alpha, beta):
    cost_s = 0
    for i in range(5):
        cost_s += 0.2 * style_cost(output_s[i], gen[i])
    cost_c = content_cost(output_c[5], gen[5])

    return alpha*cost_c + beta*cost_s

@tf.function()
def train_epoch(model, gen):
    with tf.GradientTape() as tape:
        outputs = model(gen)
        loss = cost(outputs_c, outputs_s, outputs, 0.001, 1)

    tf.print("Loss:", loss)
    grad = tape.gradient(loss, gen)
    opt.apply_gradients([(grad, gen)])
    gen.assign(tf.clip_by_value(gen, 0, 1))

def construct_img(tensor):
    return PIL.Image.fromarray((np.array(tensor[0]) + CONFIG_MEANS).astype(np.uint8))

content_src = input("Content Image: ")
style_src = input("Style Image: ")

content_img = imageio.imread(content_src)
content = np.expand_dims(content_img - CONFIG_MEANS, axis=0).astype(np.float32)

style_img = imageio.imread(style_src)
style = skimage.transform.resize(style_img - CONFIG_MEANS, (content.shape[1], content.shape[2]), anti_aliasing=True)
style = np.expand_dims(style, axis=0).astype(np.float32)

# first 5 are for style rep, last is for content
layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1", "block4_conv2"]

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=content[0].shape)
vgg19.trainable = False

output_layers = [vgg19.get_layer(name).output for name in layers]
model = Model(inputs=[vgg19.input], outputs=output_layers)

outputs_c = model(content)
outputs_s = model(style)

epochs = 2000

gen = tf.Variable(content)
opt = tf.optimizers.Adam(learning_rate = 2)

start = time.time()

for epoch in range(epochs):
    start_e = time.time()
    print("Epoch:", epoch+1)
    train_epoch(model, gen)
    end_e = time.time()
    print("Time:", end_e - start_e, "secs\n")

end = time.time()
print("Total Time Elasped:", end - start, "secs")

imageio.imwrite("generated_image.jpg", construct_img(gen))
