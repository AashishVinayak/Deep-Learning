##################   THINGS I'VE TRIED    ###################


""" 
_________________________________________________________________________________________

    # loss and optimization

    # use the discriminator to differentiate between real and fake images
    real_prob = discriminator(real_x)
    fake_prob = discriminator(g_output)
    
    # loss function for discriminator
    #d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_prob, labels=tf.zeros_like(real_prob)) +
    #                       tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_prob, labels=tf.ones_like(fake_prob)))
    
    d_loss = model.compile(loss="mean_squared_error", optmizer="adam")
    
    # loss function for generator
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_prob, labels=tf.ones_like(fake_prob)))
    
    # optimizer for discriminator
    d_ops = tf.train.AdamOptimizer(0.001).minimize(d_loss)
    # optimizer for generator
    g_ops = tf.train.AdamOptimizer(0.001).minimize(g_loss)
    
    
_________________________________________________________________________________________
tf.reset_default_graph()

# input and labels placeholders
fake_x = tf.placeholder(tf.float32, [None, 32,32,3])  
real_x = tf.placeholder(tf.float32, [None, x_train.shape[1], x_train.shape[2], x_train.shape[3]])

# generator network weights
gen_wt = {'w_c1': tf.get_variable('w_c1', [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer()),
              'w_c2': tf.get_variable('w_c2', [3,3,64,128], initializer=tf.contrib.layers.xavier_initializer()),
              'w_c3': tf.get_variable('w_c3', [3,3,128,128], initializer=tf.contrib.layers.xavier_initializer()),
              'w_c4': tf.get_variable('w_c4', [3,3,128,1], initializer=tf.contrib.layers.xavier_initializer())
    
}

# discriminator network weights
dis_wt = {'d_c1': tf.get_variable('d_c1', [3,3,1,32], initializer=tf.contrib.layers.xavier_initializer()), 
              'd_c2': tf.get_variable('d_c2', [3,3,32,64], initializer=tf.contrib.layers.xavier_initializer()),
              'd_c3': tf.get_variable('d_c3', [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer()),
              'd_c4': tf.get_variable('d_c4', [3,3,64,1], initializer=tf.contrib.layers.xavier_initializer())
}


_________________________________________________________________________________________
# generator network
G_input = Input(shape=(32,32,3,))
generator = Conv2D(64, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(G_input)
generator = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(generator)
generator = BatchNormalization()(generator)
generator = Conv2D(128, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(generator)
generator = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(generator)
generator = BatchNormalization()(generator)
generator = Conv2D(256, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(generator)
generator = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(generator)
generator = BatchNormalization()(generator)

generator = UpSampling2D(size=(2,2))(generator)
generator = Conv2D(128, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(generator)
generator = UpSampling2D(size=(2,2))(generator)
generator = Conv2D(64, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(generator)
generator = UpSampling2D(size=(2,2))(generator)
generator = Conv2D(3, kernel_size=(3,3), strides=(1,1), activation="relu", padding="same")(generator)

G_model = Model(inputs=G_input, outputs=generator)
G_model.compile(loss="mean_squared_error", optimizer="adam")


# discriminator network
# using a pretrained model
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=x_train.shape[1:])
for layer in vgg16.layers:
    layer.trainable = False

discriminator = vgg16.output
#discriminator = concatenate([generator, vgg16.input])
#################  concat vgg16 input with generator
discriminator = Flatten()(discriminator)
discriminator = Dense(128, activation='relu')(discriminator)
discriminator = Dropout(0.75)(discriminator)
discriminator = Dense(32, activation='relu')(discriminator)
discriminator = Dense(1, activation='sigmoid')(discriminator)
    
discriminator_model = Model(inputs=D_input, outputs=discriminator)
discriminator_model.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["accuracy"]) 
discriminator_model.summary()


_________________________________________________________________________________________
        g_conv1 = tf.layers.conv2d(filters=64, inputs=G_input, kernel_size=(5,5), strides=(1,1), activation=tf.nn.leaky_relu, padding="same")
        g_pool1 = tf.layers.max_pooling2d(g_conv1, pool_size=(2,2) ,strides=(2,2), padding="same")
        g_norm1 = tf.layers.batch_normalization(g_pool1)
        
        g_conv2 = tf.layers.conv2d(g_norm1, filters=128, kernel_size=(5,5), strides=(1,1), activation=tf.nn.leaky_relu, padding="same")
        g_pool2 = tf.layers.max_pooling2d(g_conv2, pool_size=(2,2) ,strides=(2,2), padding="same")
        g_norm2 = tf.layers.batch_normalization(g_pool2)
        
        g_conv3 = tf.layers.conv2d(g_norm2, filters=256, kernel_size=(5,5), strides=(1,1), activation=tf.nn.leaky_relu, padding="same")
        g_pool3 = tf.layers.max_pooling2d(g_conv3, pool_size=(2,2) ,strides=(2,2), padding="same")
        g_norm3 = tf.layers.batch_normalization(g_pool3)
    
        g_conv4 = tf.layers.conv2d_transpose(g_norm3, 128, kernel_size=(2,2), strides=(2,2), activation=tf.nn.leaky_relu, padding="same")
        g_resi4 = tf.image.resize_images(g_conv4, size = (7,7), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        g_norm4 = tf.layers.batch_normalization(g_resi4)
        
        g_conv5 = tf.layers.conv2d_transpose(g_norm4, 64, kernel_size=(2,2), strides=(2,2), activation=tf.nn.leaky_relu, padding="same")
        g_norm5 = tf.layers.batch_normalization(g_conv5)
        
        g_conv6 = tf.layers.conv2d_transpose(g_norm5, 64, kernel_size=(3,3), strides=(2,2), activation=tf.nn.leaky_relu, padding="same")
        g_norm6 = tf.layers.batch_normalization(g_conv6)
        
        generator = tf.layers.conv2d(g_norm6, 1, kernel_size=(3,3), strides=(1,1), activation="tanh", padding="same")
        

_________________________________________________________________________________________
with tf.device("/gpu:0"):
    def Discriminator(D_input, train_flag):
        global print_flag_dis
        """
        """
            D_input    : Tensor holds image data
            train_flag : if True, the layers will be trainable, untrainable otherwise
        """
        """
        d_conv1 = tf.layers.conv2d(filters=128, inputs=D_input, trainable=train_flag, kernel_size=(3,3), strides=(1,1), 
                                   activation=tf.nn.leaky_relu(alpha=,0.2), padding="same")
        d_conv2 = tf.layers.conv2d(d_conv1, 64, kernel_size=(3,3), trainable=train_flag, strides=(1,1), 
                                   activation=tf.nn.leaky_relu(alpha=,0.2), padding="same")
        d_pool1 = tf.layers.max_pooling2d(d_conv2, pool_size=(2,2) ,strides=(2,2), padding="same")
        d_norm1 = tf.layers.batch_normalization(d_pool1)
        
        d_conv3 = tf.layers.conv2d(d_norm1, 128, kernel_size=(3,3), trainable=train_flag, strides=(1,1), 
                                   activation=tf.nn.leaky_relu(alpha=,0.2), padding="same")
        d_conv4 = tf.layers.conv2d(d_conv3, 128, kernel_size=(3,3), trainable=train_flag, strides=(1,1), 
                                   activation=(alpha=,0.2), padding="same")
        d_pool2 = tf.layers.max_pooling2d(d_conv4, pool_size=(2,2) ,strides=(2,2), padding="same")
        d_norm2 = tf.layers.batch_normalization(d_pool2)
        
        d_conv5 = tf.layers.conv2d(d_norm2, 256, kernel_size=(3,3), trainable=train_flag, strides=(1,1), 
                                   activation=tf.nn.leaky_relu(alpha=,0.2), padding="same")
        d_conv6 = tf.layers.conv2d(d_conv5, 256, kernel_size=(3,3), trainable=train_flag, strides=(1,1), 
                                   activation=tf.nn.leaky_relu(alpha=,0.2), padding="same")
        d_pool3 = tf.layers.max_pooling2d(d_conv6, pool_size=(2,2) ,strides=(2,2), padding="same")
        d_norm3 = tf.layers.batch_normalization(d_pool3)
        
        flat = tf.layers.flatten(d_norm3)
        d_dense1 = tf.layers.dense(flat, 128, trainable=train_flag, activation="tanh")
    
        d_dense2 = tf.layers.dense(d_dense1, 64, trainable=train_flag, activation="tanh")
    
        d_dense3 = tf.layers.dense(d_dense2, 32, trainable=train_flag, activation="tanh")   
    
        discriminator = tf.layers.dense(d_dense1, 1, trainable=train_flag, activation="tanh")
        
        if print_flag_dis == True:
            print("input            : ", D_input.shape)
            print("conv_block_1     : ", d_pool1.shape)
            print("conv_block_2     : ", d_pool2.shape)
            print("conv_block_3     : ", d_pool3.shape)
            print("dense            : ", d_dense1.shape)
            print("dense 2          : ", d_dense2.shape)
            print("dense 3          : ", d_dense3.shape)
            print("Discrimin output : ", discriminator.shape)
            print_flag_dis = False
            
        return discriminator


_________________________________________________________________________________________
tf.reset_default_graph()

D_in = tf.placeholder(tf.float32, [None, 28,28,1])
X_in = tf.placeholder(tf.float32, [None, 28,28,1])
G_in = tf.placeholder(tf.float32, [None, 512])
D_label = tf.placeholder(tf.float32, [None, 1])
G_label = tf.placeholder(tf.float32, [None, 1])

with tf.device("/gpu:0"):
    def Generator(z):
        global print_flag_gen
        
        #    z    : Tensor holds noise sample
        
        
        input_reshape = tf.reshape(z, [tf.shape(z)[0], 4,4,int(z.shape[1]//(4*4))])
        g_conv1 = tf.layers.conv2d_transpose(input_reshape, 1024, kernel_size=(2,2),strides=(1,1),activation="relu",padding="same")
        g_norm1 = tf.layers.batch_normalization(g_conv1)
    
        g_conv2 = tf.layers.conv2d_transpose(g_norm1, 512, kernel_size=(5,5), strides=(2,2), activation=tf.nn.relu, padding="same")
        g_resi1 = tf.image.resize_images(g_conv2, size = (7,7), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        g_norm2 = tf.layers.batch_normalization(g_resi1)
    
        g_conv3 = tf.layers.conv2d_transpose(g_norm2, 256, kernel_size=(5,5), strides=(2,2), activation=tf.nn.relu, padding="same")
        g_norm3 = tf.layers.batch_normalization(g_conv3)
    
        g_conv4 = tf.layers.conv2d_transpose(g_norm3, 128, kernel_size=(5,5), strides=(2,2), activation=tf.nn.relu, padding="same")
        g_norm4 = tf.layers.batch_normalization(g_conv4)
    
        g_conv5 = tf.layers.conv2d_transpose(g_norm4, 1, kernel_size=(5,5), strides=(1,1), activation=tf.nn.tanh, padding="same")
    
        if print_flag_gen == True:
            print("Input            : ",z.shape)
            print("g_conv1          : ",g_norm1.shape)
            print("g_conv2          : ",g_norm2.shape)
            print("g_conv3          : ",g_norm3.shape)
            print("g_conv4          : ",g_norm4.shape)
            print("generator        : ",g_conv5.shape)
            print_flag_gen = False
        
        return g_conv5


_________________________________________________________________________________________
# generate fake images from noise
        #fake_imgs = sess.run(G_z, {G_in:noise})
        
        # concatenating fake and real images
        disc_input = np.concatenate((real_imgs, fake_imgs), axis=0)
        # labels for discriminator
        # using soft and noisy labels instead of 0 and 1
        disc_labels = []
        for i in range(len(disc_input)):
            if i < len(disc_input)//2:
                # ~0 for real images
                disc_labels.append(np.random.uniform(0,0.1))
            else:
                # ~1 for real images
                disc_labels.append(np.random.uniform(0.9,1.0))
        disc_labels = np.asarray(disc_labels).reshape(2*sample_size,1)
        
"""