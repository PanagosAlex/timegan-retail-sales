import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, Flatten, GaussianNoise
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
import os

Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load the dataset
file_path = 'data/your_dataset.npy'  # Users replace with their data
data = np.load(file_path)

# Split into training and testing data
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Define TimeGAN components
def build_generator(input_dim, hidden_dim):
    inputs = Input(shape=(None, input_dim))
    x = LSTM(hidden_dim, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = GaussianNoise(0.1)(x)
    x = LSTM(hidden_dim, return_sequences=True)(x)
    x = Dense(input_dim, activation="sigmoid")(x)
    return Model(inputs, outputs=x, name="generator")

def build_discriminator(input_dim, hidden_dim):
    inputs = Input(shape=(sequence_length, input_dim))  # Specify fixed input shape
    
    # Add convolutional layers for feature extraction
    x = Conv1D(filters=hidden_dim, kernel_size=3, activation="relu", padding="same")(inputs)
    x = Dropout(0.3)(x)
    x = Conv1D(filters=hidden_dim // 2, kernel_size=3, activation="relu", padding="same")(x)
    x = Dropout(0.3)(x)
    
    # Flatten and pass through dense layers
    x = Flatten()(x)  # Flatten output to ensure fully-defined shape
    x = Dense(hidden_dim, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(hidden_dim // 2, activation="relu")(x)
    
    # Final output layer
    outputs = Dense(1, activation="sigmoid")(x)
    
    return Model(inputs, outputs=outputs, name="discriminator")

def gradient_penalty_loss(real, fake, discriminator):
    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)
    batch_size = tf.shape(real)[0]
    alpha = tf.random.uniform(shape=[batch_size, 1, 1], minval=0.0, maxval=1.0, dtype=tf.float32)
    interpolated = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = tape.gradient(pred, [interpolated])[0]
    penalty = tf.reduce_mean(tf.square(tf.norm(grads, axis=-1) - 1.0))
    return penalty

# Hyperparameters
hidden_dim = 512
batch_size = 16
epochs = 150
learning_rate = 0.00001 
gradient_penalty_weight = 2.0

# Learning rate scheduler
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)
generator_optimizer = Adam(learning_rate=learning_rate_schedule)
discriminator_optimizer = Adam(learning_rate=learning_rate_schedule)

# Losses
reconstruction_loss = tf.keras.losses.MeanSquaredError()
adversarial_loss = tf.keras.losses.BinaryCrossentropy()

# Build models
generator = build_generator(input_dim, hidden_dim)
discriminator = build_discriminator(input_dim, hidden_dim)

# Pretrain Generator as an Autoencoder
autoencoder_optimizer = Adam(learning_rate=0.0001)

@tf.function
def pretrain_generator_step(real_data):
    with tf.GradientTape() as tape:
        reconstructed_data = generator(real_data, training=True)
        reconstruction_loss_value = reconstruction_loss(real_data, reconstructed_data)
    gradients = tape.gradient(reconstruction_loss_value, generator.trainable_variables)
    autoencoder_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return reconstruction_loss_value

# Pretraining loop
pretrain_epochs = 20  # Number of epochs for pretraining
for epoch in range(pretrain_epochs):
    np.random.shuffle(train_data)
    batch_start = 0
    while batch_start < len(train_data):
        batch_end = batch_start + batch_size
        real_batch = train_data[batch_start:batch_end]
        pretrain_loss = pretrain_generator_step(real_batch)
        batch_start += batch_size
    print(f"Pretraining Epoch {epoch + 1}/{pretrain_epochs} - Reconstruction Loss: {pretrain_loss:.4f}")

print("Generator pretraining complete.")

# Training loop with feature-matching loss
@tf.function
def train_step_with_feature_matching(real_data):
    real_data = tf.cast(real_data, tf.float32)
    noise = tf.random.normal([tf.shape(real_data)[0], sequence_length, input_dim], dtype=tf.float32)
    fake_data = generator(noise, training=True)

    # Train discriminator
    with tf.GradientTape() as disc_tape:
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)
        disc_loss = adversarial_loss(tf.ones_like(real_output), real_output) + \
                    adversarial_loss(tf.zeros_like(fake_output), fake_output)
        gradient_penalty = gradient_penalty_loss(real_data, fake_data, discriminator)
        disc_loss += gradient_penalty_weight * gradient_penalty

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    clipped_discriminator_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_discriminator]
    discriminator_optimizer.apply_gradients(zip(clipped_discriminator_gradients, discriminator.trainable_variables))

    # Train generator with feature-matching loss
    with tf.GradientTape() as gen_tape:
        fake_data = generator(noise, training=True)
        fake_output = discriminator(fake_data, training=True)

        # Extract features (intermediate layer outputs) for real and fake data
        real_features = tf.reduce_mean(real_output, axis=0)  # Mean across batch
        fake_features = tf.reduce_mean(fake_output, axis=0)

        # Feature-matching loss
        feature_matching_loss = tf.reduce_mean(tf.square(real_features - fake_features))

        # Combined generator loss
        gen_loss = reconstruction_loss(real_data, fake_data) + \
                   adversarial_loss(tf.ones_like(fake_output), fake_output) + \
                   feature_matching_loss

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    clipped_generator_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_generator]
    generator_optimizer.apply_gradients(zip(clipped_generator_gradients, generator.trainable_variables))

    return disc_loss, gen_loss

# Training loop
for epoch in range(epochs):
    np.random.shuffle(train_data)
    batch_start = 0
    while batch_start < len(train_data):
        batch_end = batch_start + batch_size
        real_batch = train_data[batch_start:batch_end]
        try:
            disc_loss, gen_loss = train_step_with_feature_matching(real_batch)
        except Exception as e:
            print(f"Error encountered at Epoch {epoch + 1}. Stopping training: {e}")
            break
        batch_start += batch_size

    print(f"Epoch {epoch + 1}/{epochs} - Discriminator Loss: {disc_loss:.4f}, Generator Loss: {gen_loss:.4f}")

# Save the trained generator
model_save_path = 'models/timegan_generator.keras'
generator.save(model_save_path)
print(f"Generator saved to {model_save_path}")