import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='horse2zebra')
py.arg('--datasets_dir', default='datasets')
py.arg('--output_dir', default=None)
py.arg('--size_h', type=int, default=480)
py.arg('--size_w', type=int, default=640)
py.arg('--batch_size', type=int, default=10)
py.arg('--epochs', type=int, default=262)
py.arg('--lr', type=float, default=2e-5)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--beta_2', type=float, default=0.999)
py.arg('--adversarial_loss_mode', default='wgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='wgan-gp', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=1.0)
py.arg('--color_error_loss_weight', type=float, default=1.0)
py.arg('--class_distribution_loss_weight', type=float, default=1.0)
py.arg('--perceptual_loss_weight', type=float, default=1.0)
py.arg('--sample-interval', type=int, default=100)
args = py.args()
size = (args.size_h,args.size_w)
epoch_decay = args.epochs // 2 # epoch to start decaying learning rate

# output_dir
if args.output_dir is None:
    output_dir = py.join('output', args.dataset)
else:
    output_dir = py.join('output', args.output_dir)

py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'train'), '*.jpg')
A_B_dataset, len_dataset = data.make_zip_dataset(img_paths, args.batch_size, size, training=True, repeat=False)

img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'test'), '*.jpg')
A_B_dataset_test, _ = data.make_zip_dataset(img_paths_test, args.batch_size, size, training=False, repeat=True)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B = module.ResnetGenerator(input_shape=( size[0],size[1],3) )
D_B = module.ConvDiscriminator(input_shape=( size[0],size[1],3) )

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
color_error_loss_fn = tf.losses.mean_squared_error()
class_distribution_loss_fn = tf.keras.losses.KLDivergence()

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        A2B = G_A2B(A, training=True)

        A2B_d_logits = D_B(A2B, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        A2B_color_error_loss = color_error_loss_fn(A2B[...,:2], B[...,:2])

        G_loss = args.perceptual_loss_weight*A2B_g_loss + args.color_error_loss_weight*A2B_color_error_loss

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables))

    return A2B, {
        'A2B_g_loss': A2B_g_loss,
        'A2B_color_error_loss': A2B_color_error_loss
     }


@tf.function
def train_D(B, A2B):
    with tf.GradientTape() as t:
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_loss = args.perceptual_loss_weight*(B_d_loss + A2B_d_loss) + args.gradient_penalty_weight*D_B_gp

    D_grad = t.gradient(D_loss, D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_B.trainable_variables))

    return {
        'B_d_loss': B_d_loss + A2B_d_loss,
        'D_B_gp': D_B_gp
    }


def train_step(A, B):
    A2B, G_loss_dict = train_G(A, B)

    # pool was here

    D_loss_dict = train_D(B, A2B)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    A2B = G_A2B(A, training=False)
    return A2B


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A, B)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % args.sample_interval == 0:
                A_tensor, B_tensor = next(test_iter)
                A2B_tensor = sample(A_tensor)

                A = np.clip(A_tensor.numpy(),-1,1)
                B = np.clip(B_tensor.numpy(),-1,1)
                A2B = np.clip(A2B_tensor.numpy(),-1,1)

                colorImages=[B,A2B]
                for i in range(len(colorImages)):
                    colorImage = colorImages[i]
                    colorImage += 1
                    colorImage /= 2
                    for j in range(colorImage.shape[0]):
                        colorImage[j,...] = hsv2rgb(colorImage[j,...])
                    colorImage *= 2
                    colorImage -= 1

                B,A2B=colorImages

                images = np.concatenate((
                    np.repeat(A.numpy(),3,axis=3),
                    B,
                    A2B
                ), axis=0)

                img = im.immerge(images, n_rows=2)
                im.imwrite(img, py.join(sample_dir, 'iter-%015d.jpg' % G_optimizer.iterations.numpy()))


        # save checkpoint
        checkpoint.save(ep)
