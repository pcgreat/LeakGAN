import logging
import os
import pickle
import random

import numpy as np
import tensorflow as tf

from Discriminator import Discriminator
from LeakGANModel import LeakGAN
from config import model_path, SEED, START_TOKEN, SEQ_LENGTH, BATCH_SIZE, dis_filter_sizes, dis_embedding_dim, \
    dis_num_filters, HIDDEN_DIM, EMB_DIM, GOAL_OUT_SIZE, GOAL_SIZE, positive_file, PRE_EPOCH_NUM, generated_num, \
    negative_file, dis_dropout_keep_prob, TOTAL_BATCH
from dataloader import Gen_Data_loader, Dis_dataloader
from rewards import get_reward

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('restore', True, 'Training or testing a model')
flags.DEFINE_boolean('resD', True, 'Training or testing a D model')
flags.DEFINE_boolean('Continue', True, 'Continue unfinished training (default is True)')
flags.DEFINE_string('model', "leakgan", 'Model NAME')

# create logger with 'spam_application'
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('save/experiment.log')
ch = logging.StreamHandler()
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file, train=1):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess, 1.0, train))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch // 20):
        batch = data_loader.next_batch()
        _, g_loss, _, _ = trainable_model.pretrain_step(sess, batch, 1.0)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    #########################################################################################
    #  Epoch Recorder
    #########################################################################################
    epoch_record = 10 * 5
    global_step = 1
    if FLAGS.Continue:
        try:
            with open(model_path + '/epochRecordInfo.pkl', 'rb') as file:
                epoch_record = pickle.load(file)

            with open(model_path + '/global_step.pkl', 'rb') as file:
                global_step = pickle.load(file)

        except Exception as e:
            epoch_record = 10 * 5
            global_step = 1
    else:
        with open(model_path + '/epochRecordInfo.pkl', 'wb') as file:
            pickle.dump(epoch_record, file)

        with open(model_path + '/global_step.pkl', 'wb') as file:
            pickle.dump(global_step, file)

    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    vocab_size = 4839
    dis_data_loader = Dis_dataloader(BATCH_SIZE, SEQ_LENGTH)
    discriminator = Discriminator(SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, dis_emb_dim=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, start_token=START_TOKEN,
                                  goal_out_size=GOAL_OUT_SIZE, step_size=4)
    leakgan = LeakGAN(SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, emb_dim=EMB_DIM, dis_emb_dim=dis_embedding_dim,
                      filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                      batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, start_token=START_TOKEN,
                      goal_out_size=GOAL_OUT_SIZE, goal_size=GOAL_SIZE, step_size=4, D_model=discriminator)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Run as an example
    for i in range(1):
        g = sess.run(leakgan.gen_x, feed_dict={leakgan.drop_out: 0.8, leakgan.train: 1})
        logger.info(("gen_x shape:", g.shape))
        logger.info("epoch: %s" % i)

    generate_samples(sess, leakgan, BATCH_SIZE, generated_num, negative_file, 0)
    gen_data_loader.create_batches(positive_file)
    saver_variables = tf.global_variables()
    saver = tf.train.Saver(saver_variables)
    model = tf.train.latest_checkpoint(model_path)
    import pdb
    pdb.set_trace()
    if FLAGS.restore and model:
        abs_model_path = os.path.join(model_path, FLAGS.model)
        logger.info("Restore model from: %s" % abs_model_path)
        saver.restore(sess, abs_model_path)
    else:
        if FLAGS.resD:
            abs_model_path = os.path.join(model_path, FLAGS.model)
            logger.info("Restore model from: %s" % abs_model_path)
            saver.restore(sess, abs_model_path)

            logger.info('Start pre-training for generator...')
            for epoch in range(PRE_EPOCH_NUM):
                loss = pre_train_epoch(sess, leakgan, gen_data_loader)
                if epoch % 5 == 0:
                    generate_samples(sess, leakgan, BATCH_SIZE, generated_num, negative_file)
                buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(loss)
                logger.info(buffer)
            saver.save(sess, model_path + '/leakgan_pre')
        else:
            logger.info('Start pre-training discriminator...')

            # Train 3 epoch on the generated data and do this for 50 times
            if FLAGS.Continue:
                try:
                    saver.restore(sess, model_path + '/leakgan_pre')
                except Exception:
                    try:
                        saver.restore(sess, model_path + '/leakgan_preD')
                    except Exception:
                        print('new model')

            for i in range((epoch_record + 10 - 1) // 5):
                if epoch_record == 0:
                    break
                for _ in range(epoch_record % 5):
                    generate_samples(sess, leakgan, BATCH_SIZE, generated_num, negative_file, 0)
                    # gen_data_loader.create_batches(positive_file)
                    dis_data_loader.load_train_data(positive_file, negative_file)
                    for _ in range(3):
                        dis_data_loader.reset_pointer()
                        for it in range(dis_data_loader.num_batch):
                            x_batch, y_batch = dis_data_loader.next_batch()
                            feed = {
                                discriminator.D_input_x: x_batch,
                                discriminator.D_input_y: y_batch,
                                discriminator.dropout_keep_prob: dis_dropout_keep_prob
                            }
                            D_loss, _ = sess.run([discriminator.D_loss, discriminator.D_train_op], feed)
                            # print 'D_loss ', D_loss
                            buffer = str(D_loss)
                            logger.info(buffer)
                    leakgan.update_feature_function(discriminator)
                    epoch_record -= 1
                    with open(model_path + '/epochRecordInfo.pkl', 'wb') as file:
                        pickle.dump(epoch_record, file)
                saver.save(sess, model_path + '/leakgan_preD')

                # saver.save(sess, model_path + '/leakgan')
                #  pre-train generator
                logger.info('Start pre-training for generator ...')
                for epoch in range(PRE_EPOCH_NUM // 16):
                    loss = pre_train_epoch(sess, leakgan, gen_data_loader)
                    if epoch % 5 == 0:
                        generate_samples(sess, leakgan, BATCH_SIZE, generated_num, negative_file, 0)
                        logger.info(('pre-train epoch', epoch, 'test_loss', loss))
                    buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(loss)
                    logger.info(buffer)
            saver.save(sess, model_path + '/leakgan_pre')

    gencircle = 1

    logger.info('#########################################################################')
    logger.info('Start Adversarial Training...')
    if FLAGS.Continue:
        saver.restore(sess, tf.train.latest_checkpoint(model_path + '/'))

    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):

            for gi in range(gencircle):
                samples = leakgan.generate(sess, 1.0, 1)
                rewards = get_reward(leakgan, discriminator, sess, samples, 4, dis_dropout_keep_prob, total_batch,
                                     gen_data_loader)
                feed = {leakgan.x: samples, leakgan.reward: rewards, leakgan.drop_out: 1.0}
                _, _, g_loss, w_loss = sess.run(
                    [leakgan.manager_updates, leakgan.worker_updates, leakgan.goal_loss, leakgan.worker_loss],
                    feed_dict=feed)
                logger.info('total_batch: ', total_batch, "  ", g_loss, "  ", w_loss)
        global_step += 1
        # Test
        if total_batch % 10 == 1 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, leakgan, BATCH_SIZE, generated_num, "./save/coco_" + str(total_batch) + ".txt", 0)
            saver.save(sess, model_path + '/leakgan', global_step=global_step)
        if total_batch % 15 == 0:
            for epoch in range(1):
                loss = pre_train_epoch(sess, leakgan, gen_data_loader)
        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, leakgan, BATCH_SIZE, generated_num, negative_file, 0)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.D_input_x: x_batch,
                        discriminator.D_input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    D_loss, _ = sess.run([discriminator.D_loss, discriminator.D_train_op], feed)
                    # print 'D_loss ', D_loss
            leakgan.update_feature_function(discriminator)
        with open(model_path + '/global_step.pkl', 'wb') as file:
            pickle.dump(global_step, file)
        saver.save(sess, model_path + '/leakgan')


if __name__ == '__main__':
    main()
