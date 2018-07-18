import numpy as np

from config import BATCH_SIZE


def redistribution(idx, total, min_v):
    idx = (idx + 0.0) / (total + 0.0) * 16.0
    return (np.exp(idx - 8.0) / (1.0 + np.exp(idx - 8.0)))


def rescale(reward, rollout_num=1.0):
    reward = np.array(reward)
    x, y = reward.shape
    ret = np.zeros((x, y))
    for i in range(x):
        l = reward[i]
        rescalar = {}
        for s in l:
            rescalar[s] = s
        idxx = 1
        min_s = 1.0
        max_s = 0.0
        for s in rescalar:
            rescalar[s] = redistribution(idxx, len(l), min_s)
            idxx += 1
        for j in range(y):
            ret[i, j] = rescalar[reward[i, j]]
    return ret


def get_reward(model, dis, sess, input_x, rollout_num, dis_dropout_keep_prob, total_epoch, data_loader):
    rewards = []

    pos_num = (total_epoch / 20.0) * 10
    # pos_num = 64
    pos_num = int(pos_num)

    pos_num = min(BATCH_SIZE, pos_num)  # add posnum
    for i in range(rollout_num):
        batch = data_loader.next_batch()
        for given_num in range(1, model.sequence_length // model.step_size):
            real_given_num = given_num * model.step_size
            feed = {model.x: input_x, model.given_num: real_given_num, model.drop_out: 1.0}
            samples = sess.run(model.gen_for_reward, feed)

            samples = np.concatenate((samples, batch[0:pos_num, :]), axis=0)
            # print samples.shape
            feed = {dis.D_input_x: samples, dis.dropout_keep_prob: dis_dropout_keep_prob}
            ypred_for_auc = sess.run(dis.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[given_num - 1] += ypred

        # the last token reward
        samples = np.concatenate((input_x, batch[0:pos_num, :]), axis=0)
        feed = {dis.D_input_x: samples, dis.dropout_keep_prob: 1.0}
        ypred_for_auc = sess.run(dis.ypred_for_auc, feed)
        ypred = np.array([item[1] for item in ypred_for_auc])
        if i == 0:
            rewards.append(ypred)
        else:
            rewards[model.sequence_length // model.step_size - 1] += ypred
    rewards = rescale(np.array(rewards), rollout_num)
    rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
    rewards = rewards[0:BATCH_SIZE, :]
    return rewards
