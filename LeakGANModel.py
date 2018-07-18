import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class LeakGAN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 emb_dim, dis_emb_dim, filter_sizes, num_filters, batch_size, hidden_dim, start_token, goal_out_size,
                 goal_size, step_size, D_model, LSTMlayer_num=1, l2_reg_lambda=0.0, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.LSTMlayer_num = LSTMlayer_num
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate
        self.num_filters_total = sum(self.num_filters)
        self.grad_clip = 5.0
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size
        self.step_size = step_size
        self.D_model = D_model
        self.FeatureExtractor_unit = self.D_model.FeatureExtractor_unit

        self.scope = self.D_model.feature_scope
        self.worker_params = []
        self.manager_params = []

        self.epis = 0.65
        self.tem = 0.8

        self.build_placeholder(self.batch_size, self.step_size, self.sequence_length)

        # Build vars for Worker and Manager
        with tf.variable_scope('Worker'):
            self.g_embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.emb_dim], stddev=0.1))
            self.worker_params.append(self.g_embeddings)
            self.g_worker_recurrent_unit = self.create_Worker_recurrent_unit(
                self.worker_params)  # maps h_tm1 to h_t for generator
            self.g_worker_output_unit = self.create_Worker_output_unit(
                self.worker_params)  # maps h_t to o_t (output token logits)
            self.W_workerOut_change = tf.Variable(tf.random_normal([self.vocab_size, self.goal_size], stddev=0.1))

            self.g_change = tf.Variable(tf.random_normal([self.goal_out_size, self.goal_size], stddev=0.1))
            self.worker_params.extend([self.W_workerOut_change, self.g_change])

            self.h0_worker = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0_worker = tf.stack([self.h0_worker, self.h0_worker])

        with tf.variable_scope('Manager'):
            self.g_manager_recurrent_unit = self.create_Manager_recurrent_unit(
                self.manager_params)  # maps h_tm1 to h_t for generator
            self.g_manager_output_unit = self.create_Manager_output_unit(
                self.manager_params)  # maps h_t to o_t (output token logits)
            self.h0_manager = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0_manager = tf.stack([self.h0_manager, self.h0_manager])

            self.goal_init = tf.get_variable("goal_init",
                                             initializer=tf.truncated_normal([self.batch_size, self.goal_out_size],
                                                                             stddev=0.1))
            self.manager_params.extend([self.goal_init])

        # setup padding array
        self.padding_array = tf.constant(-1, shape=[self.batch_size, self.sequence_length], dtype=tf.int32)

        with tf.name_scope("roll_out"):
            self.gen_for_reward = self.rollout(self.x, self.given_num)

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x),
                                            perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=1, dynamic_size=True, infer_shape=True,
                                             clear_after_read=False)

        goal = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                            dynamic_size=False, infer_shape=True, clear_after_read=False)

        feature_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length + 1,
                                                     dynamic_size=False, infer_shape=True, clear_after_read=False)
        real_goal_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length // self.step_size,
                                                       dynamic_size=False, infer_shape=True, clear_after_read=False)

        gen_real_goal_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                           dynamic_size=False, infer_shape=True, clear_after_read=False)

        gen_o_worker_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length // self.step_size,
                                                          dynamic_size=False, infer_shape=True, clear_after_read=False)

        _g_recurrence_f = self.get_g_recurrence_f()

        _, _, _, _, self.gen_o, self.gen_x, _, _, _, _, self.gen_real_goal_array, self.gen_o_worker_array = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11: i < self.sequence_length,
            body=_g_recurrence_f,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0_worker, self.h0_manager,
                       gen_o, gen_x, goal, tf.zeros([self.batch_size, self.goal_out_size]), self.goal_init, step_size,
                       gen_real_goal_array, gen_o_worker_array), parallel_iterations=1)

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        self.gen_real_goal_array = self.gen_real_goal_array.stack()  # seq_length x batch_size x goal
        self.gen_real_goal_array = tf.transpose(self.gen_real_goal_array,
                                                perm=[1, 0, 2])  # batch_size x seq_length x goal

        self.gen_o_worker_array = self.gen_o_worker_array.stack()  # seq_length x batch_size* vocab*goal

        self.gen_o_worker_array = tf.transpose(self.gen_o_worker_array,
                                               perm=[1, 0, 2, 3])  # batch_size x seq_length * vocab*goal

        sub_feature = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length // self.step_size,
                                                   dynamic_size=False, infer_shape=True, clear_after_read=False)

        all_sub_features = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                        dynamic_size=False, infer_shape=True, clear_after_read=False)
        all_sub_goals = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                     dynamic_size=False, infer_shape=True, clear_after_read=False)

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False,
                                                     infer_shape=True)
        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        # build preTrain
        preTrain_f = self.get_preTrain_f(self.step_size, ta_emb_x)

        _, _, self.g_predictions, _, _, _, _, _, self.feature_array, self.real_goal_array, self.sub_feature, self.all_sub_features, self.all_sub_goals = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12: i < self.sequence_length + 1,
            body=preTrain_f,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                g_predictions,
                self.h0_worker,
                self.x, self.h0_manager, tf.zeros([self.batch_size, self.goal_out_size]), self.goal_init, feature_array,
                real_goal_array, sub_feature, all_sub_features, all_sub_goals),
            parallel_iterations=1)

        self.sub_feature = self.sub_feature.stack()  # seq_length x batch_size x num_filter
        self.sub_feature = tf.transpose(self.sub_feature, perm=[1, 0, 2])

        self.real_goal_array = self.real_goal_array.stack()
        self.real_goal_array = tf.transpose(self.real_goal_array, perm=[1, 0, 2])
        print(self.real_goal_array.shape)
        print(self.sub_feature.shape)
        self.pretrain_goal_loss = -tf.reduce_sum(1 - tf.losses.cosine_distance(tf.nn.l2_normalize(self.sub_feature, 2),
                                                                               tf.nn.l2_normalize(self.real_goal_array,
                                                                                                  2), 2)
                                                 ) / (self.sequence_length * self.batch_size // self.step_size)

        with tf.name_scope("Manager_PreTrain_update"):
            pretrain_manager_opt = tf.train.AdamOptimizer(self.learning_rate)

            self.pretrain_manager_grad, _ = tf.clip_by_global_norm(
                tf.gradients(self.pretrain_goal_loss, self.manager_params), self.grad_clip)
            self.pretrain_manager_updates = pretrain_manager_opt.apply_gradients(
                list(zip(self.pretrain_manager_grad, self.manager_params)))
        # self.real_goal_array = self.real_goal_array.stack()

        self.g_predictions = tf.transpose(self.g_predictions.stack(),
                                          perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.cross_entropy = tf.reduce_sum(
            self.g_predictions * tf.log(tf.clip_by_value(self.g_predictions, 1e-20, 1.0))) / (
                                     self.batch_size * self.sequence_length * self.vocab_size)

        self.pretrain_worker_loss = -tf.reduce_sum(
            tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
            )
        ) / (self.sequence_length * self.batch_size)

        with tf.name_scope("Worker_PreTrain_update"):
            # training updates
            pretrain_worker_opt = tf.train.AdamOptimizer(self.learning_rate)

            self.pretrain_worker_grad, _ = tf.clip_by_global_norm(
                tf.gradients(self.pretrain_worker_loss, self.worker_params), self.grad_clip)
            self.pretrain_worker_updates = pretrain_worker_opt.apply_gradients(
                list(zip(self.pretrain_worker_grad, self.worker_params)))

        self.goal_loss = -tf.reduce_sum(tf.multiply(self.reward, 1 - tf.losses.cosine_distance(
            tf.nn.l2_normalize(self.sub_feature, 2), tf.nn.l2_normalize(self.real_goal_array, 2), 2)
                                                    )) / (self.sequence_length * self.batch_size / self.step_size)

        with tf.name_scope("Manager_update"):
            manager_opt = tf.train.AdamOptimizer(self.learning_rate)

            self.manager_grad, _ = tf.clip_by_global_norm(
                tf.gradients(self.goal_loss, self.manager_params), self.grad_clip)
            self.manager_updates = manager_opt.apply_gradients(
                list(zip(self.manager_grad, self.manager_params)))

        self.all_sub_features = self.all_sub_features.stack()
        self.all_sub_features = tf.transpose(self.all_sub_features, perm=[1, 0, 2])

        self.all_sub_goals = self.all_sub_goals.stack()
        self.all_sub_goals = tf.transpose(self.all_sub_goals, perm=[1, 0, 2])
        # self.all_sub_features = tf.nn.l2_normalize(self.all_sub_features, 2)
        self.Worker_Reward = 1 - tf.losses.cosine_distance(tf.nn.l2_normalize(self.all_sub_features, 2),
                                                           tf.nn.l2_normalize(self.all_sub_goals, 2), 2)
        # print self.Worker_Reward.shape
        self.worker_loss = -tf.reduce_sum(
            tf.multiply(self.Worker_Reward,
                        tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                            tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.vocab_size]), 1e-20, 1.0))
                        )
        ) / (self.sequence_length * self.batch_size)

        with tf.name_scope("Worker_update"):
            # training updates
            worker_opt = tf.train.AdamOptimizer(self.learning_rate)
            self.worker_grad, _ = tf.clip_by_global_norm(
                tf.gradients(self.worker_loss, self.worker_params), self.grad_clip)
            self.worker_updates = worker_opt.apply_gradients(
                list(zip(self.worker_grad, self.worker_params)))

    def get_preTrain_f(self, step_size, ta_emb_x):
        def preTrain(i, x_t, g_predictions, h_tm1, input_x, h_tm1_manager, last_goal, real_goal, feature_array,
                     real_goal_array, sub_feature, all_sub_features, all_sub_goals):
            ## padding sentence by -1
            cur_sen = \
                tf.split(tf.concat([tf.split(input_x, [i, self.sequence_length - i], 1)[0], self.padding_array], 1),
                         [self.sequence_length, i], 1)[0]  # padding sentence
            with tf.variable_scope(self.scope):
                feature = self.FeatureExtractor_unit(cur_sen, self.drop_out)
            feature_array = feature_array.write(i, feature)

            real_goal_array = tf.cond(i > 0, lambda: real_goal_array,
                                      lambda: real_goal_array.write(0, self.goal_init))
            h_t_manager = self.g_manager_recurrent_unit(feature, h_tm1_manager)
            sub_goal = self.g_manager_output_unit(h_t_manager)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)

            h_t_Worker = tf.cond(i > 0, lambda: self.g_worker_recurrent_unit(x_t, h_tm1),
                                 lambda: h_tm1)  # hidden_memory_tuple
            o_t_Worker = self.g_worker_output_unit(h_t_Worker)  # batch x vocab , logits not prob
            o_t_Worker = tf.reshape(o_t_Worker, [self.batch_size, self.vocab_size, self.goal_size])

            real_sub_goal = tf.cond(i > 0, lambda: tf.add(last_goal, sub_goal),
                                    lambda: real_goal)
            all_sub_goals = tf.cond(i > 0, lambda: all_sub_goals.write(i - 1, real_goal),
                                    lambda: all_sub_goals)

            w_g = tf.matmul(real_goal, self.g_change)  # batch x goal_size
            w_g = tf.nn.l2_normalize(w_g, 1)
            w_g = tf.expand_dims(w_g, 2)  # batch x goal_size x 1

            x_logits = tf.matmul(o_t_Worker, w_g)
            x_logits = tf.squeeze(x_logits)

            g_predictions = tf.cond(i > 0, lambda: g_predictions.write(i - 1, tf.nn.softmax(x_logits)),
                                    lambda: g_predictions)

            sub_feature = tf.cond(((((i) % step_size) > 0)),
                                  lambda: sub_feature,
                                  lambda: (tf.cond(i > 0, lambda: sub_feature.write(i // step_size - 1,
                                                                                    tf.subtract(feature,
                                                                                                feature_array.read(
                                                                                                    i - step_size))),
                                                   lambda: sub_feature)))

            all_sub_features = tf.cond(i > 0, lambda: tf.cond((i % step_size) > 0, lambda: all_sub_features.write(i - 1,
                                                                                                                  tf.subtract(
                                                                                                                      feature,
                                                                                                                      feature_array.read(
                                                                                                                          i - i % step_size))), \
                                                              lambda: all_sub_features.write(i - 1, tf.subtract(feature,
                                                                                                                feature_array.read(
                                                                                                                    i - step_size)))),
                                       lambda: all_sub_features)

            real_goal_array = tf.cond(((i) % step_size) > 0, lambda: real_goal_array,
                                      lambda: tf.cond((i) // step_size < self.sequence_length // step_size,
                                                      lambda: tf.cond(i > 0,
                                                                      lambda: real_goal_array.write((i) // step_size,
                                                                                                    real_sub_goal),
                                                                      lambda: real_goal_array),
                                                      lambda: real_goal_array))
            x_tp1 = tf.cond(i > 0, lambda: ta_emb_x.read(i - 1),
                            lambda: x_t)

            return i + 1, x_tp1, g_predictions, h_t_Worker, input_x, h_t_manager, \
                   tf.cond(((i) % step_size) > 0, lambda: real_sub_goal,
                           lambda: tf.constant(0.0, shape=[self.batch_size, self.goal_out_size])), \
                   tf.cond(((i) % step_size) > 0, lambda: real_goal, lambda: real_sub_goal), \
                   feature_array, real_goal_array, sub_feature, all_sub_features, all_sub_goals

        return preTrain

    def get_g_recurrence_f(self):
        # build g_recurrence
        def _g_recurrence(i, x_t, h_tm1, h_tm1_manager, gen_o, gen_x, goal, last_goal, real_goal, step_size,
                          gen_real_goal_array, gen_o_worker_array):
            ## padding sentence by -1
            cur_sen = tf.cond(i > 0, lambda:
            tf.split(tf.concat([tf.transpose(gen_x.stack(), perm=[1, 0]), self.padding_array], 1),
                     [self.sequence_length, i], 1)[0], lambda: self.padding_array)
            with tf.variable_scope(self.scope):
                feature = self.FeatureExtractor_unit(cur_sen, self.drop_out)
            h_t_Worker = self.g_worker_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t_Worker = self.g_worker_output_unit(h_t_Worker)  # batch x vocab , logits not prob
            o_t_Worker = tf.reshape(o_t_Worker, [self.batch_size, self.vocab_size, self.goal_size])

            h_t_manager = self.g_manager_recurrent_unit(feature, h_tm1_manager)
            sub_goal = self.g_manager_output_unit(h_t_manager)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)
            goal = goal.write(i, sub_goal)

            real_sub_goal = tf.add(last_goal, sub_goal)

            w_g = tf.matmul(real_goal, self.g_change)  # batch x goal_size
            w_g = tf.nn.l2_normalize(w_g, 1)
            gen_real_goal_array = gen_real_goal_array.write(i, real_goal)

            w_g = tf.expand_dims(w_g, 2)  # batch x goal_size x 1

            gen_o_worker_array = gen_o_worker_array.write(i, o_t_Worker)

            x_logits = tf.matmul(o_t_Worker, w_g)
            x_logits = tf.squeeze(x_logits)

            log_prob = tf.log(tf.nn.softmax(
                tf.cond(i > 1, lambda: tf.cond(self.train > 0, lambda: self.tem, lambda: 1.5), lambda: 1.5) * x_logits))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            with tf.control_dependencies([cur_sen]):
                gen_x = gen_x.write(i, next_token)  # indices, batch_size
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                             tf.nn.softmax(x_logits)), 1))  # [batch_size] , prob
            return i + 1, x_tp1, h_t_Worker, h_t_manager, gen_o, gen_x, goal, \
                   tf.cond(((i + 1) % step_size) > 0, lambda: real_sub_goal,
                           lambda: tf.constant(0.0, shape=[self.batch_size, self.goal_out_size])) \
                , tf.cond(((i + 1) % step_size) > 0, lambda: real_goal,
                          lambda: real_sub_goal), step_size, gen_real_goal_array, gen_o_worker_array

        return _g_recurrence

    def build_placeholder(self, batch_size, step_size, sequence_length):
        with tf.variable_scope('place_holder'):
            self.x = tf.placeholder(tf.int32, shape=[batch_size,
                                                     sequence_length])  # sequence of tokens generated by generator
            self.reward = tf.placeholder(tf.float32, shape=[batch_size,
                                                            sequence_length // step_size])  # sequence of tokens generated by generator
            self.given_num = tf.placeholder(tf.int32, name="given_num")
            self.drop_out = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.train = tf.placeholder(tf.int32, None, name="train")

    def rollout(self, input_x, given_num):
        with tf.device("/cpu:0"):
            processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, input_x),
                                       perm=[1, 0, 2])  # seq_length x batch_size x emb_dim
        ta_emb_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(processed_x)

        # Next is rollout
        gen_for_reward = tensor_array_ops.TensorArray(dtype=tf.int32, size=1, dynamic_size=True, infer_shape=True,
                                                      clear_after_read=False)
        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(input_x, perm=[1, 0]))

        # When current index i < given_num, use the provided tokens as the input at each time step
        def _g_recurrence_1(i, x_t, input_x, gen_x, h_tm1, h_tm1_manager, last_goal, real_goal, give_num):
            cur_sen = \
                tf.split(tf.concat([tf.split(input_x, [i, self.sequence_length - i], 1)[0], self.padding_array], 1),
                         [self.sequence_length, i], 1)[0]
            with tf.variable_scope(self.scope):
                feature = self.FeatureExtractor_unit(cur_sen, self.drop_out)

            h_t_manager = self.g_manager_recurrent_unit(feature, h_tm1_manager)
            sub_goal = self.g_manager_output_unit(h_t_manager)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)

            h_t_Worker = tf.cond(i > 0, lambda: self.g_worker_recurrent_unit(x_t, h_tm1),
                                 lambda: h_tm1)  # hidden_memory_tuple

            real_sub_goal = tf.cond(i > 0, lambda: tf.add(last_goal, sub_goal), lambda: real_goal)
            # real_goal_array = real_goal_array.write(i, real_sub_goal)

            x_tp1 = tf.cond(i > 0, lambda: ta_emb_x.read(i - 1), lambda: x_t)

            # hidden_memory_tuple
            with tf.control_dependencies([cur_sen]):
                gen_x = tf.cond(i > 0, lambda: gen_x.write(i - 1, ta_x.read(i - 1)), lambda: gen_x)
            return i + 1, x_tp1, input_x, gen_x, h_t_Worker, h_t_manager, \
                   tf.cond(((i) % self.step_size) > 0, lambda: real_sub_goal,
                           lambda: tf.constant(0.0, shape=[self.batch_size, self.goal_out_size])), \
                   tf.cond(((i) % self.step_size) > 0, lambda: real_goal, lambda: real_sub_goal), give_num

        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        def _g_recurrence_2(i, x_t, gen_x, h_tm1, h_tm1_manager, last_goal, real_goal):
            # with tf.device('/cpu:0'):
            cur_sen = tf.cond(i > 0, lambda:
            tf.split(tf.concat([tf.transpose(gen_x.stack(), perm=[1, 0]), self.padding_array], 1),
                     [self.sequence_length, i - 1], 1)[0], lambda: self.padding_array)
            with tf.variable_scope(self.scope):
                feature = self.FeatureExtractor_unit(cur_sen, self.drop_out)
            h_t_Worker = self.g_worker_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t_Worker = self.g_worker_output_unit(h_t_Worker)  # batch x vocab , logits not prob

            o_t_Worker = tf.reshape(o_t_Worker, [self.batch_size, self.vocab_size, self.goal_size])

            h_t_manager = self.g_manager_recurrent_unit(feature, h_tm1_manager)
            sub_goal = self.g_manager_output_unit(h_t_manager)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)

            real_sub_goal = tf.add(last_goal, sub_goal)
            w_g = tf.matmul(real_goal, self.g_change)  # batch x goal_size
            w_g = tf.nn.l2_normalize(w_g, 1)
            w_g = tf.expand_dims(w_g, 2)  # batch x goal_size x 1

            x_logits = tf.matmul(o_t_Worker, w_g)
            x_logits = tf.squeeze(x_logits)

            log_prob = tf.log(tf.nn.softmax(x_logits))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            with tf.control_dependencies([cur_sen]):
                gen_x = gen_x.write(i - 1, next_token)  # indices, batch_size
            return i + 1, x_tp1, gen_x, h_t_Worker, h_t_manager, \
                   tf.cond(((i) % self.step_size) > 0, lambda: real_sub_goal,
                           lambda: tf.constant(0.0, shape=[self.batch_size, self.goal_out_size])), \
                   tf.cond(((i) % self.step_size) > 0, lambda: real_goal, lambda: real_sub_goal)

        i, x_t, _, gen_for_reward, h_worker, h_manager, self.last_goal_for_reward, self.real_goal_for_reward, given_num = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, given_num: i < given_num + 1,
            body=_g_recurrence_1,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.x,
                gen_for_reward,
                self.h0_worker, self.h0_manager, tf.zeros([self.batch_size, self.goal_out_size]), self.goal_init,
                given_num), parallel_iterations=1)  ##input groud-truth

        _, _, gen_for_reward, _, _, _, _ = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6: i < self.sequence_length + 1,
            body=_g_recurrence_2,
            loop_vars=(
                i, x_t, gen_for_reward, h_worker, h_manager, self.last_goal_for_reward, self.real_goal_for_reward),
            parallel_iterations=1)  ## rollout by original policy

        gen_for_reward = gen_for_reward.stack()  # seq_length x batch_size

        gen_for_reward = tf.transpose(gen_for_reward, perm=[1, 0])  # batch_size x seq_length

        return gen_for_reward

    def update_feature_function(self, D_model):
        self.FeatureExtractor_unit = D_model.FeatureExtractor_unit

    def pretrain_step(self, sess, x, dropout_keep_prob):
        outputs = sess.run([self.pretrain_worker_updates, self.pretrain_worker_loss, self.pretrain_manager_updates,
                            self.pretrain_goal_loss],
                           feed_dict={self.x: x, self.drop_out: dropout_keep_prob})
        return outputs

    def generate(self, sess, dropout_keep_prob, train=1):
        outputs = sess.run(self.gen_x, feed_dict={self.drop_out: dropout_keep_prob, self.train: train})
        return outputs

    def create_Worker_recurrent_unit(self, params):
        with tf.variable_scope('Worker'):
            # Weights and Bias for input and hidden tensor
            self.Wi_worker = tf.Variable(tf.random_normal([self.emb_dim, self.hidden_dim], stddev=0.1))
            self.Ui_worker = tf.Variable(tf.random_normal([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bi_worker = tf.Variable(tf.random_normal([self.hidden_dim], stddev=0.1))

            self.Wf_worker = tf.Variable(tf.random_normal([self.emb_dim, self.hidden_dim], stddev=0.1))
            self.Uf_worker = tf.Variable(tf.random_normal([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bf_worker = tf.Variable(tf.random_normal([self.hidden_dim], stddev=0.1))

            self.Wog_worker = tf.Variable(tf.random_normal([self.emb_dim, self.hidden_dim], stddev=0.1))
            self.Uog_worker = tf.Variable(tf.random_normal([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bog_worker = tf.Variable(tf.random_normal([self.hidden_dim], stddev=0.1))

            self.Wc_worker = tf.Variable(tf.random_normal([self.emb_dim, self.hidden_dim], stddev=0.1))
            self.Uc_worker = tf.Variable(tf.random_normal([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bc_worker = tf.Variable(tf.random_normal([self.hidden_dim], stddev=0.1))
            params.extend([
                self.Wi_worker, self.Ui_worker, self.bi_worker,
                self.Wf_worker, self.Uf_worker, self.bf_worker,
                self.Wog_worker, self.Uog_worker, self.bog_worker,
                self.Wc_worker, self.Uc_worker, self.bc_worker])

            def unit(x, hidden_memory_tm1):
                previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

                # Input Gate
                i = tf.sigmoid(
                    tf.matmul(x, self.Wi_worker) +
                    tf.matmul(previous_hidden_state, self.Ui_worker) + self.bi_worker
                )

                # Forget Gate
                f = tf.sigmoid(
                    tf.matmul(x, self.Wf_worker) +
                    tf.matmul(previous_hidden_state, self.Uf_worker) + self.bf_worker
                )

                # Output Gate
                o = tf.sigmoid(
                    tf.matmul(x, self.Wog_worker) +
                    tf.matmul(previous_hidden_state, self.Uog_worker) + self.bog_worker
                )

                # New Memory Cell
                c_ = tf.nn.tanh(
                    tf.matmul(x, self.Wc_worker) +
                    tf.matmul(previous_hidden_state, self.Uc_worker) + self.bc_worker
                )

                # Final Memory cell
                c = f * c_prev + i * c_

                # Current Hidden state
                current_hidden_state = o * tf.nn.tanh(c)

                return tf.stack([current_hidden_state, c])

            return unit

    def create_Worker_output_unit(self, params):
        with tf.variable_scope('Worker'):
            self.W_worker = tf.Variable(
                tf.random_normal([self.hidden_dim, self.vocab_size * self.goal_size], stddev=0.1))
            self.b_worker = tf.Variable(tf.random_normal([self.vocab_size * self.goal_size], stddev=0.1))
            params.extend([self.W_worker, self.b_worker])

            def unit(hidden_memory_tuple):
                hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
                # hidden_state : batch x hidden_dim
                logits = tf.matmul(hidden_state, self.W_worker) + self.b_worker
                # output = tf.nn.softmax(logits)
                return logits

            return unit

    def create_Manager_recurrent_unit(self, params):
        with tf.variable_scope('Manager'):
            # Weights and Bias for input and hidden tensor
            self.Wi = tf.Variable(tf.random_normal([self.num_filters_total, self.hidden_dim], stddev=0.1))
            self.Ui = tf.Variable(tf.random_normal([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bi = tf.Variable(tf.random_normal([self.hidden_dim], stddev=0.1))

            self.Wf = tf.Variable(tf.random_normal([self.num_filters_total, self.hidden_dim], stddev=0.1))
            self.Uf = tf.Variable(tf.random_normal([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bf = tf.Variable(tf.random_normal([self.hidden_dim], stddev=0.1))

            self.Wog = tf.Variable(tf.random_normal([self.num_filters_total, self.hidden_dim], stddev=0.1))
            self.Uog = tf.Variable(tf.random_normal([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bog = tf.Variable(tf.random_normal([self.hidden_dim], stddev=0.1))

            self.Wc = tf.Variable(tf.random_normal([self.num_filters_total, self.hidden_dim], stddev=0.1))
            self.Uc = tf.Variable(tf.random_normal([self.hidden_dim, self.hidden_dim], stddev=0.1))
            self.bc = tf.Variable(tf.random_normal([self.hidden_dim], stddev=0.1))
            params.extend([
                self.Wi, self.Ui, self.bi,
                self.Wf, self.Uf, self.bf,
                self.Wog, self.Uog, self.bog,
                self.Wc, self.Uc, self.bc])

            def unit(x, hidden_memory_tm1):
                previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

                # Input Gate
                i = tf.sigmoid(
                    tf.matmul(x, self.Wi) +
                    tf.matmul(previous_hidden_state, self.Ui) + self.bi
                )

                # Forget Gate
                f = tf.sigmoid(
                    tf.matmul(x, self.Wf) +
                    tf.matmul(previous_hidden_state, self.Uf) + self.bf
                )

                # Output Gate
                o = tf.sigmoid(
                    tf.matmul(x, self.Wog) +
                    tf.matmul(previous_hidden_state, self.Uog) + self.bog
                )

                # New Memory Cell
                c_ = tf.nn.tanh(
                    tf.matmul(x, self.Wc) +
                    tf.matmul(previous_hidden_state, self.Uc) + self.bc
                )

                # Final Memory cell
                c = f * c_prev + i * c_

                # Current Hidden state
                current_hidden_state = o * tf.nn.tanh(c)

                return tf.stack([current_hidden_state, c])

            return unit

    def create_Manager_output_unit(self, params):
        with tf.variable_scope('Manager'):
            self.Wo = tf.Variable(tf.random_normal([self.hidden_dim, self.goal_out_size], stddev=0.1))
            self.bo = tf.Variable(tf.random_normal([self.goal_out_size], stddev=0.1))
            params.extend([self.Wo, self.bo])

            def unit(hidden_memory_tuple):
                hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
                # hidden_state : batch x hidden_dim
                logits = tf.matmul(hidden_state, self.Wo) + self.bo
                # output = tf.nn.softmax(logits)
                return logits

            return unit
