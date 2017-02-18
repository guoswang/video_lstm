from TensorflowToolbox.model_flow.model_abs import ModelAbs
from TensorflowToolbox.model_flow import model_func as mf
import tensorflow as tf

class Model(ModelAbs):
    def __init__(self, data_ph, model_params):
        self.model_infer(data_ph, model_params)
        self.model_loss(data_ph, model_params)
        self.model_mini(model_params)

    def model_infer(self, data_ph, model_params):
        cell_dim = model_params["lstm_cell_dim"]
        cell_layer = model_params["lstm_cell_layer"]
        batch_size = model_params["batch_size"]

        input_ph_list = data_ph.get_input()
        label_ph_list = data_ph.get_label()

        fc_list = self._model_infer_cnn(data_ph, model_params)

        single_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_dim)
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] 
                        * cell_layer)
        
        cell_initial_state = multi_cell.zero_state(batch_size, 
                    tf.float32)
        
        output, state = tf.nn.rnn(multi_cell, fc_list, \
                initial_state = cell_initial_state)

        self.output = output

        count = self._model_infer_to_count(output, model_params)
        self.count = count
    
    def _model_infer_cnn(self, data_ph, model_params):
        input_ph_list = data_ph.get_input()
        unroll_num = len(input_ph_list)
        fc_list = list()

        for i, input_ph in enumerate(input_ph_list):
            fc = self._model_infer_cnn_single(input_ph, model_params, i)
            fc_list.append(fc)

        return fc_list

    def _model_infer_cnn_single(self, input_ph, model_params, index):
        wd = model_params["weight_decay"]
        fc = mf.fully_connected_layer(input_ph, 100, wd, "fc%d"%index)
        return fc
    
    def _model_infer_to_count(self, lstm_output, model_params):
        wd = model_params["weight_decay"]
        count_list = list()
        for i , output in enumerate(lstm_output):
            with tf.variable_scope("count_fc", reuse = (i != 0)):
                count = mf.fully_connected_layer(output, 1, wd, "fc")
                count_list.append(count)

        return count_list

    def model_loss(self, data_ph, model_params):
        label = data_ph.get_label()
        mask = data_ph.get_mask()

        unroll_num = model_params["unroll_num"]

        count_loss_list = list()

        for i in range(unroll_num):
            count_label = tf.reduce_sum(label[i], [1,2,3])
            count_loss = mf.l2_loss(self.count[i], count_label, 
                        "SUM", "count_loss")
            count_loss_list.append(count_loss)

        self.l2_loss = tf.add_n(count_loss_list)
        self.loss = self.l2_loss

    def model_mini(self, model_params):
        optimizer = tf.train.AdamOptimizer(
            model_params["init_learning_rate"],
            epsilon=1.0)
        self.train_op = optimizer.minimize(self.loss)

    def get_train_op(self):
        return self.train_op

    def get_loss(self):
        return self.loss

    def get_l2_loss(self):
        return self.l2_loss

