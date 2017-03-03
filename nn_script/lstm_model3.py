from TensorflowToolbox.model_flow.model_abs import ModelAbs
from TensorflowToolbox.model_flow import model_func as mf
from TensorflowToolbox.utility import image_utility_func as iuf
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

        predict_list, fc_list = self._model_infer_cnn(data_ph, model_params)

        self.predict_list = predict_list

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
        predict_list = list()

        desmap_scale = model_params["desmap_scale"]

        with tf.variable_scope("CNN"):
            for i, input_ph in enumerate(input_ph_list):
                if i != 0:
                    tf.get_variable_scope().reuse_variables()
                deconv_list, fc = self._model_infer_cnn_single(
                                input_ph, model_params)
                
                #fc = tf.reduce_sum(fc, 1, True) / desmap_scale
                fc_list.append(fc)
                predict_list.append(deconv_list)

        return predict_list, fc_list

    def _deconv2_wrapper(self, input_tensor, sample_tensor, 
                output_channel, wd, layer_name):
        [b, h, w, _] = sample_tensor.get_shape().as_list()
        [_,_,_,c] = input_tensor.get_shape().as_list()

        deconv = mf.deconvolution_2d_layer(input_tensor, 
                    [3, 3, output_channel, c], 
                    [2, 2], [b, h, w, output_channel], 'VALID', 
                    wd, layer_name)
        return deconv

    def _model_infer_cnn_single(self, input_ph, model_params):
        leaky_param = model_params["leaky_param"]
        wd = model_params["weight_decay"]

        hyper_list = list()

        print(input_ph)

        conv11 = mf.add_leaky_relu(mf.convolution_2d_layer(
            input_ph, [3, 3, 3, 64], [1, 1],
            "SAME", wd, "conv1_1"), leaky_param)

        conv12 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv11, [3, 3, 64, 64], [1, 1],
            "SAME", wd, "conv1_2"), leaky_param)

        conv12_maxpool = mf.maxpool_2d_layer(conv12, [3, 3],
                                             [2, 2], "maxpool1")

        print(conv12_maxpool)

        conv21 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv12_maxpool, [3, 3, 64, 128], [1, 1],
            "SAME", wd, "conv2_1"), leaky_param)

        conv22 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv21, [3, 3, 128, 128], [1, 1],
            "SAME", wd, "conv2_2"), leaky_param)

        conv22_maxpool = mf.maxpool_2d_layer(conv22, [3, 3],
                                             [2, 2], "maxpool2")

        print(conv22_maxpool)
        hyper_list.append(conv22_maxpool)

        conv31 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv22_maxpool, [3, 3, 128, 256], [1, 1],
            "SAME", wd, "conv3_1"), leaky_param)

        conv32 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv31, [3, 3, 256, 256], [1, 1],
            "SAME", wd, "conv3_2"), leaky_param)

        atrous3 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            conv32, [3, 3, 256, 256], 2,
            "SAME", wd, "atrous3"), leaky_param)

        print(atrous3)
        hyper_list.append(atrous3)

        conv41 = mf.add_leaky_relu(mf.convolution_2d_layer(
            atrous3, [3, 3, 256, 512], [1, 1],
            "SAME", wd, "conv4_1"), leaky_param)

        conv42 = mf.add_leaky_relu(mf.convolution_2d_layer(
            conv41, [3, 3, 512, 512], [1, 1],
            "SAME", wd, "conv4_2"), leaky_param)

        atrous4 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            conv42, [3, 3, 512, 512], 2,
            "SAME", wd, "atrous4"), leaky_param)

        print(atrous4)
        hyper_list.append(atrous4)

        atrous51 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            atrous4, [3, 3, 512, 512], 2,
            "SAME", wd, "atrous5_1"), leaky_param)

        atrous52 = mf.add_leaky_relu(mf.atrous_convolution_layer(
            atrous51, [3, 3, 512, 512], 2,
            "SAME", wd, "atrous5_2"), leaky_param)

        print(atrous52)

        hyper_list.append(atrous52)

        hypercolumn = self._pack_tensor_list(hyper_list)
        print(hypercolumn)

        [b, w, h, c] = hypercolumn.get_shape().as_list()
        conv6 = mf.add_leaky_relu(mf.convolution_2d_layer(
            hypercolumn, [1, 1, c, 512], [1, 1],
            "SAME", wd, "conv6"), leaky_param)
        
        deconv1 = self._deconv2_wrapper(conv6, conv21, 
                    256, wd, "deconv1")
        print(deconv1)

        deconv2 = self._deconv2_wrapper(deconv1, conv11, 
                    64, wd, "deconv2")
        print(deconv2)

        conv7 = mf.add_leaky_relu(mf.convolution_2d_layer(
            deconv2, [1, 1, 64, 1], [1, 1],
            "SAME", wd, "conv7"), leaky_param)
        print(conv7)

        predict_list = list()
        predict_list.append(conv7)

        fc = tf.reshape(conv7, [b, -1], "vectorize") 

        return predict_list, fc

    def _pack_tensor_list(self, tensor_list):
        hypercolumn = tf.concat(3, tensor_list)

        return hypercolumn

    def _filter_mask(self, tensor, mask):
        tensor = tensor * mask
        return tensor
    
    def _model_infer_to_count(self, lstm_output, model_params):
        wd = model_params["weight_decay"]
        desmap_scale = model_params["desmap_scale"]

        count_list = list()
        with tf.variable_scope("count_fc"):
            for i , output in enumerate(lstm_output):
                if i != 0:
                    tf.get_variable_scope().reuse_variables()

                count_bias = mf.fully_connected_layer(output, 1, wd, "fc")
                image_sum = tf.expand_dims(tf.reduce_sum(
                                self.predict_list[i][-1], [1,2,3]),1)/desmap_scale

                count = count_bias + image_sum / desmap_scale

                tf.summary.scalar("check/image_sum/%d"%(i), 
                                tf.reduce_sum(image_sum))
                tf.summary.scalar("check/count_bias/%d"%(i), 
                                tf.reduce_sum(count_bias))

                count_list.append(count)
        return count_list

        #count = mf.fully_connected_layer(lstm_output[-1], 1, wd, "fc")

        #return count
        
    def _image_l2_loss(self, label, mask, predict_list, index, model_params):
        """
        Args:
            label: [b, h, w, c]
            mask: [b, h, w, c]
            predict_list: list of [b, h, w, c]
        """
        desmap_scale = model_params["desmap_scale"]
        l2_loss_list = list()
        for i, deconv in enumerate(predict_list):
            deconv = self._filter_mask(deconv, mask)
            label = self._filter_mask(label, mask)
            l2_loss = mf.image_l2_loss(deconv, label, 
                        "image_loss_%d_%d"%(index, i))
            l2_loss_list.append(l2_loss)
            count_diff = mf.count_diff(deconv, 
                        label, "count_diff_%d_%d"%(index, i)) / desmap_scale
            tf.summary.scalar("image_count_diff/%d_%d"%(index,i), count_diff)
            #tf.add_to_collection("losses", l2_loss)

        l2_loss = tf.add_n(l2_loss_list)
        return l2_loss

    def _add_image_sum(self, input_img, label, mask):
        with tf.variable_scope("image_sum"):
            concat_1 = iuf.merge_image(2, input_img)
            concat_2 = iuf.merge_image(2, label)
            deconv_img_list = [img[-1] for img in self.predict_list]
            concat_3 = iuf.merge_image(2, deconv_img_list)
            concat_4 = iuf.merge_image(2, mask)
            image_sum = iuf.merge_image(1, [concat_1, concat_2, 
                        concat_3, concat_4])

            tf.add_to_collection("image_to_write", image_sum)
        

    def model_loss(self, data_ph, model_params):
        input_img = data_ph.get_input()
        label = data_ph.get_label()
        mask = data_ph.get_mask()

        self._add_image_sum(input_img, label, mask)

        unroll_num = model_params["unroll_num"]
        desmap_scale = model_params["desmap_scale"]

        with tf.variable_scope("loss"):

            count_loss_list = list()
            image_loss_list = list()
            count_l1_loss_list = list()

            for i in range(unroll_num):
                count_label = tf.reduce_sum(label[i], [1,2,3])/desmap_scale
                count_infer = tf.reduce_sum(self.count[i], 1)
                
                count_loss = mf.l2_loss(count_infer, count_label, 
                            "MEAN", "count_loss_%d"%i)
                count_loss_list.append(count_loss)

                tf.summary.scalar("count_diff/%d"%i,
                                    mf.l1_loss(count_infer,
                                    count_label, "MEAN", "l1_loss"))

                tf.summary.scalar("count_label/%d"%i,
                                    tf.reduce_mean(count_label))

                tf.summary.scalar("count_infer/%d"%i,
                                    tf.reduce_mean(count_infer))
            
                count_l1_loss = mf.l1_loss(count_infer, count_label,
                            "MEAN", "count_loss_%d"%i)

                count_l1_loss_list.append(count_l1_loss)

                image_loss = self._image_l2_loss(label[i], mask[i], 
                            self.predict_list[i], i, model_params)

                image_loss_list.append(image_loss)

            self.l1_loss = tf.reduce_mean(count_l1_loss_list)
            #tf.add_to_collection("losses", self.l1_loss)

            self.l2_loss = tf.add_n(count_loss_list)
            tf.add_to_collection("losses", self.l2_loss)

            self.image_loss = tf.add_n(image_loss_list)
            tf.add_to_collection("losses", self.image_loss)

            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    def model_mini(self, model_params):
        with tf.variable_scope("optimization"):
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

    def get_l1_loss(self):
        return self.l1_loss

    def get_image_loss(self):
        return self.image_loss

