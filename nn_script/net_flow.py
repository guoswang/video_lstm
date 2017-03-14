from traffic_data_ph import DataPh
from traffic_data_input import DataInput
from TensorflowToolbox.utility import file_io
import tensorflow as tf
from TensorflowToolbox.model_flow import save_func as sf
from TensorflowToolbox.utility import utility_func as uf 
from TensorflowToolbox.utility import result_obj as ro
import cv2
import numpy as np
import os

from save_density_map import save_density_map

TF_VERSION = tf.__version__.split(".")[1]


class NetFlow(object):
    def __init__(self, model_params, load_train, load_test):
        
        self.load_train = load_train
        self.load_test = load_test
        self.model_params = model_params
        self.check_model_params(model_params)

        if load_train:
            self.train_data_input = DataInput(model_params, is_train=True)
        if load_test:
            self.test_data_input = DataInput(model_params, is_train=False)

        self.data_ph = DataPh(model_params)
        model = file_io.import_module_class(model_params["model_def_name"],
                                            "Model")
        self.model = model(self.data_ph, model_params)
        self.loss = self.model.get_loss()
        self.l2_loss = self.model.get_l2_loss()
        self.l1_loss = self.model.get_l1_loss()
        self.count = self.model.get_count()
        self.label_count = self.model.get_label_count()
        self.image_loss = self.model.get_image_loss()
        self.train_op = self.model.get_train_op()
        self.infer_density_map = self.model.get_infer_density_map()

    @staticmethod
    def check_model_params(model_params):
        field_list = ["restore_model", "model_dir", "max_training_iter",
                      "train_log_dir", "test_per_iter", "save_per_iter"]

        for field in field_list:
            assert(field in model_params)

    def get_feed_dict(self, sess, is_train):
        
        if is_train:
            input_v, label_v, mask_v, file_line_v = sess.run([
                self.train_data_input.get_input(),
                self.train_data_input.get_label(),
                self.train_data_input.get_mask(),
                self.train_data_input.get_file_line()])
        else:
            input_v, label_v, mask_v, file_line_v = sess.run([
                self.test_data_input.get_input(),
                self.test_data_input.get_label(),
                self.test_data_input.get_mask(),
                self.test_data_input.get_file_line()])

        unroll_num = self.model_params["unroll_num"]

        input_ph_list = self.data_ph.get_input()
        label_ph_list = self.data_ph.get_label()
        mask_ph_list = self.data_ph.get_mask()

        feed_dict = dict()

        desmap_scale = self.model_params["desmap_scale"]

        for i in range(unroll_num):
            feed_dict[input_ph_list[i]] = input_v[i]
            feed_dict[label_ph_list[i]] = label_v[i] * desmap_scale
            feed_dict[mask_ph_list[i]] = mask_v[i]
            #feed_dict[self.data_ph.get_input()] = input_v
            #feed_dict[self.data_ph.get_label()] = label_v * 100
            #feed_dict[self.data_ph.get_mask()] = mask_v

        self.file_line = file_line_v

        return feed_dict

    @staticmethod
    def check_feed_dict(feed_dict):
        data_list = list()

        for key in feed_dict:
            print(key.op.name, np.sum(feed_dict[key]))
        #print(np.sum(data_list[1][0] > 0.1))
        #print(data_list[1][0].max())
        #cv2.imshow("image", data_list[0][0])
        #cv2.imshow("label", data_list[1][0] * 255)
        #cv2.imshow("mask", data_list[2][0])
        #cv2.waitKey(0)

    def init_var(self, sess):
        sf.add_train_var()
        sf.add_loss()
        sf.add_image("image_to_write", 4)
        self.saver = tf.train.Saver()

        if TF_VERSION > '11':
            if self.load_train:
                self.sum_writer = tf.summary.FileWriter(self.model_params["train_log_dir"], 
                                         sess.graph)
            self.summ = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
        else:
            if self.load_train:
                self.sum_writer = tf.train.SummaryWriter(self.model_params["train_log_dir"],
                                         sess.graph)
            self.summ = tf.merge_all_summaries()
            init_op = tf.initialize_all_variables()

        sess.run(init_op)

        if self.model_params["restore_model"]:
            sf.restore_model(sess, self.saver, self.model_params["model_dir"],
                            self.model_params["restore_model_name"])

    def mainloop(self):
        config_proto = uf.define_graph_config(self.model_params["gpu_fraction"])
        sess = tf.Session(config = config_proto)
        self.init_var(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        batch_size = self.model_params["batch_size"]
        unroll_num = self.model_params["unroll_num"]
        desmap_scale = self.model_params["desmap_scale"]

        if self.load_train:
            for i in range(self.model_params["max_training_iter"]):
                feed_dict = self.get_feed_dict(sess, is_train=True)
                #self.check_feed_dict(feed_dict)

                _, tl2_loss_v, timage_loss_v, tl1_loss_v = sess.run([self.train_op, 
                                    self.l2_loss, self.image_loss, self.l1_loss],
                                    feed_dict)
                if i % self.model_params["test_per_iter"] == 0:

                    feed_dict = self.get_feed_dict(sess, is_train=False)
                    l2_loss_v, image_loss_v, l1_loss_v, summ_v = sess.run([
                                self.l2_loss, self.image_loss, 
                                self.l1_loss, self.summ], feed_dict)

                    tcount_diff = tl1_loss_v 
                    count_diff = l1_loss_v 

                    print_string = \
                        "i: %d, train_count_loss: %.2f, train_image_loss: %.2f, "\
                        "test_count_loss: %.2f, test_image_loss: %.2f" %\
                          (i, tcount_diff, timage_loss_v, count_diff, image_loss_v)

                    print(print_string)
                   
                    file_io.save_string(print_string, 
                            self.model_params["train_log_dir"] + 
                            self.model_params["string_log_name"])

                    self.sum_writer.add_summary(summ_v, i)
                    sf.add_value_sum(self.sum_writer, timage_loss_v, 
                                    "train_image_loss", i)
                    sf.add_value_sum(self.sum_writer, tcount_diff, 
                                                    "train_count_diff", i)
                    sf.add_value_sum(self.sum_writer, image_loss_v, 
                                    "test_image_loss", i)
                    sf.add_value_sum(self.sum_writer, count_diff, 
                                                    "test_count_diff", i)

                if i != 0 and (i % self.model_params["save_per_iter"] == 0 or \
                                i == self.model_params["max_training_iter"] - 1):
                    sf.save_model(sess, self.saver,
                                  self.model_params["model_dir"],i)
                    
        else:
            file_len = file_io.get_file_length(self.model_params["test_file_name"])
            batch_size = self.model_params["batch_size"]
            test_iter = int(file_len / batch_size) + 1
            result_file_name = self.model_params["result_file_name"]
            result_obj = ro.ResultObj(result_file_name)
            for i in range(test_iter):
                feed_dict = self.get_feed_dict(sess, is_train=False)
                loss_v, count_v, label_count_v, infer_density_map_v = \
                        sess.run([self.loss, self.count, \
                        self.label_count, self.infer_density_map], feed_dict)

                file_line = [f.decode("utf-8").split(" ")[:unroll_num] \
                            for f in self.file_line]

                file_line = result_obj.vectorize_list(file_line)
                save_density_map(file_line, infer_density_map_v, 
                        unroll_num, desmap_scale)

                label_count_v = result_obj.vectorize_nparray(label_count_v)
                count_v = result_obj.vectorize_nparray(count_v)

                label_count_v = result_obj.float_to_str(label_count_v, "%.2f")
                count_v = result_obj.float_to_str(count_v, "%.2f")

                result_obj.add_to_list(file_line, label_count_v, count_v)


            result_obj.save_to_file(True)

        coord.request_stop()
        coord.join(threads)
