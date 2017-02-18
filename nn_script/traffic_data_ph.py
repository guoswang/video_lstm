from TensorflowToolbox.data_flow.data_ph_abs import DataPhAbs
import tensorflow as tf


class DataPh(DataPhAbs):
    def __init__(self, model_params):
        unroll_num = model_params["unroll_num"]
        self.input_ph_list = list()
        self.label_ph_list = list()
        self.mask_ph_list = list()

        for i in range(unroll_num): 
            input_ph = tf.placeholder(tf.float32, shape=[
                model_params["batch_size"],
                model_params["feature_ph_row"],
                model_params["feature_ph_col"],
                model_params["feature_cha"]],
                name = "feature_%d"%i)

            self.input_ph_list.append(input_ph)

        for i in range(unroll_num): 
            label_ph = tf.placeholder(tf.float32, shape=[
                model_params["batch_size"],
                model_params["label_ph_row"],
                model_params["label_ph_col"],
                model_params["label_cha"]],
                name = "label_%d"%i)
            self.label_ph_list.append(label_ph)

        for i in range(unroll_num): 
            mask_ph = tf.placeholder(tf.float32, shape=[
                model_params["batch_size"],
                model_params["mask_ph_row"],
                model_params["mask_ph_col"],
                model_params["mask_cha"]],
                name = "mask_%d"%i)
            self.mask_ph_list.append(mask_ph)

    def get_label(self):
        return self.label_ph_list

    def get_input(self):
        return self.input_ph_list

    def get_mask(self):
        return self.mask_ph_list
