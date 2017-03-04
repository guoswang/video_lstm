from TensorflowToolbox.utility import read_proto as rp
from TensorflowToolbox.utility import file_io
from net_flow import NetFlow
import sys

if __name__ == "__main__":
    if len(sys.argv) == 2:
        model_proto_file = sys.argv[1]
    else:
        model_proto_file = "model.tfproto"

    model_params = rp.load_proto("model.tfproto")
    file_io.check_exist(model_params["train_file_name"]) 
    file_io.check_exist(model_params["test_file_name"]) 

    net = NetFlow(model_params, True, True)
    net.mainloop()
