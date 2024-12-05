
print("............. Initialization .............")
from fastapi import FastAPI, HTTPException, Form, Request
import uvicorn
import configparser
from pathlib import Path
from preprocess_input import prepare_input, undo_padding_and_resize
import os 
import uuid
from infer_utils import load_model, infer, get_args
from time import time
from tqdm import tqdm
import torch
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cpu':
    torch.cuda.set_device(args.gpu)
# input 
args.white_balance = True
# args.g_multiplier = True
# args.model_name = 'C5_m_7_h_64_w_G'
args.model_name = "C5_m_7_h_64"
args.data_num = 7 # number of input histograms to C5 network (m in the paper);
                  # default value is 7
args.batchsize = 8 # if len(folder) < batchsize , it loads len(folder)
                    # smaller batchsize for smaller vram takes 
                    # 8 because , len folder min is 7 so i think 7 is like minimum here 
                    # author make batchsize 64 so why not 8 instead of 7 for good performance
                    # latency vs batch size = 2 is not much , about 57 to 59 , so 8 is faster 2 sec 
                    # vram not take much , even 64 still same time as 8 but more vram just 1 gb , since histogram is light input 

net = load_model(args, device)

from all_utils.utils import fix_path, create_new_source_folder
config = configparser.ConfigParser()
current_script_directory = os.path.dirname(os.path.abspath(__file__))
config.read(os.path.join(current_script_directory,'config.ini'))
host_ip = config['DEFAULT']['host'] 
port_num = config['DEFAULT']['port_num'] 
preprocessed_folder = os.path.join("materials","preprocess")
os.makedirs(preprocessed_folder, exist_ok=True)


app = FastAPI()
@app.post("/white-balance-sequence") 
async def white_balance_sequence(source_folder_path: str = Form(...)):
    print("------------REQUEST RECEIVED------------")
    # prepare input
    start = time()
    frame_dict = {}
    source_folder_path = fix_path(source_folder_path)
    image_extension = '.' + os.listdir(source_folder_path)[0].split('.')[-1]
    sensor_name = 'sensorname_pseudo'
    uuid_str = str(uuid.uuid4())
    # take random 7 pic for less infer time 
    # create_new_source_folder(source_folder_path, 
    #                          num_save=7, 
    #                          save_file_path=save_file_path)
    save_folder = os.path.join(current_script_directory, preprocessed_folder,uuid_str)
    print ('PREPARE INPUT FOR MODEL INFERENCE : RESIZE AND PADDING TO 384X256')

    original_img_shape_dict = prepare_input(source_folder_path, sensor_name, save_folder)

    save_folder = fix_path(save_folder)
    args.in_tedir = save_folder

    # 1 need image and 1 addional image , though best is 7 by default 
    print ('START INFERENCE')
    result_folder = infer(args, net, device)

    new_result_folder = result_folder + '_processed'
    os.makedirs(new_result_folder, exist_ok=True)
    print ('PROCESS INFERENCE RESULT : UNPAD AND RESIZE TO ORIGINAL SIZE')

    for new_filename in tqdm(os.listdir(result_folder), total = len(os.listdir(result_folder))):
        new_img_path = os.path.join(result_folder, new_filename)
        ori_filename = os.path.basename(new_img_path).split(f'_{sensor_name}')[0]
        original_width, original_height = original_img_shape_dict[ori_filename]
        final_img_path = undo_padding_and_resize(img_path = new_img_path, 
                                target_width = original_width, 
                                target_height = original_height, 
                                new_save_folder = new_result_folder
                                )
        ori_frame_path = os.path.join(source_folder_path, ori_filename + image_extension)
        ori_frame_path = fix_path(ori_frame_path)
        frame_dict[ori_frame_path] = final_img_path  
    res = {
            "status": 1,
            "error_code": None,
            "error_message": None, 
            "result": 
                {
                    "result_folder" : fix_path(result_folder),
                    "result_folder_processed" : fix_path(new_result_folder),
                    "frame_dict": frame_dict
                }
        }    
    print ("total infer time", time() - start)
    print (res)
    return res
    # return 'Done'
    '''
    take most time í compute histogram file npy for each image 
    average 2.18 
    '''


def main():
    print('INITIALIZING FASTAPI SERVER')
    uvicorn.run(app, host=host_ip, port=int(port_num), reload=False)
    # uvicorn.run("sample_seg:app", host=host_ip, port=int(seg_port_num), reload=True)


if __name__ == "__main__":
    main()

    