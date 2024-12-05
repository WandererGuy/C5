import cv2
import os 
import os.path as osp
from tqdm import tqdm
import logging

# Set up basic logging configuration
# Basic logging configuration with a cleaner format
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",  # Customized format
    datefmt="%Y-%m-%d %H:%M:%S"  # Timestamp format
)
os.makedirs("materials/preprocess", exist_ok=True)


PADDING_COLOR = (0,0,0) 


def make_indexed_folder(base_folder_name):
    index = 1
    save_folder = base_folder_name

    # Loop until a non-existing folder name is found
    while os.path.exists(save_folder):
        save_folder = f"{base_folder_name}_{index}"
        index += 1

    # Create the directory
    os.makedirs(save_folder)
    return save_folder




def frame_crop(video_path, sensor_name, save_folder):

    cap = cv2.VideoCapture(video_path)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    # fps = cap.get(cv2.CAP_PROP_FPS)
    save_video_name = video_path.split("/")[-1]

    save_video_name = save_video_name.split(".")[0]
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id % 4 == 0:
                    save_name = "{:03d}".format(frame_id) + f"_{sensor_name}.png"
                    path_save = osp.join(save_folder, save_name)
                    cv2.imwrite(path_save, frame)
        frame_id = frame_id + 1

def pseudo_metadata_json(pseudo_groundtruth_string, folder_path):
        for filename in os.listdir(folder_path):
            path = osp.join(folder_path, filename)
            pseudo_path = osp.join(folder_path, filename.replace(".png", "_metadata.json"))
            with open(pseudo_path, 'w') as f_pseudo_metadata:
                f_pseudo_metadata.write(pseudo_groundtruth_string)
            


def sort_filename(folder_path):
    # List all files in the directory
    files = os.listdir(folder_path)

    # Sort the files based on the numeric part at the beginning of each filename
    sorted_files = sorted(files, key=lambda x: int((x.split('.')[0]).split('-')[0]))
    return sorted_files


def remove_padding(image):
    # image = cv2.imread(source_path)

    # Convert to grayscale (black padding is easy to detect in grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary mask (black padding is 0, other regions are 255)
    '''
    Any pixel value greater than 1 will be converted to 255 (white).
    Any pixel value less than or equal to 1 will be set to 0 (black).

    '''
    _, thresh = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)

    # Find contours of the white regions in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Finds the four vertices of a straight rect. Useful to draw the rotated rectangle.
    x, y, w, h = cv2.boundingRect(max_contour)  # Get bounding box of the contour (minimum rect)


    '''
    x1 ----> x2
    |      |
    |      |
    |      |
    |      v
    x4 <---- x3
    '''
    cropped_image = image[int(y):int(y+h), int(x):int(x+w)]
    
    return cropped_image


def padding(im, new_unpad_shape, target_shape):
    # Define the color to use for padding (a shade of gray)
    dw = target_shape[0] - new_unpad_shape[0]  # Width padding
    dh = target_shape[1] - new_unpad_shape[1]  # Height padding

    # Divide the padding equally for left/right and top/bottom
    dw /= 2  
    dh /= 2

    # Compute padding for each side, adjusting for rounding errors
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    logging.info("padding pixels for top, bottom, left, right: {}".format((top, bottom, left, right)))
    # Add border (padding) to the image using the specified color
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=PADDING_COLOR
    )
    logging.info("final im.shape: {}".format(im.shape))
    return im  # Return the padded image

def resize_before_padding(im, target_width, target_height):
    logging.info("start resize while keeping aspect ratio.")
    logging.info('height, width')
    # Get original image dimensions
    original_height, original_width = im.shape[:2]


    # Calculate scaling factor to maintain aspect ratio
    scale = min(
        target_width / original_width,
        target_height / original_height
    )

    # Compute new dimensions while maintaining aspect ratio
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    # Resize the image to the new dimensions
    im_resized = cv2.resize(
        im, (new_width, new_height),
        interpolation=cv2.INTER_CUBIC
    )
    
    '''
    INTER_AREA FOR SMOOTH IMAGE 
    CUBIC IS MOST BALANCE 
    THERE IS ALWAYS PIXELATED IMAGES
    '''
    
    # h = int(round(original_height*scale))
    # im_resized = imutils.resize(im, height=h)
    # new_height, new_width = im_resized.shape[:2]

    cv2.imwrite('im_resized.png', im_resized)
    logging.info("scale: {}".format(scale))
    logging.info("im.shape: {}".format(im.shape))
    logging.info('im_resized.shape: {}'.format(im_resized.shape))
    logging.info("target size: {}".format((target_height, target_width)))
    # Return the resized image and both the new and target dimensions
    return im_resized, (new_width, new_height), (target_width, target_height)



def remove_padding(source_path):
    image = cv2.imread(source_path)

    # Convert to grayscale (black padding is easy to detect in grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary mask (black padding is 0, other regions are 255)
    '''
    Any pixel value greater than 1 will be converted to 255 (white).
    Any pixel value less than or equal to 1 will be set to 0 (black).

    '''
    _, thresh = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)

    # Find contours of the white regions in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Finds the four vertices of a straight rect. Useful to draw the rotated rectangle.
    x, y, w, h = cv2.boundingRect(max_contour)  # Get bounding box of the contour (minimum rect)


    '''
    x1 ----> x2
    |      |
    |      |
    |      |
    |      v
    x4 <---- x3
    '''
    cropped_image = image[int(y):int(y+h), int(x):int(x+w)]
    
    return cropped_image

def undo_padding_and_resize(img_path, target_width, target_height, new_save_folder):
    # undo padding
    unpad_img = remove_padding(img_path)
    im_ori_shape, _ , _ = resize_before_padding(unpad_img, 
                                             target_width, 
                                             target_height)   
    final_img_path = os.path.join(new_save_folder, os.path.basename(img_path))
    cv2.imwrite(final_img_path, im_ori_shape)
    return final_img_path

    # resize back to original size

def resize_and_padding(cropped_image,sensor_name, save_folder, source_path, resize_enable):
    '''
    resize but keeping aspect ratio of image 
    '''
    if resize_enable:
        im, new_unpad_shape, target_shape = resize_before_padding(cropped_image, 
                                                      target_width = 384, 
                                                      target_height = 256)
        pad_img = padding(im, new_unpad_shape, target_shape)

    else: 
        pad_img = cropped_image
    filename = os.path.basename(source_path)
    name = filename.replace('.' + filename.split('.')[1], '')
    save_name = name + f"_{sensor_name}.png"
    save_path = osp.join(save_folder, save_name)

    cv2.imwrite(save_path, pad_img)
    logging.info("file_path: {}".format(source_path))
    logging.info("save_path: {}".format(save_path))


def take_and_process_frames(source_folder_path, sensor_name, save_folder, num_frequency = 1, resize_enable = True):
    original_img_shape_dict = {}
    ls = sort_filename(source_folder_path)
    for i,filename in tqdm(enumerate(ls), total=len(ls)):
        if i % num_frequency == 0:
            source_path = os.path.join(source_folder_path, filename)
            # cropped_image = remove_padding(source_path)
            cropped_image = cv2.imread(source_path)
            resize_and_padding(cropped_image,sensor_name, save_folder, source_path, resize_enable)

            original_height, original_width = cropped_image.shape[:2]
            name = os.path.basename(source_path).split('.')[:-1]
            full_name = ''
            for i in name:
                full_name = full_name + i
            original_img_shape_dict[full_name] = (original_width, original_height)
    return original_img_shape_dict
        
#### create every 4 frame in video into folder with name of image have sensor info 
# # frame_crop(video_path, sensor_name, save_folder)

def prepare_input(source_folder_path, sensor_name, save_folder, num_frequency = 1, resize_enable = True):
    #### preprocess images +  create pseudo metadata json for a folder of images for every n_frequency  for each image
    new_save_folder = make_indexed_folder(base_folder_name  = save_folder)
    original_img_shape_dict = take_and_process_frames(source_folder_path, sensor_name, new_save_folder, num_frequency = num_frequency, resize_enable = resize_enable)
    pseudo_groundtruth_string = '{"illuminant_color_raw":[0.70450692619758248,0.65185828786184485,0.28062566433856068]}'
    pseudo_metadata_json(pseudo_groundtruth_string, new_save_folder)
    logging.info('FOLDER FOR INFERENCE: {}'.format(new_save_folder))
    return original_img_shape_dict

if __name__ == "__main__":
    source_folder_path = "C:/Users/Admin/Documents/001"
    save_folder = "materials/001_preprocessed"
    sensor_name = 'sensorname_gayy'
    prepare_input(source_folder_path, sensor_name, save_folder)