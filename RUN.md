conda activate /home/ai-ubuntu/hddnew/Manh/C5/env_2
python capture_img.py
-> preprocess, padding,  create folder input from regular folder (images from a camera at a time in the day need white balance)




conda activate /home/ai-ubuntu/hddnew/Manh/C5/env_2
python test.py --white-balance True --g-multiplier True --model-name C5_m_7_h_64_w_G --testing-dir-in /home/ai-ubuntu/hddnew/Manh/C5/materials/indoor_preprocessed_8
-> white balance result 