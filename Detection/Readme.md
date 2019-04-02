# Training the model
To train the model you need to run this command:
python3.6 tower_data_train.py --dataset_dir dataset/ --coco_model_path ../mask_rcnn_coco.h5 --trained_model_path model/     

coco_model_path specify the path where the weights of the model trained previously on COCO dataset is located, that will be used
to intitialize the model before training on our dataset

# Prediciton
To run the  model (saved in model_v1) in inferece mode on some image 

python3.6 predict.py --trained_model_path model_v1/ --image ~/TOWERDATA/Batch-4_Corona/Best/449_P5030503.JPG   --min_confidence 0.85 --save_dir output/

Note that in this file in main, it simply define the trained model args and then run detect_big_image() on the big image which divide it into smaller images and run the model on each one. You can use this part inside the main as a guidline of how to incorporate it in your app. Note that the results which are the image  with bboxes, classes and confidence and the csv file will be saved in the --save_dir

# model evaluation
To evaluate the model you can run this commands

python3.6 evaluate.py --trained_model_path model_v1/ --in_dir dataset/imgs/ --min_confidence 0.5
cd mAP/
python3.6 main.py -na -np --set-class-iou Insulator 0.1 Corona_Ring 0.01

but for lower false positive you can use higher min_confidence (recommended: 0.80 +)
Expected results:

71.85% = Corona_Ring AP 
91.05% = Davit_Arm AP 
90.48% = Insulator AP
mAP = 84.46%


# To run on directory
The inDir should contain only images. each image will be divided and processed and its two output files (csv and detections visualized) will be save in the outDir with the same name of the image

python3.6 batch_detect.py --trained_model_path model_v1/ --image   --min_confidence 0.85 --outDir output/ --inDir ../batch_test

If you run this code on batch_test directory that contain 16 images you should get logs simialr to that

INFO | Loading the model 
INFO | time = 11 secs 
INFO | Running the model on ../batch_test that contain 16 images 
INFO | Total processing time = 82 secs 
INFO | Processing time per image = 5.1 secs

As you can see loading the model take 11 secand all images processed in 82 secs.


