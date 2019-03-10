# Problem type
problem_type                = 'classification'  # Option: ['segmentation','classification','detection']
# Model
model_type                  = "VGG16"  # Options: ['DenseNetFCN', 'FCN8', 'FCN8atOnce' 'VGG16']
"""
    ### DenseNetFCN options ####
model_blocks                = 5               # Number of block densenetFCN_Custom only
model_layers                = 4               # Number of layers per block densenetFCN_Custom only
model_growth                = 12              # Growth rate per block (k) densenetFCN_Custom only
model_upsampling            = 'deconv'        # upsampling types available: 'upsampling' , 'subpixel', 'deconv'
model_dropout               = 0.0             # Dropout rate densenetFCN_Custom only
model_compression           = 0.0             # Compression rate for DenseNet densenetFCN_Custom only
    ### RPN
anchor_scales               = [8,16,32]
anchor_ratios               = [0.5,1,2]
feat_stride                 = 16 #[16, ]
clobber_positives           = False # If an anchor statisfied by positive and negative conditions set to negative
negative_overlap            = 0.3 # IOU < thresh: negative example
positive_overlap            = 0.7 # IOU >= thresh: positive example
fg_fraction                 = 0.5 # Max number of foreground examples
anchor_samples              = 256 # Total number of anchor samples per image to calculate the loss
positive_weight             = -1.0 # Set to -1.0 to use uniform example weighting
rpn_pre_nms_top_n           = 6000 # Number of top scoring boxes to keep before apply NMS to RPN proposals
rpn_post_nms_top_n          = 300 # Number of top scoring boxes to keep after applying NMS to RPN proposals
rpn_nms_thresh              = 0.7 # NMS threshold used on RPN proposals
"""

    ### load options
resume_experiment           = False           # Restore the best model obtained in the experiment defined if exist
pretrained_model            = 'basic'         # 'None': from scratch, 'basic': pretraned from imagenet, 'custom': personal model
input_model_path            = None            # Path and pretrained file to load [None uses experiment path and model name by default]
load_weight_only            = True            # Recomended true, loads only weights and parameters
basic_models_path           = './pretrained_models/' # Path for the basic models (ImageNet weights) where they will be download
    ### Save options
save_weight_only            = True            # Recomended true, stores only weights and parameters
model_name                  = 'VGG16'          # Name of the model to store
output_model_path           = '../Resultados'           # Path to store the model using model_name [None uses the default experiment path]

# Loss type
loss_type                   = 'cross_entropy_segmentation' # options: ['cross_entropy_segmentation','focal_segmentation']
normalize_loss              = True

# General parameters

train_samples               = 50 #-1 uses all the data available inside the dataset files
valid_samples               = -1 #-1 uses all the data available inside the dataset files
test_samples                = 10 #-1 uses all the data available inside the dataset files
train_batch_size            = 8
valid_batch_size            = 1
test_batch_size             = 1
train                       = True
validation                  = True
test                        = True # Calculate metrics on test giving the gt
predict_test                = True  # True when you want to generate predictions from test, doesn't need gt
predict_path_output         = None # None uses the default output in the experiment folder /predictions

# Image properties
size_image_train            = (1024, 2048)#(1280, 960) 
size_image_valid            = (1024, 2048)#(1280, 960)
size_image_test             = (1024, 2048)#(1280, 960)
resize_image_train          = None #(320, 640)#(640, 480)
resize_image_valid          = None #(320, 640)#(640, 480)
resize_image_test           = None #(320, 640)#(640, 480)
crop_train                  = (320, 320)
grayscale                   = False #Use this option to convert to rgb a grascale dataset

# Dataset properties

train_images_txt = '/home/grupo09/M5/dataset/classificator/TT100K_trafficSigns/TT100K_trafficSigns_train_images.txt'
train_gt_txt = '/home/grupo09/M5/dataset/classificator/TT100K_trafficSigns/TT100K_trafficSigns_train_gt.txt'
valid_images_txt = '/home/grupo09/M5/dataset/classificator/TT100K_trafficSigns/TT100K_trafficSigns_valid_images.txt'
valid_gt_txt = '/home/grupo09/M5/dataset/classificator/TT100K_trafficSigns/TT100K_trafficSigns_valid_gt.txt'
test_images_txt = '/home/grupo09/M5/dataset/classificator/TT100K_trafficSigns/TT100K_trafficSigns_test_images.txt'
test_gt_txt = '/home/grupo09/M5/dataset/classificator/TT100K_trafficSigns/TT100K_trafficSigns_test_gt.txt'

labels                       = ['i1', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'i2', 'i3', 'i4', 'i5', 'il100', 'il110', 'il50', 'il60', 'il70', 'il80', 'il90', 'io', 'ip', 'p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p2', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pb', 'pc', 'pg', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.5', 'ph3.8', 'ph4', 'ph4.2', 'ph4.3', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'pl10', 'pl100', 'pl110', 'pl120', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl5', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'pm10', 'pm13', 'pm15', 'pm1.5', 'pm2', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm5', 'pm50', 'pm55', 'pm8', 'pn', 'pne', 'po', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'ps', 'pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5', 'w1', 'w10', 'w12', 'w13', 'w16', 'w18', 'w20', 'w21', 'w22', 'w24', 'w28', 'w3', 'w30', 'w31', 'w32', 'w34', 'w35', 'w37', 'w38', 'w41', 'w42', 'w43', 'w44', 'w45', 'w46', 'w47', 'w48', 'w49', 'w5', 'w50', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w62', 'w63', 'w66', 'w8', 'wo', 'i6', 'i7', 'i8', 'i9', 'ilx', 'p29', 'w29', 'w33', 'w36', 'w39', 'w4', 'w40', 'w51', 'w52', 'w53', 'w54', 'w6', 'w61', 'w64', 'w65', 'w67', 'w7', 'w9', 'pax', 'pd', 'pe', 'phx', 'plx', 'pmx', 'pnl', 'prx', 'pwx', 'w11', 'w14', 'w15', 'w17', 'w19', 'w2', 'w23', 'w25', 'w26', 'w27', 'pl0', 'pl4', 'pl3', 'pm2.5', 'ph4.4', 'pn40', 'ph3.3', 'ph2.6']

map_labels                  = None
num_classes                 = 19
shuffle                     = True
void_class                  = 255   # void id or value on the image

# Training
epochs                      = 2     # Max number of epochs, use 0 to save directly a model, useful to make conversions
ini_epochs                      = 1     # min
valid_samples_epoch         = 10    # Number of validation images used to validate an epoch

    ### Optimizer ###
optimizer                   = 'SGD' #Options available ['SGD','Adam','RMSProp']
momentum1                   = 0.95
momentum2                   = 0.99
learning_rate               = 1.0e-4
learning_rate_bias          = 1.0e-4
weight_decay                = 0.0005
    ### Scheduler
scheduler                   = 'ReduceLROnPlateau' # ['ReduceLROnPlateau','Step','MultiStep','Exponential', None]
decay                       = 0.1   # Learnng rate decay to apply (lr*decay)
sched_patience              = 5     # ReduceLROnPlateau option: epoch patience without loss change until a lr decrement
step_size                   = 20    # Step option: epoch counter to decrease lr
milestone                   = [60,30,10] # MultiStep option: define different milestones (epochs) to decrease lr
    ### Save criteria
save_condition              = 'valid_mIoU'        # ['always','(x)_loss','(x)_mAcc','(x)_mIoU'] x = valid or train_loss
                                                  # ['precision', 'recall', 'f1score' for classification]
    ### Early Stopping
early_stopping              = True
stop_condition              = 'valid_mIoU'        # [(x)_loss','(x)_mAcc','(x)_mIoU'] x = valid or train_loss
                                                  # ['precision', 'recall', 'f1score' for classification]
patience                    = 5

# Image preprocess
rescale                     = 1/255.
mean                        = [0.28689553, 0.32513301, 0.28389176] #[0.37296272, 0.37296272, 0.37296272]
std                         = [0.18696375, 0.19017339, 0.18720214]#[0.21090189, 0.21090189, 0.21090189]

# Data augmentation
hflips                      = True

# Tensorboard info
predict_to_save = 2         #
color_map = None
