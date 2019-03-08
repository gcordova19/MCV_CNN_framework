#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /home/grupo09/M5/MCV_CNN_framework # working directory
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH -p mhigh # Partition to submit to
#SBATCH -q masterhigh # Required to requeue other users mlow queue jobs
                      # With this parameter only 1 job will be running in queue mhigh
                      # By defaulf the value is masterlow if not defined
#SBATCH --mem 12000 # 4GB solicitados.
#SBATCH --gres gpu:Pascal:1 # 
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written


source activate python27
python main.py --exp_name Test2 --exp_folder Test2 --config_file config/classification_sample_vgg16_kitti.yml