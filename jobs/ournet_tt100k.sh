#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /home/grupo09/M5/sergi/MCV_CNN_framework_basem  # working directory
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH -p mhigh # Partition to submit to
#SBATCH -q masterhigh # Required to requeue other users mlow queue jobs
                      # With this parameter only 1 job will be running in queue mhigh
                      # By defaulf the value is masterlow if not defined
#SBATCH --mem 12000 # 4GB solicitados.
#SBATCH --gres gpu:Pascal:1 #
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written


python main.py --exp_name exp1_small_net --exp_folder OurNet_tt100k --config_file config/classification_ournet_tt100k.yml
