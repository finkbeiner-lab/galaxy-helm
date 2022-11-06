''' This interface sends Unix commands to run In-Silico Labeling TensorFlow program (Christiansen et al 2018). This script is to run prediction on multiple images.
'''

import subprocess
import argparse
import os
import sys
sys.path.append('/finkbeiner/imaging/smb-robodata/Sina/ISL_Scripts/')
import configure
from datetime import datetime
from time import strftime


def main():
	""" First the script makes sure the Bazel has been shutdown properly. Then it starts the bazel command with the following arguments:

	Args:
	crop_size: the image crop size the user chose the prediction to be done for.
	model_location: wheter the user wants to use the model that has been trained before in the program, or use their own trained model.
	output_path: The location where the folder (eval_eval_infer) containing the prediction image will be stored at.
	dataset_eval_path: The location where the images to be used for prediction are sotred at.
	infer_channels: The microscope inference channels.

	"""

	#Making sure the Bazel program has been shutdown properly.
	base_directory_path = 'cd '+ configure.base_directory + '; '
	# cmd1 = [base_directory_path + 'bazel version;']
	# process1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE)
	# process1.wait()

	print(os.listdir('/finkbeiner/imaging/home/in-silico/datasets/'),'\n')

	# Loop through subfolders in the dataset folder
	for folder in os.listdir('/finkbeiner/imaging/home/in-silico/datasets/'):

		print(folder)
		# use re.match
		if 'condition_' in folder:
			#Running Bazel for prediction. Note txt log files are also being created incase troubleshooting is needed.
			date_time = datetime.now().strftime("%m-%d-%Y_%H:%M")
			dataset_eval_path = str(os.path.join('/finkbeiner/imaging/home/in-silico/datasets/', folder))
			print("Dataset Eval Path is: ",dataset_eval_path,'\n')

			print("Bazel Launching", '\n')

			base_dir = 'export BASE_DIRECTORY=' + configure.base_directory + '/isl;  '

			baz_cmd = [base_directory_path + base_dir + 'bazel run isl:launch -- \
			--alsologtostderr \
			--base_directory $BASE_DIRECTORY \
			--mode EVAL_EVAL \
			--metric INFER_FULL \
			--stitch_crop_size '+ crop_size + ' ' +  configure.model_location + '\
			--output_path '+ output_path + ' \
			--read_pngs \
			--dataset_eval_directory ' + dataset_eval_path + ' \
			--infer_channel_whitelist ' + infer_channels + ' \
			--error_panels False \
			--infer_simplify_error_panels \
			> ' + output_path + '/predicting_output_'+ mod + '_'+ date_time +'_'+ crop_size +'_'+ folder + '_images.txt \
			2> ' + output_path + '/predicting_error_'+ mod + '_'+ date_time +'_'+ crop_size + '_'+ folder + '_images.txt;']

			process = subprocess.Popen(baz_cmd, shell=True, stdout=subprocess.PIPE)
			process.wait()

			print("Bazel Shutdown")

			#Here we shutdown the Bazel program.
			cmd3 = [base_directory_path + 'bazel shutdown;']
			process3 = subprocess.Popen(cmd3, shell=True, stdout=subprocess.PIPE)
			process3.wait()
		else:
			continue

			return


if __name__ == '__main__':

  #Receiving the variables from the XML script, parse them, initialize them, and verify the paths exist.

  # ----Parser-----------------------
  parser = argparse.ArgumentParser(description="ISL Predicting.")
  parser.add_argument("crop_size", help="Image Crop Size.")
  parser.add_argument("model_location", help="Model Location.")
  parser.add_argument("output_path", help="Output Image Folder location.")
  parser.add_argument("dataset_eval_path", help="Folder path to images directory.")
  parser.add_argument("infer_channels", help="Channel Inferences.")

  args = parser.parse_args()

  # ----Initialize parameters------------------
  crop_size = args.crop_size
  model_location = args.model_location
  output_path = args.output_path
  dataset_eval_path = args.dataset_eval_path
  infer_channels = args.infer_channels

  if model_location != '':
  	model_location = '--restore_directory ' + configure.model_location
  	mod = 'ISL-Model'
  else:
  	model_location = ''
  	mod = 'Your-Model'

	# ----Confirm given folders exist--
	if not os.path.exists(dataset_eval_path):
		print('Confirm the given path to input images (transmitted images used to generate prediction image) exists.')
		assert os.path.exists(dataset_eval_path), 'Path to input images (transmitted images used to generate prediction image).'
		if not os.path.exists(output_path):
			print('Confirm the given path to output of prediction for fluorescent and validation images exists.')

			assert os.path.abspath(output_path) != os.path.abspath(dataset_eval_path) , 'Please provide a unique data path.'
			assert os.path.abspath(output_path) != os.path.abspath(model_location),  'Please provide a unique model path.'

			print('\n The Evaluation Directory is:')
			print(dataset_eval_path)
			print('\n The Output Directory is:')
			print(output_path)

			main()


