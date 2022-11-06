import os
import glob
import random
import ij.io.DirectoryChooser

def main():
	'''
	Takes all files with extension .arff in a given folder.
	Concatenates them to a merged.arff file.
	Returns path of the .arff files.
	'''

	output_name = "merged.arff"

	chooser = ij.io.DirectoryChooser("Select directory containing arff files")

	dir_path = chooser.getDirectory()
	print dir_path

	os.chdir(dir_path)
	files = glob.glob("*arff")
	print files

	## Grab the header info which is assumed to be shared between all arff files
	header_file = open(os.path.join(dir_path, files[0]), 'r') 

	header = []
	for line in header_file:
		# Stop when it goes past "@Ddata" in the header file.
		# but preserve spacing from newlines/
		if len(line) != 1 and not line.startswith("@"):
			break
		header.append(line)

	header_file.close()

	print header

	## Concatenate all the data
	data = []
	for filename in files:
		f = open(os.path.join(dir_path, filename), 'r')
		for line in f:
				if len(line) == 1 or line.startswith("@"):
					continue
				if random.randint(0, 199) < 1:
					data.append(line)

	print data
	output_file = open(os.path.join(dir_path, output_name), 'w')
	for line in header:
		output_file.write(line)

	for line in data:
		output_file.write(line)

	output_file.close()

	return dir_path

main()