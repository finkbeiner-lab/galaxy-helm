'''
Reads file and looks for x and y coordinates. Saves coordinate values to vector.
Vectors represent translation coordinates calculated by ITKv4/ImageRegistration1
'''

def txt_to_vector(path_itkreg_txt):
	''' Reads text file, searches for string, returns (x, y) vector.'''
	x_vector = [0]
	y_vector = [0]
	with open(path_itkreg_txt, 'r') as f:
		for line in f:
			if  'Translation X = ' in line:
				x_vector.append(int(round(float(line[16:23]), 0)))
			if  'Translation Y = ' in line:
				y_vector.append(int(round(float(line[16:23]), 0)))
	return zip(x_vector, y_vector)

if __name__ == '__main__':
	xy_vector = txt_to_vector(path_itkreg_txt)
