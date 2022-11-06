import pprint, pickle, sys

def main():
	infile = sys.argv[1]
	path_dict = pickle.load(open(infile, 'rb'))
	pprint.pprint(path_dict.items())

if __name__ == '__main__':
	main()