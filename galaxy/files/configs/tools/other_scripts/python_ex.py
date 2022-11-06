import sys

def main():

	string_to_print = sys.argv[1:len(sys.argv)]
	string_to_print = ' '.join(string_to_print)
	print(string_to_print)

	return string_to_print.upper()

if __name__ == '__main__':
	main()