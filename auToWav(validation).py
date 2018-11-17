import sys
import os

# Store all command line args in genre_dirs
genre_dirs = ["rename/"]

for genre_dir in genre_dirs:
	# change directory to genre_dir
	# os.chdir(genre_dir)

	# echo contents before altering
	# print('Contents of ' + genre_dir + ' before conversion: ')
	# os.system("ls")

	# loop through each file in current dir
	for curr_dir in genre_dirs:
		for file in os.listdir(curr_dir):
			# SOX
			if file.endswith("au"):
				os.system("sox " + str(curr_dir + file) + " " + str(curr_dir + file[:-3]) + ".wav")

print("Done!!!")
