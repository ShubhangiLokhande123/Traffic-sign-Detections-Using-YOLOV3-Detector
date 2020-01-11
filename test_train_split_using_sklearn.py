import glob, os

# variable declarations

dirName = '/home/o2i/image data/'

# User defined functions to create a list of file and sub directories 

def getListOfFiles(dirName):
 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    
    for entry in listOfFile:
       
        fullPath = os.path.join(dirName, entry)
        
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

# Call the function

listOfFiles = getListOfFiles(dirName)

#print(listOfFiles)

# Using sklearn
# make the array of 'listOfFiles' and then test_train_split 
from sklearn.model_selection import train_test_split
#Test, Train = [0, 0]
Test, Train = train_test_split(listOfFiles, test_size = 0.2, random_state = 0)

#print (Train)
#print (Test)


file_train = open('Train.txt', 'w')  
file_test = open('Test.txt', 'w')

for path in Train:
        file_train.write(path + "\n")

for path in Test:
        file_test.write(path + "\n")




