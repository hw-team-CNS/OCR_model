from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
import random
import cv2
import os

vowel_numbers = list(range(3077,3084))
vowel_numbers.extend([3168,3084,3169])
vowel_numbers.extend(list(range(3085,3093)))
vowel_numbers.remove(3085)
vowel_numbers.remove(3089)

consonant_numbers = list(range(3093,3130))
consonant_numbers.remove(3113)

deergalu_numbers = list(range(3134,3149))
deergalu_numbers.remove(3141)
deergalu_numbers.remove(3145)
deergalu_numbers.extend([3073,3074,3075])
deergalu_numbers.insert(0,3149)


vowels = []
for num in vowel_numbers:
    vowels.append(chr(num))
vowels.extend([vowels[0]+chr(3074),vowels[0]+chr(3075)])

consonants = []
for num in consonant_numbers:
    consonants.append(chr(num))

deergalu = []
for num in deergalu_numbers:
    deergalu.append(chr(num))

Guninthalu = []
for cons in consonants:
    lis = []
    for deerg in deergalu:
        lis.append(cons+deerg)
    Guninthalu.extend(lis)

#important character ik(sound)
ik = deergalu[0]
dvithvakshar = []
for conso in consonants:
    for cons in consonants:
        for deerg in deergalu:
            dvithvakshar.append(conso+ik+cons+deerg)


def crop(img_pillow):

    image1 = np.array(img_pillow)
    start_x = 0;
    end_x = image1.shape[1]-1;
    endx = end_x
    start_y = 0;
    end_y = image1.shape[0]-1;
    endy = end_y
    mid = 200
    lim = 7

    while(start_y < end_y):
        if (min(image1[start_y]) < mid):
            break;
        start_y+=1;

    while(end_y > start_y):
        if (min(image1[end_y]) < mid):
            break;
        end_y-=1;

    while(start_x < end_x):
        if (min(image1.T[start_x]) < mid):
            break;
        start_x+=1;

    while(end_x > start_x):
        if (min(image1.T[end_x]) < mid):
            break;
        end_x-=1;

    #padding on four sides
    if (start_x > lim):
        start_x = start_x-random.choice(list(range(0,lim)))
    else:
        start_x = start_x-random.choice(list(range(0,start_x)))

    if (start_y > lim):
        start_y = start_y-random.choice(list(range(0,lim)))
    else:
        start_y = start_y-random.choice(list(range(0,start_y)))

    if (endy-end_y > lim):
        end_y = end_y+random.choice(list(range(0,lim)))
    else:
        end_y = end_y+random.choice(list(range(0,endy-end_y)))

    if (endx-end_x > lim):
        end_x = end_x+random.choice(list(range(0,lim)))
    else:
        end_x = end_x+random.choice(list(range(0,endx-end_x)))
    
    im2 = img_pillow.crop((start_x,start_y,end_x,end_y))
    return im2

def Make_Image(word,font):
	# empty image
	img = 255 * np.ones((60, 400),np.uint8)
	img_pillow = Image.fromarray(img)

	draw = ImageDraw.Draw(img_pillow)
	draw.text((7, 7),  word, font = font)
	# img_pillow.save('./4_4_0.png')
	img1 = crop(img_pillow)
	return img1


# Adding noise to an image and changing the thickness
def modify_image(img):
	img_array = np.array(img)
	kernel = np.ones((1, 1), np.uint8)
	img_array = cv2.erode(img_array, kernel, iterations = 1)
	mean = 0
	sigma = 30
	guass = np.random.normal(mean,sigma,img_array.shape)
	img_array = img_array+guass
	return img_array
	

#Making IAM compatible Dataset
class DataProvider():
  "this class creates machine-written text for a word list. TODO: change getNext() to return your samples."

  def __init__(self, wordList,ImageList):
    self.wordList = wordList
    self.ImageList = ImageList
    self.idx = 0

  def hasNext(self):
    "are there still samples to process?"
    return self.idx < len(self.wordList)

  def getNext(self):
    "TODO: return a sample from your data as a tuple containing the text and the image"
    img = self.ImageList[self.idx]
    word = self.wordList[self.idx]
    self.idx += 1
    # cv2.putText(img, word, (2,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0), 1, cv2.LINE_AA)
    return (word, img)


def createIAMCompatibleDataset(dataProvider):
	"this function converts the passed dataset to an IAM compatible dataset"

	# create files and directories
	f = open('words.txt', 'w+')
	if not os.path.exists('sub'):
		os.makedirs('sub')
	if not os.path.exists('sub/sub-sub'):
		os.makedirs('sub/sub-sub')

	# go through data and convert it to IAM format
	ctr = 0
	while dataProvider.hasNext():
		sample = dataProvider.getNext()
		
		# write img
		cv2.imwrite('sub/sub-sub/sub-sub-%d.png'%ctr, sample[1])
		
		# write filename, dummy-values and text
		line = 'sub-sub-%d'%ctr + ' X X X X X X X ' + sample[0] + '\n'
		f.write(line)
		
		ctr += 1


# Download the following ttf files in the current folder
fontpaths = [r"resources/Gautami.ttf",
			r"resources/GIST-TLOT Manu Italic.ttf",
			r"resources/GIST-TLOT Manu Normal.ttf",
			r"resources/Pothana-2000.ttf",
			r"resources/Sree Krushnadevaraya.ttf",
			r"resources/Suguna.ttf",
            r"resources/Mandali.ttf"]

fonts = []
for fontpath in fontpaths:
    fonts.append(ImageFont.truetype(fontpath, 20))


#Guninthalu = simple letters
#dvithvakshar = complex letters

Simple_letters = vowels+consonants+Guninthalu
complex_letters = dvithvakshar

word_list = []
image_list = []
for word in Simple_letters[:3]:
	for index in tqdm(range(len(fonts))):
		word_list.append(word)
		img = Make_Image(word,fonts[index])
		img = modify_image(img)
		image_list.append(img)

#random shuffling of word_list and image_list
temp = list(zip(word_list, image_list)) 
random.shuffle(temp) 
word_list, image_list = zip(*temp)

#creating the IAM format dataset
dataProvider = DataProvider(word_list,image_list)
createIAMCompatibleDataset(dataProvider)

