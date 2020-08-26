model scores tracker:
acc:
0
1
fixed shuffle parameter
.68
increase batch size and nb filters from 32 to 40
.68
batch size 100, add dropout 0.3, filters to 32
oops predictions are uniform..

just do these categories:
Established Campground
Wild Camping
16232 files
0.7344

20 epochs on colab
.80 but overfit train data (.99)

add dropout .3
.825

dropout 0.5
.83

filters from 32 to 16
.82

filters to 64
.8287

try diff kernel and pool size
kernel from 3 to 4,4
.81

kernal to 2,2
.837

try diff optimizer sgd (.66 stuck in local min? need more epochs?) adadelta

add more layers?
this actually reduces trainable parameters.. tried with filters 64
.81

more augmentation
.76

more layers with increasing numbers of filters
.78

more epochs 100
.8484 !!!

200 epochs
.8672

shuffle 200

200 epochs with more filters in each layer..

try sobel transformation

would need to reload dataset to run:
try bigger image size full 400
Validation size

# maybe later:
greyscale

clean data by eliminating images with a low pixel range?
diff zoom level?

Why were you working with this data set? What questions did you have at the beginning of the week? How did you go about working with your data? What tools, specifically, did you utilize and why? What were your findings? What problems did you face? How did you overcome them? What questions do you have now that you didn't have at the beginning of the week, pertaining to this data? OR What features/future implementations would improve your findings?