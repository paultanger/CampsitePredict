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
.861

500 epochs
0.8786

test on new data from diff states:

Established Campground   0.4701
Wild Camping             0.5299

Established Campground    433
Wild Camping              488

'ID', 'MT', 'NM'

try sobel transformation

would need to reload dataset to run:
try bigger image size full 400
Validation size

# maybe later:
greyscale

clean data by eliminating images with a low pixel range?
diff zoom level?

Why were you working with this data set? What questions did you have at the beginning of the week? How did you go about working with your data? What tools, specifically, did you utilize and why? What were your findings? What problems did you face? How did you overcome them? What questions do you have now that you didn't have at the beginning of the week, pertaining to this data? OR What features/future implementations would improve your findings?

Use nlp to create tags to then label sat images to predict type of site based on sat images? For recommender as well as quality checker for data?

|                        | Category Description:                                                                                    |
|------------------------|----------------------------------------------------------------------------------------------------------|
| Established Campground | Campgrounds that advertise camping, irrespective of amenities provided.                                  |
| Informal Campsite      | An unadvertised camp spot associated with another business or institution (restaurant, gas station, etc) |
| Wild Camping           | A camping spot not linked to a business or institution irrespective of amenities or formal permissions.  |
| Water                  | A place to purchase potable water                                                                        |
| Short-term Parking     | Short-term, usually daytime, parking for overlander vehicles                                             |
| Eco-Friendly           | Places to help eco-friendly overlanders, like recycling centers, bio fuel stations, etc                  |

for NLP:

Category
Eco-Friendly                21
Established Campground    3739
Informal Campsite         2745
Short-term Parking          28
Showers                    264
Water                      490
Wild Camping              5124

silhouette scores not that diff.. best was 4 (despite other plots..)
but better with more..

so 9000 columns (features - words)
top 1000 SVD explain 73% of var
top 2000 88%

top 15 features for each cluster with 1000 max features:

0: camping, area, free, spot, beautiful, lake, nice, views, lots, great, site, large, camp, spots, open
1: water, station, potable, free, dump, hose, spigot, gallon, gas, park, drinking, filling, air, store, area
2: campground, sites, toilets, nice, water, pit, tables, lake, night, forest, site, free, creek, picnic, national
3: night, place, quiet, good, nice, highway, area, rest, park, spot, road, just, stay, rv, stop
4: river, road, nice, spot, right, access, sites, spots, pit, campground, free, area, small, quiet, beautiful
5: road, spot, dirt, forest, spots, just, service, nice, pull, great, right, view, little, gravel, views
6: showers, hot, shower, park, clean, rv, wifi, laundry, nice, campground, tent, pool, water, hookups, site
7: parking, lot, overnight, street, quiet, night, park, signs, free, rv, walmart, big, area, stayed, parked


top 15 features for each cluster with 10,000 max features:

0: area, night, camping, spot, quiet, great, good, nice, large, free, beautiful, views, road, view, highway
1: campground, sites, toilets, lake, nice, pit, forest, tables, free, water, night, creek, national, picnic, camping
2: water, station, potable, dump, free, hose, spigot, park, gallon, gas, drinking, area, rest, available, air
3: place, nice, night, good, quiet, road, great, free, near, view, just, big, rv, sleep, highway
4: showers, hot, park, shower, rv, clean, nice, campground, wifi, tent, sites, laundry, night, pool, hookups
5: road, spot, dirt, forest, spots, just, service, nice, camping, pull, little, right, quiet, small, gravel
6: river, road, nice, sites, right, access, spot, campground, pit, spots, free, area, quiet, night, forest
7: parking, lot, overnight, street, quiet, night, park, signs, free, area, rv, stayed, big, walmart, casino


all data:

top 15 features for each cluster with 10,000 max features:

0: water, station, potable, dump, free, spigot, hose, gas, park, drinking, available, rv, area, parking, gallon, rest, restrooms, picnic, air, building
1: campground, sites, pit, toilets, tables, lake, nice, picnic, water, river, free, forest, night, site, toilet, national, rings, small, 10, camping
2: place, area, night, nice, camping, quiet, spot, park, free, good, great, highway, river, just, big, beautiful, view, near, large, camp
3: road, spot, forest, dirt, spots, nice, just, quiet, good, service, camping, gravel, small, creek, little, right, pull, river, site, great
4: showers, hot, park, clean, rv, nice, campground, shower, laundry, sites, tent, night, water, wifi, site, hookups, pool, bathrooms, 25, hook
5: parking, lot, overnight, walmart, night, quiet, park, signs, street, big, area, rv, stayed, free, good, near, parked, stay, large, trucks

this cluster is actually a mix of two categories...

Cluster 5:
top features: place, night, area, nice, camping, quiet, spot, good, great, free, park, river, highway, beautiful, just, big, view, near, large, camp

     Wild Camping (2240 categories)
     Informal Campsite (1016 categories)
     Established Campground (888 categories)
     Water (48 categories)
     Showers (25 categories)
     Eco-Friendly (20 categories)
     Short-term Parking (11 categories)

if we make 6 instead of 5 it breaks better:

Cluster 0:
top features: water, station, potable, dump, free, spigot, hose, gas, park, drinking, rv, available, gallon, parking, area, air, building, right, rest, inside

     Water (398 categories)
     Established Campground (139 categories)
     Informal Campsite (95 categories)
     Wild Camping (32 categories)
     Showers (11 categories)
     Eco-Friendly (1 categories)
Cluster 1:
top features: showers, hot, clean, park, shower, laundry, nice, rv, campground, wifi, water, night, pool, site, tent, hookups, bathrooms, free, sites, available

     Established Campground (728 categories)
     Showers (210 categories)
     Informal Campsite (37 categories)
     Wild Camping (10 categories)
     Water (2 categories)
     Short-term Parking (1 categories)
Cluster 2:
top features: place, nice, night, quiet, good, road, great, stay, near, free, overnight, view, big, just, river, sleep, small, park, rv, highway

     Wild Camping (392 categories)
     Informal Campsite (180 categories)
     Established Campground (170 categories)
     Showers (9 categories)
     Water (6 categories)
     Short-term Parking (2 categories)
     Eco-Friendly (2 categories)
Cluster 3:
top features: camping, area, spot, night, nice, park, quiet, free, great, river, good, site, beautiful, large, just, camp, highway, big, spots, small

     Wild Camping (2029 categories)
     Established Campground (873 categories)
     Informal Campsite (831 categories)
     Water (44 categories)
     Showers (24 categories)
     Eco-Friendly (18 categories)
     Short-term Parking (10 categories)
Cluster 4:
top features: parking, lot, overnight, walmart, night, quiet, park, street, signs, stayed, big, area, rv, free, parked, near, good, 24, stay, large

     Informal Campsite (1230 categories)
     Wild Camping (480 categories)
     Established Campground (37 categories)
     Water (19 categories)
     Short-term Parking (13 categories)
     Showers (6 categories)
Cluster 5:
top features: road, spot, forest, dirt, spots, just, service, nice, quiet, gravel, good, creek, camping, little, small, pull, right, site, great, camp

     Wild Camping (1949 categories)
     Established Campground (173 categories)
     Informal Campsite (133 categories)
     Water (4 categories)
     Short-term Parking (1 categories)
Cluster 6:
top features: tables, picnic, pit, toilets, free, area, water, toilet, campground, nice, sites, rings, lake, pits, river, table, night, camping, quiet, rest

     Established Campground (409 categories)
     Informal Campsite (202 categories)
     Wild Camping (136 categories)
     Water (15 categories)
     Showers (3 categories)
     Short-term Parking (1 categories)
Cluster 7:
top features: sites, campground, nice, lake, water, night, river, park, site, toilets, forest, tent, state, 20, free, rv, 10, small, available, camping

     Established Campground (1210 categories)
     Wild Camping (96 categories)
     Informal Campsite (37 categories)
     Water (2 categories)
     Showers (1 categories)