#!bin/sh
# mkdir wild_est_new_data/Established\ Campground/
# mkdir wild_est_new_data/Wild\ Camping/

# CA
find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Established\ Campground/ -name '*satimg_CA_[0-4]*' -exec ln -s {} wild_est_new_data/Established\ Campground/ \;
find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Established\ Campground/ -name '*satimg_CA_[5-9]*' -exec ln -s {} wild_est_new_data/Established\ Campground/ \;

find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Wild\ Camping -name '*satimg_CA_[0-4]*' -exec ln -s {} wild_est_new_data/Wild\ Camping/ \;
find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Wild\ Camping -name '*satimg_CA_[5-9]*' -exec ln -s {} wild_est_new_data/Wild\ Camping/ \;

# # OR
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_OR/Established\ Campground/* wild_est_new_data/Established\ Campground/
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_OR/Wild\ Camping/* wild_est_new_data/Wild\ Camping/

# # CO
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CO/Established\ Campground/* wild_est_new_data/Established\ Campground/
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CO/Wild\ Camping/* wild_est_new_data/Wild\ Camping/

# # AZ
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Established\ Campground/* wild_est_new_data/Established\ Campground/

# # too long
# find ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Wild\ Camping -name '*satimg_AZ_[0-4]*' -exec ln -s {} wild_est_new_data/Wild\ Camping/ \;
# find ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Wild\ Camping -name '*satimg_AZ_[5-9]*' -exec ln -s {} wild_est_new_data/Wild\ Camping/ \;

# # UT
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_UT/Established\ Campground/* wild_est_new_data/Established\ Campground/
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_UT/Wild\ Camping/* wild_est_new_data/Wild\ Camping/

# # WA
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_WA/Established\ Campground/* wild_est_new_data/Established\ Campground/
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_WA/Wild\ Camping/* wild_est_new_data/Wild\ Camping/


