#!bin/sh
mkdir wild_est_new_data/Established\ Campground/
mkdir wild_est_new_data/Wild\ Camping/

# CA
# find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Established\ Campground/ -name '*satimg_CA_[0-4]*' -exec ln -s {} wild_est_new_data/Established\ Campground/ \;
# find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Established\ Campground/ -name '*satimg_CA_[5-9]*' -exec ln -s {} wild_est_new_data/Established\ Campground/ \;

# find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Wild\ Camping -name '*satimg_CA_[0-4]*' -exec ln -s {} wild_est_new_data/Wild\ Camping/ \;
# find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Wild\ Camping -name '*satimg_CA_[5-9]*' -exec ln -s {} wild_est_new_data/Wild\ Camping/ \;

# # MT
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_MT/Established\ Campground/* wild_est_new_data/Established\ Campground/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_MT/Wild\ Camping/* wild_est_new_data/Wild\ Camping/

# # ID
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_ID/Established\ Campground/* wild_est_new_data/Established\ Campground/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_ID/Wild\ Camping/* wild_est_new_data/Wild\ Camping/

# # NM
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_NM/Established\ Campground/* wild_est_new_data/Established\ Campground/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_NM/Wild\ Camping/* wild_est_new_data/Wild\ Camping/