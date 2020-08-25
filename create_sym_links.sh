#!bin/sh
mkdir all_together/Eco-Friendly/
mkdir all_together/Short-term\ Parking/
mkdir all_together/Water/
mkdir all_together/Informal\ Campsite/
mkdir all_together/Showers/
mkdir all_together/Wild\ Camping/

# CA
# this works from the destination directory..
#ln -s ../../sites_CA/Eco-Friendly/* ./
# but this doesn't work
#ln -s ../../sites_CA/Eco-Friendly/* all_together/Eco-Friendly/
# what about this? - this works.. not relative.. 
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Eco-Friendly/* all_together/Eco-Friendly/
# reorganize and move into each dir separately
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Short-term\ Parking/* all_together/Short-term\ Parking/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Water/* all_together/Water/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Informal\ Campsite/* all_together/Informal\ Campsite/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Showers/* all_together/Showers/

# doesn't work too many
#ln -s sites_CA/Wild\ Camping/satimg_CA_[0-4] all_together/Wild\ Camping/
# try this instead
find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Wild\ Camping -name '*satimg_CA_[0-4]*' -exec ln -s {} all_together/Wild\ Camping/ \;
find ~/Desktop/github/CampsitePredict/data/sat_images/sites_CA/Wild\ Camping -name '*satimg_CA_[5-9]*' -exec ln -s {} all_together/Wild\ Camping/ \;

# OR
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_OR/Eco-Friendly/* all_together/Eco-Friendly/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_OR/Short-term\ Parking/* all_together/Short-term\ Parking/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_OR/Water/* all_together/Water/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_OR/Informal\ Campsite/* all_together/Informal\ Campsite/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_OR/Showers/* all_together/Showers/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_OR/Wild\ Camping/* all_together/Wild\ Camping/

# CO
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CO/Eco-Friendly/* all_together/Eco-Friendly/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CO/Short-term\ Parking/* all_together/Short-term\ Parking/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CO/Water/* all_together/Water/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CO/Informal\ Campsite/* all_together/Informal\ Campsite/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CO/Showers/* all_together/Showers/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_CO/Wild\ Camping/* all_together/Wild\ Camping/

# AZ
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Eco-Friendly/* all_together/Eco-Friendly/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Short-term\ Parking/* all_together/Short-term\ Parking/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Water/* all_together/Water/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Informal\ Campsite/* all_together/Informal\ Campsite/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Showers/* all_together/Showers/
# too long
# ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Wild\ Camping/* all_together/Wild\ Camping/
find ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Wild\ Camping -name '*satimg_AZ_[0-4]*' -exec ln -s {} all_together/Wild\ Camping/ \;
find ~/Desktop/github/CampsitePredict/data/sat_images/sites_AZ/Wild\ Camping -name '*satimg_AZ_[5-9]*' -exec ln -s {} all_together/Wild\ Camping/ \;

# UT
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_UT/Eco-Friendly/* all_together/Eco-Friendly/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_UT/Short-term\ Parking/* all_together/Short-term\ Parking/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_UT/Water/* all_together/Water/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_UT/Informal\ Campsite/* all_together/Informal\ Campsite/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_UT/Showers/* all_together/Showers/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_UT/Wild\ Camping/* all_together/Wild\ Camping/

# WA
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_WA/Eco-Friendly/* all_together/Eco-Friendly/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_WA/Short-term\ Parking/* all_together/Short-term\ Parking/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_WA/Water/* all_together/Water/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_WA/Informal\ Campsite/* all_together/Informal\ Campsite/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_WA/Showers/* all_together/Showers/
ln -s ~/Desktop/github/CampsitePredict/data/sat_images/sites_WA/Wild\ Camping/* all_together/Wild\ Camping/

