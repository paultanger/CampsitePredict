#!bin/sh
# mkdir all_together/Eco-Friendly/
# mkdir all_together/Short-term\ Parking/
# mkdir all_together/Water/
# mkdir all_together/Informal\ Campsite/
# mkdir all_together/Showers/
# mkdir all_together/Wild\ Camping/

# CA
# ln -s sites_CA/Eco-Friendly/* all_together/Eco-Friendly/
# ln -s sites_CA/Short-term\ Parking/* all_together/Short-term\ Parking/
# ln -s sites_CA/Water/* all_together/Water/
# ln -s sites_CA/Informal\ Campsite/* all_together/Informal\ Campsite/
# ln -s sites_CA/Showers/* all_together/Showers/

# doesn't work too many
#ln -s sites_CA/Wild\ Camping/satimg_CA_[0-4] all_together/Wild\ Camping/
# try this instead
# find ./sites_CA/Wild\ Camping -name '*satimg_CA_[0-4]*' -exec ln -s {} all_together/Wild\ Camping/ \;
# find ./sites_CA/Wild\ Camping -name '*satimg_CA_[5-9]*' -exec ln -s {} all_together/Wild\ Camping/ \;

# OR
# ln -s sites_OR/Eco-Friendly/* all_together/Eco-Friendly/
# ln -s sites_OR/Short-term\ Parking/* all_together/Short-term\ Parking/
# ln -s sites_OR/Water/* all_together/Water/
# ln -s sites_OR/Informal\ Campsite/* all_together/Informal\ Campsite/
# ln -s sites_OR/Showers/* all_together/Showers/
# ln -s sites_OR/Wild\ Camping/* all_together/Wild\ Camping/

# CO
# ln -s sites_CO/Eco-Friendly/* all_together/Eco-Friendly/
# ln -s sites_CO/Short-term\ Parking/* all_together/Short-term\ Parking/
# ln -s sites_CO/Water/* all_together/Water/
# ln -s sites_CO/Informal\ Campsite/* all_together/Informal\ Campsite/
# ln -s sites_CO/Showers/* all_together/Showers/
# ln -s sites_CO/Wild\ Camping/* all_together/Wild\ Camping/

# AZ
# ln -s sites_AZ/Eco-Friendly/* all_together/Eco-Friendly/
# ln -s sites_AZ/Short-term\ Parking/* all_together/Short-term\ Parking/
# ln -s sites_AZ/Water/* all_together/Water/
# ln -s sites_AZ/Informal\ Campsite/* all_together/Informal\ Campsite/
# ln -s sites_AZ/Showers/* all_together/Showers/
# ln -s sites_AZ/Wild\ Camping/* all_together/Wild\ Camping/

# UT
# ln -s sites_UT/Eco-Friendly/* all_together/Eco-Friendly/
# ln -s sites_UT/Short-term\ Parking/* all_together/Short-term\ Parking/
# ln -s sites_UT/Water/* all_together/Water/
# ln -s sites_UT/Informal\ Campsite/* all_together/Informal\ Campsite/
# ln -s sites_UT/Showers/* all_together/Showers/
# ln -s sites_UT/Wild\ Camping/* all_together/Wild\ Camping/

# WA
# ln -s sites_WA/Eco-Friendly/* all_together/Eco-Friendly/
# ln -s sites_WA/Short-term\ Parking/* all_together/Short-term\ Parking/
# ln -s sites_WA/Water/* all_together/Water/
# ln -s sites_WA/Informal\ Campsite/* all_together/Informal\ Campsite/
# ln -s sites_WA/Showers/* all_together/Showers/
# ln -s sites_WA/Wild\ Camping/* all_together/Wild\ Camping/

