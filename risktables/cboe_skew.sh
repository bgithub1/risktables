# bash cboe_skew.sh 8600 ~/Virtualenvs3/dashrisk2 ~/pyliverisk 127.0.0.1
# use $(cd ../../;pwd) as the workspace
# bash cboe_skew.sh  8700 ~/Virtualenvs3/dashrisk2 $(cd ../../;pwd) 127.0.0.1
# specify a specific row in postgres_info.csv
# bash cboe_skew.sh  8700 ~/Virtualenvs3/dashrisk2 $(cd ../../;pwd) 127.0.0.1 secdb_aws
flask_port=$1
if [[ -z ${flask_port} ]]
then
   flask_port=8600
fi

virtualenv_path="$2"
if [[ -z ${virtualenv_path} ]]
then
   virtualenv_path="~/Virtualenvs3/dashrisk2"
fi

workspace="$3"
if [[ -z ${workspace} ]]
then
   workspace="~/pyliverisk"
fi



mip=$(ifconfig|grep -A 1 eth0 | grep inet|egrep -o "addr[:][0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}"|egrep -o "[0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}")
if [[ ! -z $4 ]]
then
    mip="$4"
fi

config_name="${5}"
if [[ -z ${config_name} ]]
then
   config_name="secdb_local"
fi

source ${virtualenv_path}/bin/activate
cd ${workspace}/risktables/risktables
python3  cboe_skew.py --host ${mip} --port ${flask_port} --database_config_name ${config_name} --additional_route /skew/
cd ~
deactivate