# bash risktables.sh 8888 ~/Virtualenvs3/dashrisk2 ~/pyliverisk 127.0.0.1
# use $(cd ../../;pwd) as the workspace
# bash risktables.sh 8700 ~/Virtualenvs3/dashrisk2 $(cd ../../;pwd) 127.0.0.1
# specify a specific row in postgres_info.csv
# bash risktables.sh 8700 ~/Virtualenvs3/dashrisk2 $(cd ../../;pwd) 127.0.0.1 dashrisk_local
# specify that you don't want to use postgres at all (only Yahoo Finance)
# bash risktables.sh 8700 ~/Virtualenvs3/dashrisk2 $(cd ../../;pwd) 127.0.0.1 nodb
flask_port=$1
if [[ -z ${flask_port} ]]
then
   flask_port=8888
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
   config_name="dashrisk_jrtr"
fi

source ${virtualenv_path}/bin/activate
cd ${workspace}/risktables/risktables

if [[ ${config_name} == 'nodb' ]]
then
    echo NOT using postgres.  Only using Yahoo Finance
    python3  dash_risk_v01.py --host ${mip} --port ${flask_port}  --additional_route /risk/
else
    echo using postgres config ${config_name}   Only using Yahoo Finance
    python3  dash_risk_v01.py --host ${mip} --port ${flask_port} --database_config_name ${config_name} --additional_route /risk/
fi

cd ~
deactivate