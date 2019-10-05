# bash risktables.sh 8888 ~/Virtualenvs3/dashrisk2 ~/pyliverisk 127.0.0.1
flask_port=$1
if [[ -z ${flask_port} ]]
then
   flask_port=8888
fi

virtualenv_path="$2"i
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

source ${virtualenv_path}/bin/activate

source ${virtualenv_path}/bin/activate
cd ${workspace}/risktables/risktables
python3 -i dash_risk_v01.py --host ${mip} --port ${flask_port} --database_config_name dashrisk_jrtr
cd ~
deactivate