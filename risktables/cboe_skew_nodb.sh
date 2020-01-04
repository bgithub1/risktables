# leave out ip address and df_all_skew_csv_path arg to use default
# bash cboe_skew_nodb.sh 8600 ~/Virtualenvs3/dashrisk2 $(cd ../../;pwd) 
# use $(cd ../../;pwd) as the workspace and specify 127.0.0.1  and  df_all_skew_csv_path
# bash cboe_skew.sh  8700 ~/Virtualenvs3/dashrisk2 $(cd ../../;pwd) 127.0.0.1 ./df_all_skew_from_db.csv

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
   workspace="$(cd ../../;pwd)"
fi

mip=$(ifconfig|grep -A 1 eth0 | grep inet|egrep -o "addr[:][0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}"|egrep -o "[0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}")
if [[ ! -z $4 ]]
then
    mip="$4"
fi


df_all_skew_csv_path="${5}"
if [[ -z ${df_all_skew_csv_path} ]]
then
   df_all_skew_csv_path="./df_all_skew_from_db.csv"
fi

source ${virtualenv_path}/bin/activate
cd ${workspace}/risktables/risktables
python3  cboe_skew.py --df_all_skew_csv_path ${df_all_skew_csv_path} --additional_route /skew/
cd ~
deactivate