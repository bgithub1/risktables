# execute the build_history.py --delete_schema True --fetch_from_yahoo True --build_table True
# example usage:
# bash update_history_from_scratch.sh dbusername dbpassword ~/Virtualenv3/dashrisk2 
#
uname=$1
passw=$2
virt_env_folder="$3"
source ${virt_env_folder}/bin/activate
python3 build_history.py   --delete_schema True --fetch_from_yahoo True --build_table True  --username ${uname} --password ${passw} --databasename testdb 
