# execute the build_history --update True 
# example usage:
# bash update_history.sh dbusername dbpassword ~/Virtualenv3/dashrisk2 10 # to add 10 most recent days of history
#
uname=$1
passw=$2
virt_env_folder="$3"
days_to_fetch=$4
source ${virt_env_folder}/bin/activate
python3 build_history.py  --update_table True  --username ${uname} --password ${passw} --databasename testdb --days_to_fetch ${days_to_fetch}