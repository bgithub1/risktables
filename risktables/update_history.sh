# execute the build_history --update True 
# example usage when use use a username and password to the db:
# bash update_history.sh dbusername dbpassword ~/Virtualenvs3/dashrisk2 10 # to add 10 most recent days of history
#
# example usage when use you DON'T us a username or password to the db:
# bash update_history.sh '' '' ~/Virtualenvs3/dashrisk2 65 # to add 65 most recent days of history
uname=$1
passw=$2
virt_env_folder="$3"
days_to_fetch=$4
source ${virt_env_folder}/bin/activate
python3 build_history.py  --update_table True  --username ${uname} --password ${passw} --databasename testdb --days_to_fetch ${days_to_fetch}