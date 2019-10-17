# schhedule updates to the postgres database
# example: run at 5 AM, get 5 new prices, and use virtualenv Virtualenv3/dashrisk2
# bash pgup.sh postgres dbpassword 
# example: run at 7 PM, get 10 new prices, and use virtualenv Virtualenv3/mycustomenv
# bash pgup.sh postgres dbpassword 19 10 Virtualenv3/mycustomenv
#
username=${1}
if [[ -z ${username} ]];then
    echo arg1 must be database username
    exit -1
fi

password=${2}
if [[ -z ${password} ]];then
    echo arg2 must be database password
    exit -1
fi


hour_to_run=${3}
if [[ -z ${hour_to_run} ]];then
   hour_to_run=5
fi
days_to_fetch=${4}
if [[ -z ${days_to_fetch} ]];then
   days_to_fetch=5
fi

virtualenv="${5}"
if [[ -z ${virtualenv} ]];then
    virtualenv="~/Virtualenvs3/dashrisk2"
fi
source ${virtualenv}/bin/activate

python3 schedule_db_updates.py --hour ${hour_to_run} --username ${username} --password ${password} --days_to_fetch ${days_to_fetch}
