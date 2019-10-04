#***************** postgres install for linux ************
# example:
# bash create_postgres_testdb.sh mypassword
passw=$1
if [[ ! -z ${passw} ]]
then
  echo  WARNING: YOU MUST PROVIDE A password in argument 1
  exit
fi
#to do linux install:
sudo apt-get install postgresql postgresql-contrib
# start it
sudo service postgresql start
echo now execute the following commands manually
echo sudo -i -u postgres
echo createdb testdb
echo psql testdb
echo psql -d testdb -c "ALTER USER postgres WITH PASSWORD '${passw}';"
echo exit
#To stop postgres:
# $ root@machinename:~#  service postgresql stop

# now you will have a postgres database named testdb, with a username of postgres (default) and a password of billy
