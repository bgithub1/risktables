# load nvm, and node locally in an Ubuntu +18.04 server
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
source .bashrc
nvm install v14.10.1
# now install all dependencies in package.json
npm install