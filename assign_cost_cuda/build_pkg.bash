cd ..
git submodule init
git submodule update
cd assign_cost_cuda
mkdir build
cd build
cmake ..
make
# export PYTHONPATH=$PWD:$PYTHONPATH
cd ..
cp build/*.so ./