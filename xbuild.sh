cd /home/klay/MNN/build
make clean
cmake -DMNN_OPENCL=ON -DMNN_BUILD_DEMO=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  -DCMAKE_BUILD_TYPE=Debug  -DMNN_SEP_BUILD=OFF ..
make -j${nproc}
rm /home/klay/MNN/out.png
./multiPose.out ../pose.mnn ../input.png ../out.png 