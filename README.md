# cudaBlockTest
cuda speed up demo,select applicable blockDim.x and blockDim.y
running in windows, vs2019,cuda 11.1
three is two situation:
1、blockDim.x=32, blockDim.y=32
2、blockDim.x=1024, blockDim.y=1
result: situation 2 is faster in my computer(NVIDIA GeFore RTX 2070 super)
