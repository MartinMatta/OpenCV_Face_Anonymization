# OpenCV Face Anonymization
 
 Face anonymization using c ++ and opencv on jetson {TX1, TX2, Xavier NX, AGX Xavier and Nano} in real time.
 
 ![system schema](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSaKCRVseFl8oQEWGriIe1GkBZ83rnjrlKb-oC-l677yv9SH-Ti&usqp=CAU)
 
 ### Prerequisites:
* opencv build with CUDA ≥ 4.2.0

### Download
```sh
$ git clone https://github.com/MartinMatta/OpenCV_Face_Anonymization
$ cd OpenCV_Face_Anonymization
```
### Build
```sh
$ mkdir build && cd build
$ cmake ..
$ make
```


### Run
```sh
$ ./main ../models/opencv_face_detector_uint8.pb ../models/opencv_face_detector.pbtxt
```

