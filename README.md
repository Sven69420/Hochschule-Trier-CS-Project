# Hochschule-Trier-CS-Project

## The file structure should be the following:

```plaintext
Workspace/
├── DynamicObjectFiltering.py
├── Helpers.py
├── ByteTrack_wrapper.py
├── Datasets/
│   ├── Images/
│   │   ├── Bicycle/
│   │   │   ├── Left/
│   │   │   └── Right/
│   │   └── Cars/
│   │       ├── Left/
│   │       └── Right/
│   ├── Videos/
│   └── Post_Processing/
│       ├── Images/
│       │   ├── Bicycle/
│       │   │   ├── Left/
│       │   │   └── Right/
│       │   └── Cars/
│       │       ├── Left/
│       │       └── Right/
│       └── VideoFrames/
├── Grounded_Sam/
│   ├── models/
│   └── results/
│       └── yolo_labels/

```

## ByteTrack

As for the implementation of ByteTrack, you might need to change the import location of ByteTrack, depending on your install version of ByteTrack.
I installed it via copying the whole github branch, hence my install location is not within the Workspace folder. (due to registry length issues)


## Grounded SAM

For the Grounded SAM models, just copy the one from CH10 given to us by Nicolas from the Hochschule Trier Robotics Lab. 
