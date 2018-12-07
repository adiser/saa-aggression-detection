# saa-aggression-detection
_by Sergi Adipraja Widjaja -- Conducted June - December 2018_

# Dataset Description

Dataset preparation consists of
1. Generating the frames
2. A folder for each set of video frames
3. Generated file splits

In this repo, I generated the file splits (.txt files) for 5 fold cross validation under the folder data. 

Refer to saa-tsn-experiments to know how to generate 

# Checkpoint file generation
Checkpoint file will be generated after each 5 training epochs in the form of a .pth.tar file. You can set the training to continue from an existing checkpoint by doing 

```
python main.py --resume <PATH TO CHECKPOINT FILE>
```

# Logging
Training logs will be generated in the form of a txt file. Unfortunately I havent used any visualization tool such as tensorboard to ease this process.

# Realtime Prototype
Run stream.py to run a realtime video prediction. It doesn't make use of any parallelization techniques hence speed can be quite slow. But good to understand and get a general feel of the algorithm 

Implementation detail:
* I fed a stack of images from a video to a buffer, this buffer can be invariant in length. Since this is a non-optimized version, a buffer size of 15 is used. In practice, the buffer size can be in the order of hundreds
* The model reads the buffered frames and generate a binary prediction

Run the demo by doing

```
python stream.py 
```

The default demo video shows a realtime prediction of two individuals involved in a fight
My python implementation of Rank Pooling Algorithm is also included inside in the form of a function accepting a stack of frames to be rankpooled.
```
frames = buffer.get()
arp(frames)
```
It makes use of the harmonic series to be able to generate the approximate rank pooled images



 
