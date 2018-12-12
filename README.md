# A*STAR Situational Awareness Analytics Aggression Detection
_by Sergi Adipraja Widjaja -- Conducted June - December 2018, a continuation of [saa-tsn-experiments](https://github.com/adiser/saa-tsn-experiments)_

# Dataset Preparation

Dataset preparation consists of
1. Generating the frames
2. A folder for each set of video frames
3. Generated file splits

In this repo, I generated the file splits (.txt files) for 5 fold cross validation under the folder data. 

Refer to saa-tsn-experiments to generate frames for videos. Your dataset structure should look something like the following

```
frames/aggressive/
    ├── video_frames_k
    │   ├── example_custom_prefix_00001.jpg
    │   ├── example_custom_prefix_00002.jpg
    │   ├── example_custom_prefix_00003.jpg
    │   ├── example_custom_prefix_00004.jpg
    │   ├── example_custom_prefix_00005.jpg
frames/passive/
    ├── video_frames_k
    │   ├── example_custom_prefix_00001.jpg
    │   ├── example_custom_prefix_00002.jpg
    │   ├── example_custom_prefix_00003.jpg
    │   ├── example_custom_prefix_00004.jpg
    │   ├── example_custom_prefix_00005.jpg
```

You can generate your own file splits for other datasets that you have. Simply place the negative examples and the positive examples in *separate* directories and run the following command
```
python generate_file_splits.py --num_splits 5 --pos_path <PATH_TO_POSITIVE_EXAMPLES> --neg_path <PATH_TO_NEGATIVE_EXAMPLES>
```
The above script will generate 5 file splits based on stratified sampling. The file splits is generated under the data/ directory. On top of the path to frames, it provides additional information such as the labels and the number of frames contained in each directory. It enables itself as an interface to the code and the dataset object to know how many frames are in each video prior to feeding it to the architecture

# Training and Testing
Here's a sample training command:

```
python main.py aggression RGB data/cctv_train_split_0.txt \
             data/cctv_test_split_0.txt --arch BNInception \
             --lr 0.001 --lr_steps 10 15 --epochs 100 --gd 20 -b 15 -j 4 \
             --pretrained_on_kinetics 1 --snapshot_pref cctv_0 
```
The above example command does the following:
- Train the *RGB* stream of the network,
- Simultanously performing cross validating with the *0th (first)* split of the dataset, 
- Using *BNInception* as the base architecture
- With a learning rate of *0.001*, decreasing by a factor of 10 on the *10th* and *15th* epochs.
- With *20* as a maximum value for the backpropagated gradient
- Batch size of *15*
- Pretraining with kinetics 

No testing command is provided due to several reasons:
1. Performance in test data is observable in the testing logs during training
2. Testing performance is *suspiciously* high in the order of 99% accuracy
3. Alternatively, we can observe the real time performance of the algorithm explained in the subsequent part of this README

Training logs will be generated in the form of a [txt file](./logs/). It shows the training and testing logs. The models can achieve as high as 99% testing accuracy. The robustness of the model on a realtime video is not yet quantified. Hence I attached a realtime video inference prototype to demonsrate how it performs on realtime video

The training logs are formatted as follows
```
<EPOCH> <ITERATION> <LOSS> <ACCURACY>
```

While the testing logs are formatted as follows

```
<ITERATIONS> <LOSS> <ACCURACY>
```

Checkpoint file will be generated after each 5 training epochs in the form of a .pth.tar file. You can set the training to continue from an existing checkpoint by doing 
```
python main.py <ARGS>--resume <PATH TO CHECKPOINT FILE>
```

# Realtime Prototype
Run stream.py to run a realtime video prediction. It doesn't make use of any parallelization techniques hence speed can be quite slow. But good to understand and get a general feel of the algorithm 

Implementation detail:
* I fed a stack of images from a video to a buffer, this buffer can be invariant in length. Since this is a non-optimized version, a buffer size of 15 is used. In practice, the buffer size can be in the order of hundreds
* The model reads the buffered frames and generate a binary prediction

Run the demo by doing

```
python stream.py 
```

The default demo video shows a realtime prediction of two individuals involved in a fight. The python implementation of Rank Pooling Algorithm is also included inside in the form of a function accepting a stack of frames to be rankpooled.
```
frames = buffer.get()
arp(frames)
```
It makes use of the harmonic series to be able to generate the approximate rank pooled images



 
