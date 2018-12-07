# saa-aggression-detection

## Dataset Description

Dataset preparation consists of
1. Generating the frames
2. A folder for each set of video frames
3. Generated file splits

In this repo, I generated the file splits (.txt files) for 5 fold cross validation under the folder data. 

## Checkpoint file generation

Checkpoint file will be generated after each 5 training epochs in the form of a .pth.tar file. You can set the training to continue from an existing checkpoint by doing 

```
python main.py --resume <PATH TO CHECKPOINT FILE>
```