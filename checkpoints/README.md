# A*STAR Situational Awareness Analytics Aggression Detection
_by Sergi Adipraja Widjaja -- Conducted June - December 2018, a continuation of [saa-tsn-experiments](https://github.com/adiser/saa-tsn-experiments)_

# How to load checkpoint file

Sample code to read checkpoint file
```
# Instantiate TSN 
model = TSN(**args)
model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

# Load the checkpoint
checkpoint = torch.load('checkpoints/sample.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
```
