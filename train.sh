python train.py --data interactable --model m-doclayout --epoch 100 --image-size 1600  --batch-size 8 --project outputs --plot 1 --optimizer SGD --lr0 0.04 --pretrain interactable_detector.pt --device=0 --save-period 100 --val-period 10