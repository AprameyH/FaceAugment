# FaceAugment
Facial Data Augmentation

## Running Face swapper
1. clone repository
2. pip install -r requirements.txt
3. cd localswap
4. python3 SwapSwap.py


### running finetune_person_test.ipynb
If you wish to recreate our results, then you can use the generated and base images folder that have included in our repo along with the test folder and run the script to get our results. Feel free to experiment.
Note, we are using the facenet-pytorch library,
run pip install facenet-pytorch to install it.
Second note, in order to get the fscore working, we have to alter the training file in facenet. I have included our altered facenet_training file in examples\new-facenet-training-script Replace the following file in your facenet package
facenet_pytorch\models\utils\training.py 
with this
examples\new-facenet-training-script

