# FaceAugment
Facial Data Augmentation

# Running the GPT-4 prompt generator
- The prompt generation script is located in pipeline_methods/image_to_prompts.ipynb
- Replace the gpt api key and the path to the original image you want to use


# Running image generation with SDXL
- Download the Realism Engine SDXL model from https://civitai.com/models/152525
- The image generation script is in pipeline_methods/sdxl.py
- Set the path to the .safetensors file, output folder and list of prompts before running the python script


## Running Face swapper
- The code for face swapping is in pipeline_methods/face_swap.py
- Specify input and output files in the python code
- Run python3 pipeline_methods/face_swap.py


### Running finetune_person_test.ipynb
- If you wish to recreate our results, then you can use the generated and base images folder that have included in our repo along with the test folder and run the script to get our results. Feel free to experiment.
- We are using the facenet-pytorch library, run pip install facenet-pytorch to install it.
- In order to get the fscore working, we have to alter the training file in facenet. We have included our altered facenet_training file in examples\new-facenet-training-script
- Replace the contents of the following file in the installed facenet package
    facenet_pytorch\models\utils\training.py 
    with the contents of this file
    evaluation_methods\new-facenet-training-script

