# Zoo of Unsupervised Image-to-Image Translation Networks (PyTorch)
# Instruction on running nets:
  ## Dependencies  
  Firstly you need to install dependencies:  
  `pip install -r requirements.txt`  
  ## Training your model    
  Now you can train any model of your choice using this line in your CLI:  
  `python run_trainer --yaml_path <your_config_file>`  
  Also if you want you can write all of your hyperparameters with hands in CLI.    
  ## Getting translated images
  Now, when you've trained some model and have your best checkpoint, you can get translated images with this line of code:  
  `python predict.py --yaml_path <your_config_file>`  
  As in the case of training you can write your hyperparameters in CLI by hands.  
  ## Config files   
  You can get standart config files from repo or change them by looking at lightning models argparse arguments.    
# Current implemented and working networks:  
- :heavy_check_mark: CycleGAN
- :heavy_check_mark: UGATIT
- :heavy_check_mark: MUNIT
- :x: StarGAN
- :x: FUNIT
