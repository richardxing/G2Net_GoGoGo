import os
from argparse import ArgumentParser
import yaml
import torch
import pickle5 as pickle

class BaseConfig:
    #logistic
    seed = 48
    target_size = 1
    gdrive = './input/'
    output_dir = "./output_model/"
    PL_folder = "./PL_fold/"
    use_checkpoint = False
    prev_model_folder = None    
    debug = False
    use_subset = False 
    subset_frac = 0.4
    # augmentation    
    vflip = False
    shuffle01 = False
    time_shift = False
    time_stretch = False
    shuffle_channels = False  # need normalization first
    add_gaussian_noise = False # need normalization first
    timemask = False
    shift_channel = False    
    pitch_shift = False
    use_mixup = False
    mixup_alpha = 0.1
    cropping = False
    
    use_MC=False
    MC_folds=64    
    # logger
    print_num_steps=350
    use_wandb=False
    # training related
    train_folds = [0,1,2,3,4]
    epochs = 6
    batch_size = 256    
    lr= 1e-2
    weight_decay= 1e-4
    gradient_accumulation_steps=1
    scheduler='cosineWithWarmUp' 
    # SWA
    use_swa = False
    swa_lr_ratio = 0 # in terms of max lr
    swa_lr = 0
    swa_start_step_epoch = 3
    swa_anneal_ratio =  999, # 999 means anneal til the end of the training
    # speedup
    num_workers = 7
    use_cudnn = True 
    use_dp=False  # dataparallel
    optim='Adam'
    warmup=0.1
    crit='bce'
    channels = 16
    proba_final_layer = 0.8
    sdrop=0
    PL_hard=False

    
class V2_Config(BaseConfig):
    model_name = 'TCNN_repr'
    model_version="V2_c16_sGW_vflip_sc01_PL_repr"
    model_module="V2"
    vflip=True
    shuffle01=True
    use_MC=True
    
    lr=7e-3
    checkpoint_folder=True
    epochs=6
    wandb_name = 'V2_c16_sGW_vflip_sc01_PL_repr'   

class V2_Config_pretrain(V2_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=2
    wandb_name = V2_Config.model_version + "_pretrain"     

class V2SD_Config(BaseConfig):
    model_name = 'TCNN_repr'
    model_version="V2SD_sGW_vflip_sc01_PL_repr"
    model_module="V2SD"
    vflip=True
    shuffle01=True
    channels = 32
        
    checkpoint_folder=True
    epochs=6
    wandb_name = 'V2SD_sGW_vflip_sc01_PL_repr'  

class V2SD_Config_pretrain(V2SD_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=2
    wandb_name = V2SD_Config.model_version + "_pretrain"     
    

class resnet34_Config(BaseConfig):
    model_name = 'Model_2D_repr'
    model_version = "resnet34_sGW2ep_PL_sc01_5ep_repr" 
    model_2D_encoder = 'resnet34'
    
    shuffle01=True
    checkpoint_folder=True
    epochs=4
    wandb_name = 'resnet34_sGW2ep_PL_sc01_5ep_repr'
    use_dp=True    

class resnet34_Config_pretrain(resnet34_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=2
    wandb_name = resnet34_Config.model_version + "_pretrain"     


class M3D_Config(BaseConfig):
    model_name = '3D_repr'
    model_version="3D_1ep1e2_repr"
    model_module="M3D"
    
    vflip=True
    shuffle01=True
    use_MC=False
    MC_folds=64
        
    lr=5e-3
    checkpoint_folder=None
    epochs=1
    wandb_name = '3D_1ep1e2_repr'
    use_dp=True

#M-1D, M-1DS32, M-1DC16, M-SD16, M-SD32
class M_1D_Config(BaseConfig):
    model_name = 'M-1D'
    model_version="M-1D"
    model_module="V2"
    vflip=True
    shuffle01=True
    channels = 32
        
    checkpoint_folder=True
    epochs=6
    wandb_name = 'M-1D'
    optim = 'RangerLars'
    warmup = 0
    sdrop = 0.05

class M_1D_Config_pretrain(V2SD_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=4
    wandb_name = M_1D_Config.model_version + "_pretrain"
    optim = 'Adam'
    warmup = 0.1

class M_1D_Config_adjust(V2SD_Config):
    checkpoint_folder=None
    epochs=3
    wandb_name = M_1D_Config.model_version + "_adjust"
    optim = 'RangerLars'
    warmup = 0
    crit = 'rank'
    PL_hard = True
    sdrop = 0


class M_1DC16_Config(BaseConfig):
    model_name = 'M-1DC16'
    model_version="M-1DC16"
    model_module="V2"
    vflip=True
    shuffle01=True
    channels = 16
        
    checkpoint_folder=True
    epochs=6
    wandb_name = 'M-1DC16'
    optim = 'RangerLars'
    warmup = 0
    sdrop = 0.05

class M_1DC16_Config_pretrain(V2SD_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=4
    wandb_name = M_1D_Config.model_version + "_pretrain"
    optim = 'Adam'
    warmup = 0.1

class M_1DC16_Config_adjust(V2SD_Config):
    checkpoint_folder=None
    epochs=2
    wandb_name = M_1D_Config.model_version + "_adjust"
    optim = 'RangerLars'
    warmup = 0
    crit = 'rank'
    PL_hard = True
    sdrop = 0


class M_1DS32_Config(BaseConfig):
    model_name = 'M-1DS32'
    model_version="M-1DS32"
    model_module="V2S"
    vflip=True
    shuffle01=True
    channels = 32
        
    checkpoint_folder=True
    epochs=6
    wandb_name = 'M-1DS32'
    optim = 'RangerLars'
    warmup = 0
    sdrop = 0.05

class M_1DS32_Config_pretrain(V2SD_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=4
    wandb_name = M_1D_Config.model_version + "_pretrain"
    optim = 'Adam'
    warmup = 0.1

class M_1DS32_Config_adjust(V2SD_Config):
    checkpoint_folder=None
    epochs=2
    wandb_name = M_1D_Config.model_version + "_adjust"
    optim = 'RangerLars'
    warmup = 0
    crit = 'rank'
    PL_hard = True
    sdrop = 0


class M_SD16_Config(BaseConfig):
    model_name = 'M-SD16'
    model_version="M-SD16"
    model_module="V2SD"
    vflip=True
    shuffle01=True
    channels = 16
        
    checkpoint_folder=True
    epochs=6
    wandb_name = 'M-SD16'
    optim = 'RangerLars'
    warmup = 0
    sdrop = 0.05
    proba_final_layer = 0.5

class M_SD16_Config_pretrain(V2SD_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=4
    wandb_name = M_1D_Config.model_version + "_pretrain"
    optim = 'Adam'
    warmup = 0.1

class M_SD16_Config_adjust(V2SD_Config):
    checkpoint_folder=None
    epochs=2
    wandb_name = M_1D_Config.model_version + "_adjust"
    optim = 'RangerLars'
    warmup = 0
    crit = 'rank'
    PL_hard = True
    sdrop = 0


class M_SD32_Config(BaseConfig):
    model_name = 'M-SD16'
    model_version="M-SD16"
    model_module="V2SD"
    vflip=True
    shuffle01=True
    channels = 32
        
    checkpoint_folder=True
    epochs=6
    wandb_name = 'M-SD16'
    optim = 'RangerLars'
    warmup = 0
    sdrop = 0.05
    proba_final_layer = 0.5

class M_SD32_Config_pretrain(V2SD_Config):
    checkpoint_folder=None
    PL_folder=None
    epochs=4
    wandb_name = M_1D_Config.model_version + "_pretrain"
    optim = 'Adam'
    warmup = 0.1

class M_SD32_Config_adjust(V2SD_Config):
    checkpoint_folder=None
    epochs=2
    wandb_name = M_1D_Config.model_version + "_adjust"
    optim = 'RangerLars'
    warmup = 0
    crit = 'rank'
    PL_hard = True
    sdrop = 0

#======================================================================================
#M-1D, M-1DS32, M-1DC16, M-SD16, M-SD32
config_dict = {
  'V2SD':V2SD_Config, 'V2SD_pretrain':V2SD_Config_pretrain, 'V2':V2_Config, 'V2_pretrain':V2_Config_pretrain,
  'resnet34':resnet34_Config, 'resnet34_pretrain':resnet34_Config_pretrain, 'M3D':M3D_Config,
  'M-1D':M_1D_Config, 'M-1D_pretrain':M_1D_Config_pretrain, 'M-1D_adjust':M_1D_Config_adjust,
  'M-1DC16':M_1DC16_Config, 'M-1DC16_pretrain':M_1DC16_Config_pretrain, 'M-1DC16_adjust':M_1DC16_Config_adjust,
  'M-1DS32':M_1DS32_Config, 'M-1DS32_pretrain':M_1DS32_Config_pretrain, 'M-1DS32_adjust':M_1DS32_Config_adjust,
  'M-SD16':M_SD16_Config, 'M-SD16_pretrain':M_SD16_Config_pretrain, 'M-SD16_adjust':M_SD16_Config_adjust,
  'M-SD32':M_SD32_Config, 'M-SD32_pretrain':M_SD32_Config_pretrain, 'M-SD32_adjust':M_SD32_Config_adjust,
  }
    
def read_config(name):
    print("Read Configuration")
    if name in config_dict: Config = config_dict[name]
    else:
        print(f"Configuration {name} is not found")
        return None

    Config.model_output_folder = Config.output_dir + Config.model_version + "/"
    if Config.checkpoint_folder:
        Config.checkpoint_folder = Config.output_dir + Config.prev_model_folder + "/" \
                                   if Config.prev_model_folder is not None else Config.model_output_folder    
    if Config.model_output_folder and not os.path.exists(Config.model_output_folder): 
        os.makedirs(Config.model_output_folder)
    if Config.debug:
        Config.epochs = 1
    torch.backends.cudnn.benchmark = Config.use_cudnn 
    print("Model Output Folder:", Config.model_output_folder)
    return Config    
    
def read_model_dict(model_module, config):
    print(model_module)
    if model_module == "V2":
        model_dict = dict(
            model_module=model_module,
            channels=config.channels,
            use_raw_wave=True,
            sdrop=config.sdrop
        )
    if model_module == "V2SD":
        model_dict = dict(
            model_module=model_module,
            channels=config.channels,
            proba_final_layer=config.proba_final_layer,
            use_raw_wave=True,
            sdrop=config.sdrop
        )
    if model_module == "resnet34":
        model_dict = dict(
            model_module=model_module,
            encoder="resnet34",
            use_raw_wave=True,
        )
    if model_module == "M3D":
        model_dict = dict(
            model_module=model_module,
            model_1D = 'V2',
            channels = config.channels,
            proba_final_layer = config.proba_final_layer,
            use_raw_wave=True, 
            model_1D_emb=128,
            model_1D_pretrain_dir = "./output_model/main_82nd_V2_c16_sGW_vflip_sc01_PL/",

            model_2D = 'resnet34',
            encoder="resnet34",
            model_2D_emb=128,
            model_2D_pretrain_dir = "./output_model/resnet34-sGW2ep-PL-sc01-5ep/",
            
            first=128,
            ps=0.1
        )
    return model_dict

def prepare_args():
    parser = ArgumentParser()
    parser.add_argument('--model_module', type=str, help='model module')
    parser.add_argument('--model_config', type=str, help='configuration name for this run')
    args = parser.parse_args()
    return args
    
