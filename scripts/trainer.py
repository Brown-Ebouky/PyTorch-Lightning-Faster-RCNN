import argparse
import sys
import os
from torch.utils.data import DataLoader
import albumentations as A
import pytorch_lightning as pl

from utils import load_config


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from datasets.coco_dataset import CocoDataset
from networks.faster_rcnn import RegionProposalNetwork, FasterRCNN

def main(argv):
    
    parser = argparse.ArgumentParser(
        description = "Train a Faster RCNN model"
    )
    parser.add_argument(
        "config_file",
        help="Path to the configuration file to use for the training"
    )
    parser.add_argument(
        "output_directory",
        help="Path to the output directory of the experiment"   
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)
    data_config = config["data"]
    network_config = config["network"]


    dataset = CocoDataset(
        data_config["dataDir"], data_config["dataType"],
        # network_config["window_size"], network_config["stride"]
    )

    train_loader = DataLoader(dataset, batch_size=1,)

    # model = RegionProposalNetwork(
    #     in_channels=3, sliding_window= network_config["window_size"],
    #     stride=network_config["stride"], n_proposals=network_config["n_proposals"]
    # )

    # TODO: automatically have the number of classes in COCO (Maybe config file?)

    model = FasterRCNN(
        network_config["sliding_window"], network_config["stride"],
        network_config["roi_output_size"], n_classes=80  
    )

    trainer = pl.Trainer(limit_train_batches=2, max_epochs=2, log_every_n_steps=1)
    trainer.fit(model = model, train_dataloaders= train_loader)





if __name__ == "__main__":
    main(sys.argv[1:])