import os
import argparse
import torch
import sys
import traceback
sys.path.append(os.path.abspath(".."))

from utils.utils import set_seed
from utils.learning_utils import validate_and_select
from data.dataset_cifar import build_loader
from predictor.SplitPredictor import Predictor
from model.model_cifar100 import build_model
from model.logitmodel import build_logit_model


def main():
    parser = argparse.ArgumentParser(description='ConformalAdapter')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--model', type=str, default='densenet121', help='model, e.g. resnet101, resnext50, densenet121')
    parser.add_argument('--data_dir', '-s', type=str, default='../datasets', help='location of CIFAR-100 dataset')
    parser.add_argument('--conformal', type=str, default='thr', help='score function, thr, aps')
    parser.add_argument('--alpha', type=float, default=0.03, help="error rate, improvements are more evident with smaller alpha values, e.g., 0.01.")
    parser.add_argument('--device', type=str, default='0' if torch.cuda.is_available() else 'cpu', help='device to run')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save model weights')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = main()
    output_dir = "experiment_results_cifar_thr"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{args.model}_{args.conformal}_alpha{args.alpha}.txt")

    print(f"Output file path: {output_file}")
    if not os.access(output_dir, os.W_OK):
        print(f"Error: No write permission for {output_dir}")
        sys.exit(1)

    try:
        with open(output_file, 'a') as f:
            f.write(f"--- Seed: {args.seed} ---\n")
            f.write(f"Test.py Arguments: {vars(args)}\n")
            f.write(f"Starting test.py execution at {os.path.abspath(output_file)}\n")
            f.flush()  

        print(f"Starting test.py with seed {args.seed}")
        set_seed(args.seed)
        model_name = args.model
        data_dir = args.data_dir
        conformal = args.conformal
        alpha = args.alpha
        device = f'cuda:{args.device}' if args.device.isdigit() else args.device
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        save_dir = os.path.join(base_dir, args.save_dir)
        dataset = 'cifar100'
        class_num = 100

        print(f"Creating save directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

        print("Building model")
        model = build_model(model_name, use_adapter=False)
        model = model.to(device)

        print("Loading data")
        validloader, calibloader, testloader = build_loader(data_dir, model, device=device)

        print("Building logit model")
        model = build_logit_model(class_num, use_adapter=False)
        model = model.to(device)

        print("Building adapter model")
        model_w_ada = build_logit_model(class_num, use_adapter=True)
        model_w_ada = model_w_ada.to(device)

        print("Validating and selecting")
        selected_iter = validate_and_select(model_w_ada, validloader, model_name, conformal, alpha, dataset, save_dir, class_num, device)

        print("Loading saved model")
        saved_model_path = f'{save_dir}/{model_name}_{dataset}_{selected_iter}iter.pth'
        print(f"Attempting to load model from {saved_model_path}")
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError(f"Model file {saved_model_path} does not exist")
        model_w_ada.base_model.load_state_dict(torch.load(saved_model_path))
        model_w_ada.eval()

        # performance w/o C-Adapter
        print("Evaluating w/o C-Adapter")
        predictor = Predictor(model, conformal, alpha, device)
        predictor.calibrate(calibloader)
        result = predictor.evaluate(testloader, alpha, class_num)
        print("performance w/o C-Adapter")
        print(result)
        with open(output_file, 'a') as f:
            f.write("Performance w/o C-Adapter:\n")
            f.write(f"{result}\n")
            f.flush()

        # performance w/ C-Adapter
        print("Evaluating w/ C-Adapter")
        predictor = Predictor(model_w_ada, conformal, alpha, device)
        predictor.calibrate(calibloader)
        result = predictor.evaluate(testloader, alpha, class_num)
        print("Performance w/ C-Adapter")
        print(result)
        with open(output_file, 'a') as f:
            f.write("Performance w/ C-Adapter:\n")
            f.write(f"{result}\n")
            f.write("\n")
            f.flush()

    except Exception as e:
        error_msg = f"Error in seed {args.seed}: {str(e)}\n{traceback.format_exc()}\n"
        print(error_msg)
        with open(output_file, 'a') as f:
            f.write(error_msg)
            f.flush()