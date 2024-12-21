import torch


@torch.no_grad()
def evaluate(model,data_loader,loss_func,eval_func,device,config,save_dir):
    model.eval()
    acc, eval_mloss = eval_func(model, data_loader, loss_func, device, config, save_dir)

    return acc, eval_mloss


def main(args):
    device = select_device(args.device)

    print("loading model weight...")
    model = torch.load(args.weight, map_location=device)
    model = model['model']

    print("loading config file...")
    with open(args.config, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    conf["EXPERIMENT_PATH"] = args.save
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    print("loading dataset function...")
    dataset_task = get_dataset_task(conf)

    print("loading evaluate function...")
    eval_function = get_evaluate_function(conf)

    print("loading test dataset...")
    test_loader = get_test_dataloader(conf, dataset_task, word_size=1)

    print('Evaluating Network.....')
    accuracy, average_loss = evaluate(model=model,
                                      data_loader=test_loader,
                                      loss_func=None,
                                      eval_func=eval_function,
                                      device=device,
                                      config=conf,
                                      save_dir=conf["EXPERIMENT_PATH"])
    print('Average loss: {:.4f}, Accuracy: {:.4f}'.format(average_loss, accuracy))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True, help='the weight file you want to validation.')
    parser.add_argument('--config', type=str, required=True, help='The configuration file of category name, .yaml or .txt')
    parser.add_argument('--device', type=str, default="0", help="'cpu' or gpu id:'0', default='0'")
    parser.add_argument('--save', type=str, default="runs/validation", help='validation result saving path. default="runs/validation"')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os
    import argparse
    import yaml
    from utils.select_tools import select_device, get_evaluate_function, get_dataset_task
    from utils.dataloader import get_test_dataloader
    args = parse_opt()
    main(args)
    print("end")
