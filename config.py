import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Backdoor defense')

    # Dataset
    parser.add_argument("--data_path", type=str, default="/root/data/")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--num_data", type=int, default=50000)
    parser.add_argument("--num_class", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--train_ratio', type=float, default=1.0)

    # Base model
    parser.add_argument("--base_network", type=str, default="resnet18")
    parser.add_argument("--base_train_epoch", type=int, default=65)
    parser.add_argument("--base_train_lr", type=float, default=0.1)
    parser.add_argument("--base_train_scheduler_milestones", type=list, default=[25, 50])
    parser.add_argument("--base_train_scheduler_lambda", type=float, default=0.1)
    parser.add_argument('--train_bs', type=int, default=128)  # 128
    parser.add_argument('--test_bs', type=int, default=100)  # 100
    parser.add_argument('--resume', '-r', action='store_true', help="resume from checkpoint")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--use_pretrain_model", type=bool, default=False)

    # Backdoor attack
    parser.add_argument("--bd_ori_dict", type=bool, default=False)
    parser.add_argument("--data_mode", type=str, default='bd')  # bd,clean
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument('--trigger_type', type=str, default='badnets', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument("--trig_w", type=int, default=4)
    parser.add_argument("--trig_h", type=int, default=4)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--img_w", type=int, default=32)
    parser.add_argument("--img_h", type=int, default=32)
    parser.add_argument("--poison_rate", type=float, default=0.1)

    # Adversial attack
    parser.add_argument("--adv_epsilon", type=float, default=255 / 255)
    parser.add_argument("--adv_alpha", type=float, default=1 / 255)
    parser.add_argument("--adv_num_iterations", type=int, default=255)

    parser.add_argument("--label_rate", type=float, default=0.00001)
    parser.add_argument("--iso_save", type=int, default=1)

    # New Train
    parser.add_argument('--new_data_path', type=str, default='./new_data',
                        help='path of new data')
    parser.add_argument("--new_train_epoch", type=int, default=15)
    parser.add_argument("--new_train_lr", type=float, default=0.01)
    parser.add_argument('--retrain_lr', type=float, default=0.01)
    parser.add_argument("--retrain_epoch", type=int, default=15)
    parser.add_argument("--iso_ratio", type=list,
                        default=[0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01])
    parser.add_argument("--trigger_list", type=list, default=['badnets', 'blend', 'sig', 'wanet', 'trojan', 'dynamic',
                                                              'badnets', 'blend', 'sig', 'wanet', 'trojan', 'dynamic',
                                                              'badnets', 'blend', 'sig', 'wanet', 'trojan', 'dynamic',
                                                              'badnets', 'blend', 'sig', 'wanet', 'trojan', 'dynamic'])
    parser.add_argument("--drop_acc_threadhold", type=float, default=3.5)

    return parser
