from pytorch.model.model_init import *
from pytorch.model.label.model_load_label import load_labels
from pytorch.model.train.model_train import modelTrain
from pytorch.model.test.model_test import modelTest


# 按间距中的绿色按钮以运行脚本。

def load_model(checkpoint_path, use_model, fc_optimizer):
    try:
        checkpoint = torch.load(checkpoint_path)
        use_model.load_state_dict(checkpoint['model_state_dict'])
        fc_optimizer.load_state_dict(checkpoint[f'optimizer_state_dict'])
        state_dict = checkpoint['training_state_dict']
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}")
        return None

    return state_dict


def Main(checkpoint_dir_, device_, model_, dataloaders_, cat_label_to_name_, class_to_idx_, state_dict=None):
    # Define how many times each phase will be running
    PHASE_ONE = 100
    PHASE_TWO = 20
    PHASE_THREE = 10

    # Define fc_optimizer
    fc_optimizer = torch.optim.Adagrad(model_.fc.parameters(), lr=0.01, weight_decay=0.001)

    TEST = True
    # state_dict = load_model(checkpoint_dir + 'checkpoint_phase_one.pt', model, fc_optimizer)
    # model.to(device)

    # Define the phases
    if PHASE_ONE > 0:
        freeze_parameters(model_)
        freeze_parameters(model_.fc, False)

        fc_optimizer = torch.optim.Adagrad(model_.fc.parameters(), lr=0.01, weight_decay=0.001)
        optimizers = [fc_optimizer]

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer, mode='min',
                                                                  factor=0.1, patience=5,
                                                                  threshold=0.01, min_lr=0.00001)

        checkpoint_path = checkpoint_dir_ + "checkpoint_phase_one.pt"

        state_dict = modelTrain(PHASE_ONE, model_, optimizers, criterion, device_, dataloaders_,
                                lr_scheduler=lr_scheduler,
                                state_dict=None, accuracy_target=None,
                                checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, model_, fc_optimizer)

    if PHASE_TWO > 0:
        state_dict['trace_log'].append('PHASE TWO')

        freeze_parameters(model_, False)

        conv_optimizer = torch.optim.Adagrad(model_.parameters(), lr=0.0001, weight_decay=0.001)
        optimizers = [fc_optimizer, conv_optimizer]

        checkpoint_path = checkpoint_dir_ + "checkpoint_phase_two.pt"

        state_dict = modelTrain(PHASE_TWO, model_, optimizers, criterion, device_, dataloaders_, lr_scheduler=None,
                                state_dict=state_dict, accuracy_target=None,
                                checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, model_, fc_optimizer)

    if PHASE_THREE > 0:
        state_dict['trace_log'].append('PHASE THREE')

        freeze_parameters(model_)
        freeze_parameters(model_.fc, False)

        optimizers = [fc_optimizer]

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(fc_optimizer, milestones=[0], gamma=0.01)

        checkpoint_path = checkpoint_dir_ + "checkpoint_phase_three.pt"

        state_dict = modelTrain(PHASE_THREE, model_, optimizers, criterion, device_, dataloaders_,
                                lr_scheduler=lr_scheduler, state_dict=state_dict, accuracy_target=None,
                                checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

    if TEST:
        modelTest(model_, dataloader=dataloaders_['test_data'], device=device_)

    save_checkpoint(model_name, output_size, hidden_layers, model, class_to_idx_, cat_label_to_name_,
                    checkpoint_dir_ + 'checkpoint.pt')


if __name__ == '__main__':
    # 数据初始化
    model_name, output_size, hidden_layers, checkpoint_dir, data_dir, device, model, data_transforms, image_datasets, dataloaders, data_classes = initialize_model(
        which_file='Kind',
        which_model='checkpoint')
    cat_label_to_name, class_to_idx = load_labels(image_datasets, file_name='kind_cat_to_name.json')
    criterion, optimizer = init_cri_opti(model)
    # 分类的模型
    Main(checkpoint_dir, device, model, dataloaders, cat_label_to_name, class_to_idx)

    # 数据初始化
    checkpoint_dir, device, model, image_datasets, dataloaders = initialize_model(
        which_file='Diseases',
        which_model='checkpoint1',
        output_size=8,
        return_params=['checkpoint_dir', 'device', 'model', 'image_datasets', 'dataloaders'])
    # 读取json对应的label
    cat_label_to_name, class_to_idx = load_labels(image_datasets, file_name='diseases_cat_to_name.json')

    # 病虫害的模型
    Main(checkpoint_dir, device, model, dataloaders, cat_label_to_name, class_to_idx)
