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


def Main(checkpoint_path, use_device, use_model, dataloader, state_dict=None):
    # Define how many times each phase will be running
    PHASE_ONE = 100
    PHASE_TWO = 20
    PHASE_THREE = 10

    # Define fc_optimizer
    fc_optimizer = torch.optim.Adagrad(use_model.fc.parameters(), lr=0.01, weight_decay=0.001)

    TEST = True
    # state_dict = load_model(checkpoint_dir + 'checkpoint_phase_one.pt', model, fc_optimizer)
    # model.to(device)

    # Define the phases
    if PHASE_ONE > 0:
        freeze_parameters(use_model)
        freeze_parameters(use_model.fc, False)

        fc_optimizer = torch.optim.Adagrad(use_model.fc.parameters(), lr=0.01, weight_decay=0.001)
        optimizers = [fc_optimizer]

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer, mode='min',
                                                                  factor=0.1, patience=5,
                                                                  threshold=0.01, min_lr=0.00001)

        checkpoint_path = checkpoint_path + "checkpoint_phase_one.pt"

        state_dict = modelTrain(PHASE_ONE, use_model, optimizers, criterion, use_device, dataloader,
                                lr_scheduler=lr_scheduler,
                                state_dict=None, accuracy_target=None,
                                checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, use_model, fc_optimizer)

    if PHASE_TWO > 0:
        state_dict['trace_log'].append('PHASE TWO')

        freeze_parameters(use_model, False)

        conv_optimizer = torch.optim.Adagrad(use_model.parameters(), lr=0.0001, weight_decay=0.001)
        optimizers = [fc_optimizer, conv_optimizer]

        checkpoint_path = checkpoint_path + "checkpoint_phase_two.pt"

        state_dict = modelTrain(PHASE_TWO, use_model, optimizers, criterion, use_device, dataloader, lr_scheduler=None,
                                state_dict=state_dict, accuracy_target=None,
                                checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, use_model, fc_optimizer)

    if PHASE_THREE > 0:
        state_dict['trace_log'].append('PHASE THREE')

        freeze_parameters(use_model)
        freeze_parameters(use_model.fc, False)

        optimizers = [fc_optimizer]

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(fc_optimizer, milestones=[0], gamma=0.01)

        checkpoint_path = checkpoint_path + "checkpoint_phase_three.pt"

        state_dict = modelTrain(PHASE_THREE, use_model, optimizers, criterion, use_device, dataloader,
                                lr_scheduler=lr_scheduler, state_dict=state_dict, accuracy_target=None,
                                checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, use_model, fc_optimizer)

    if TEST:
        modelTest(use_model, dataloader=dataloader['test_data'], device=use_device)

    save_checkpoint(model_name, output_size, hidden_layers, model, class_to_idx, cat_label_to_name,
                    checkpoint_path + 'checkpoint.pt')


if __name__ == '__main__':
    # 数据初始化
    model_name, output_size, hidden_layers, checkpoint_dir, data_dir, device, model, data_transforms, image_datasets, dataloaders, data_classes = initialize_model()
    cat_label_to_name, class_to_idx = load_labels(image_datasets)
    criterion, optimizer = init_cri_opti(model)
    # 分类的模型
    Main(checkpoint_dir, device, model, dataloaders)

    # 数据初始化
    checkpoint_dir, device, model, image_datasets, dataloaders = initialize_model(
        which_file='Diseases',
        which_model='checkpoint1',
        return_params=['checkpoint_dir', 'device', 'model', 'image_datasets', 'dataloaders'])

    # 病虫害的模型
    Main(checkpoint_dir, device, model, dataloaders)
