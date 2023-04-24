from pytorch.model.init import *
from pytorch.model.train.model_train import modelTrain
from pytorch.model.test.model_test import modelTest


# 按间距中的绿色按钮以运行脚本。

def load_model(checkpoint_path, state_dict):
    try:
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['training_state_dict']
        model.load_state_dict(checkpoint['model_state_dict'])
        fc_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        pass

    return state_dict


if __name__ == '__main__':

    # Define how many times each phase will be running
    PHASE_ONE = 100
    PHASE_TWO = 20
    PHASE_THREE = 10

    TEST = True

    # Define the phases
    if PHASE_ONE > 0:
        freeze_parameters(model)
        freeze_parameters(model.fc, False)

        fc_optimizer = torch.optim.Adagrad(model.fc.parameters(), lr=0.01, weight_decay=0.001)
        optimizers = [fc_optimizer]

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer, mode='min',
                                                                  factor=0.1, patience=5,
                                                                  threshold=0.01, min_lr=0.00001)

        checkpoint_path = gdrive_dir + "checkpoint_phase_one.pt"

        state_dict = modelTrain(PHASE_ONE, optimizers, lr_scheduler=lr_scheduler,
                                state_dict=None, accuracy_target=None,
                                checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, state_dict)

    if PHASE_TWO > 0:
        state_dict['trace_log'].append('PHASE TWO')

        freeze_parameters(model, False)

        conv_optimizer = torch.optim.Adagrad(model.parameters(), lr=0.0001, weight_decay=0.001)
        optimizers = [fc_optimizer, conv_optimizer]

        checkpoint_path = gdrive_dir + "checkpoint_phase_two.pt"

        state_dict = modelTrain(PHASE_TWO, optimizers, lr_scheduler=None,
                                state_dict=state_dict, accuracy_target=None,
                                checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, state_dict)

    if PHASE_THREE > 0:
        state_dict['trace_log'].append('PHASE THREE')

        freeze_parameters(model)
        freeze_parameters(model.fc, False)

        optimizers = [fc_optimizer]

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(fc_optimizer, milestones=[0], gamma=0.01)

        checkpoint_path = gdrive_dir + "checkpoint_phase_three.pt"

        state_dict = modelTrain(PHASE_THREE, optimizers, lr_scheduler=lr_scheduler,
                                state_dict=state_dict, accuracy_target=None,
                                checkpoint_path=checkpoint_path)

        print(*state_dict['trace_log'], sep="\n")

        state_dict = load_model(checkpoint_path, state_dict)

    if TEST:
        modelTest()

    save_checkpoint(gdrive_dir + 'checkpoint.pt')
