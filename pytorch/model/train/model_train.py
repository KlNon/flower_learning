"""
@Project ：.ProjectCode 
@File    ：model_train
@Describe：在训练过程中也进行测试
@Author  ：KlNon
@Date    ：2023/3/29 22:03
"""

# 训练模型
import matplotlib.pyplot as plt
import time

from pytorch.model.args import *
from pytorch.model.init import criterion


def modelTrain(epochs, optimizers, lr_scheduler=None,
               dataloader=dataloaders, state_dict=None,
               checkpoint_path="checkpoint.pt", accuracy_target=None,
               show_graphs=True):
    if state_dict is None:
        state_dict = {
            'elapsed_time': 0,
            'trace_log': [],
            'trace_train_loss': [],
            'trace_train_lr': [],
            'valid_loss_min': np.Inf,
            'trace_valid_loss': [],
            'trace_accuracy': [],
            'epochs_trained': 0}
        state_dict['trace_log'].append('PHASE ONE')

    for epoch in range(1, epochs + 1):

        try:
            lr_scheduler.step()  # if instance of _LRScheduler
        except TypeError:
            try:
                if lr_scheduler.min_lrs[0] == lr_scheduler.optimizer.param_groups[0]['lr']:
                    break
                lr_scheduler.step(valid_loss)  # if instance of ReduceLROnPlateau
            except NameError:  # valid_loss is not defined yet
                lr_scheduler.step(np.Inf)
        except:
            pass  # do nothing

        epoch_start_time = time.time()
        #####################
        #       TRAIN       #
        #####################
        train_loss = 0
        model.train()
        for images, labels in dataloader['train_data']:
            # Move tensors to device
            images, labels = images.to(device), labels.to(device)

            # Clear optimizers
            [opt.zero_grad() for opt in optimizers]

            # Pass train batch through model feed-forward
            output = model(images)

            # Calculate loss for this train batch
            batch_loss = criterion(output, labels)
            # Do the backpropagation
            if batch_loss.requires_grad:
                batch_loss.backward()

            # Optimize parameters
            [opt.step() for opt in optimizers]

            # Track train loss
            train_loss += batch_loss.item() * len(images)

        # Track how many epochs has already run
        state_dict['elapsed_time'] += time.time() - epoch_start_time
        state_dict['epochs_trained'] += 1

        #####################
        #      VALIDATE     #
        #####################
        valid_loss = 0
        accuracy = 0
        top_class_graph = []
        labels_graph = []
        # Set model to evaluation mode
        model.eval()
        with torch.no_grad():
            for images, labels in dataloader['valid_data']:
                labels_graph.extend(labels)

                # Move tensors to device
                images, labels = images.to(device), labels.to(device)

                # Get predictions for this validation batch
                output = model(images)

                # Calculate loss for this validation batch
                batch_loss = criterion(output, labels)
                # Track validation loss
                valid_loss += batch_loss.item() * len(images)

                # Calculate accuracy
                output = torch.exp(output)
                top_ps, top_class = output.topk(1, dim=1)
                top_class_graph.extend(top_class.view(-1).to('cpu').numpy())
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item() * len(images)

        #####################
        #     PRINT LOG     #
        #####################

        # calculate average losses
        train_loss = train_loss / len(dataloader['train_data'].dataset)
        valid_loss = valid_loss / len(dataloader['valid_data'].dataset)
        accuracy = accuracy / len(dataloader['valid_data'].dataset)

        state_dict['trace_train_loss'].append(train_loss)
        try:
            state_dict['trace_train_lr'].append(lr_scheduler.get_lr()[0])
        except:
            state_dict['trace_train_lr'].append(
                optimizers[0].state_dict()['param_groups'][0]['lr'])
        state_dict['trace_valid_loss'].append(valid_loss)
        state_dict['trace_accuracy'].append(accuracy)

        # print training/validation statistics
        log = 'Epoch: {}: \
               lr: {:.8f}\t\
               Training Loss: {:.6f}\t\
               Validation Loss: {:.6f}\t\
               Validation accuracy: {:.2f}%\t\
               Elapsed time: {:.2f}'.format(
            state_dict['epochs_trained'],
            state_dict['trace_train_lr'][-1],
            train_loss,
            valid_loss,
            accuracy * 100,
            state_dict['elapsed_time']
        )
        state_dict['trace_log'].append(log)
        print(log)

        # save model if validation loss has decreased
        if valid_loss <= state_dict['valid_loss_min']:
            print('Validation loss decreased: \
                  ({:.6f} --> {:.6f}).   Saving model ...'
                  .format(state_dict['valid_loss_min'], valid_loss))

            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizers[0].state_dict(),
                          'training_state_dict': state_dict}
            if lr_scheduler:
                checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

            torch.save(checkpoint, checkpoint_path)
            state_dict['valid_loss_min'] = valid_loss

        if show_graphs:
            plt.figure(figsize=(25, 8))
            plt.plot(np.array(labels_graph), 'k.')
            plt.plot(np.array(top_class_graph), 'r.')
            # plt.show()
            plt.savefig('./output_data/Epoch_' + str(state_dict['epochs_trained']) + '_A.png')
            plt.close()

            plt.figure(figsize=(25, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.array(state_dict['trace_train_loss']), 'b', label='train loss')
            plt.plot(np.array(state_dict['trace_valid_loss']), 'r', label='validation loss')
            plt.plot(np.array(state_dict['trace_accuracy']), 'g', label='accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(np.array(state_dict['trace_train_lr']), 'b', label='train loss')

            # plt.show()
            plt.savefig('./output_data/Epoch_' + str(state_dict['epochs_trained']) + '_B.png')
            plt.close()

        # stop training loop if accuracy_target has been reached
        if accuracy_target and state_dict['trace_accuracy'][-1] >= accuracy_target:
            break

    return state_dict
