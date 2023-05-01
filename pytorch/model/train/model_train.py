"""
@Project ：.ProjectCode 
@File    ：model_train
@Describe：训练模型,同时也验证模型(与直接测试不同,验证模型会影响到模型)
@Author  ：KlNon
@Date    ：2023/3/29 22:03
"""

# 训练模型
import matplotlib.pyplot as plt
import time

import numpy as np
import torch


def modelTrain(epochs, model, optimizers, criterion, device, dataloader,
               lr_scheduler=None, state_dict=None,
               checkpoint_path="checkpoint.pt", accuracy_target=None,
               show_graphs=True):
    def update_state_dict():
        nonlocal state_dict
        state_dict['elapsed_time'] += time.time() - epoch_start_time
        state_dict['epochs_trained'] += 1
        state_dict['trace_train_loss'].append(train_loss)
        state_dict['trace_train_lr'].append(lr)
        state_dict['trace_valid_loss'].append(valid_loss)
        state_dict['trace_accuracy'].append(accuracy)

    if state_dict is None:
        # Initialize state_dict if not provided
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

        # Train model
        epoch_start_time = time.time()
        train_loss = 0
        model.train()
        for images, labels in dataloader['train_data']:
            images, labels = images.to(device), labels.to(device)
            [opt.zero_grad() for opt in optimizers]
            output = model(images)
            batch_loss = criterion(output, labels)
            if batch_loss.requires_grad:
                batch_loss.backward()
            [opt.step() for opt in optimizers]
            train_loss += batch_loss.item() * len(images)

        # Validate model
        valid_loss, accuracy = 0, 0
        top_class_graph, labels_graph = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in dataloader['valid_data']:
                labels_graph.extend(labels)
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                batch_loss = criterion(output, labels)
                valid_loss += batch_loss.item() * len(images)
                output = torch.exp(output)
                top_ps, top_class = output.topk(1, dim=1)
                top_class_graph.extend(top_class.view(-1).to('cpu').numpy())
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item() * len(images)

        # Calculate average losses and accuracy
        train_loss /= len(dataloader['train_data'].dataset)
        valid_loss /= len(dataloader['valid_data'].dataset)
        accuracy /= len(dataloader['valid_data'].dataset)
        lr = optimizers[0].state_dict()['param_groups'][0]['lr']

        # Update state_dict
        update_state_dict()

        # Print training/validation statistics and save logs
        log = 'Epoch: {}: lr: {:.8f}\tTraining Loss: {:.6f}\tValidation Loss: {:.6f}\tValidation accuracy: {:.2f}%\tElapsed time: {:.2f}'.format(
            state_dict['epochs_trained'],
            state_dict['trace_train_lr'][-1],
            train_loss,
            valid_loss,
            accuracy * 100,
            state_dict['elapsed_time']
        )
        state_dict['trace_log'].append(log)
        print(log)

        # Save model if validation loss has decreased
        if valid_loss <= state_dict['valid_loss_min']:
            print('Validation loss decreased: ({:.6f} --> {:.6f}).   Saving model ...'.format(
                state_dict['valid_loss_min'], valid_loss))
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizers[0].state_dict(),
                          'training_state_dict': state_dict}
            if lr_scheduler:
                checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            state_dict['valid_loss_min'] = valid_loss

        # Display graphs
        if show_graphs:
            plt.figure(figsize=(25, 8))
            plt.plot(np.array(labels_graph), 'k.', label='true labels')
            plt.plot(np.array(top_class_graph), 'r.', label='guess labels')
            plt.savefig('./output_data/Epoch_' + str(state_dict['epochs_trained']) + '_A.png')
            plt.close()

            plt.figure(figsize=(25, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.array(state_dict['trace_train_loss']), 'b', label='train loss')
            plt.plot(np.array(state_dict['trace_valid_loss']), 'r', label='validation loss')
            plt.plot(np.array(state_dict['trace_accuracy']), 'g', label='accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(np.array(state_dict['trace_train_lr']), 'b', label='train loss')

            plt.savefig('./output_data/Epoch_' + str(state_dict['epochs_trained']) + '_B.png')
            plt.close()

        # Stop training loop if accuracy_target has been reached
        if accuracy_target and state_dict['trace_accuracy'][-1] >= accuracy_target:
            break

    return state_dict
