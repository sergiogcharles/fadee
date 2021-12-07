"""Implementation of model-agnostic meta-learning for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd  # pylint: disable=unused-import
from torch.utils import tensorboard

import util

env = PupperGymEnv()
env.seed(args.seed)
torch.manual_seed(args.seed)

DEVICE = 'cuda'
class Reptile:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            num_outputs,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir
    ):
        meta_parameters = {}

        self._meta_parameters = meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0
        self.epsilon = 1e-4

    def _inner_loop(self, images, labels, train):   # pylint: disable=unused-argument
        """Computes the adapted network parameters via the MAML inner loop.

        Args:
            images (Tensor): task support set inputs
                shape (num_images, channels, height, width)
            labels (Tensor): task support set outputs
                shape (num_images,)
            train (bool): whether we are training or evaluating

        Returns:
            parameters (dict[str, Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
        """
        accuracies = []
        parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        # TODO: finish implementing this method.
        # This method computes the inner loop (adaptation) procedure for one
        # task. It also scores the model along the way.
        # Make sure to populate accuracies and update parameters.
        # Use F.cross_entropy to compute classification losses.
        # Use util.score to compute accuracies.
        for _ in range(self._num_inner_steps):
            outs = self._forward(images, parameters)
            loss = F.cross_entropy(outs, labels)
            accuracies.append(util.score(outs, labels))
            grads = torch.autograd.grad(loss, [v for _,v in self._meta_parameters.items()], create_graph=True)
            parameters = {
                pair[0]: pair[1] - self._inner_lrs[pair[0]]*grads[i]
                for i,pair in enumerate(parameters.items())
            }
        outs = self._forward(images, parameters)
        accuracies.append(util.score(outs, labels))
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        return parameters, accuracies

    def _outer_step(self, task_batch, train):  # pylint: disable=unused-argument
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        rewards_support_batch = []
        parameter_shift = {
            k: torch.zeros(v.shape)
            for k,v in self._meta_parameters.items()
        }

        for task in task_batch:
            images_support, labels_support, images_query, labels_query = task
            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
            # TODO: finish implementing this method.
            # For a given task, use the _inner_loop method to adapt, then
            # compute the MAML loss and other metrics.
            # Use F.cross_entropy to compute classification losses.
            # Use util.score to compute accuracies.
            # Make sure to populate outer_loss_batch, accuracies_support_batch,
            # and accuracy_query_batch.
            parameters, rewards = self._inner_loop(images_support, labels_support, train)
            parameter_shift = {
                k: v + parameters[k] - self._meta_parameters[k]
                for k,v in parameter_shift.items()
            }
            rewards_support_batch.append(rewards)
            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
        self._meta_parameters = {
            k: v + self.epsilon * parameter_shift[k]/len(task_batch)
            for k,v in self._meta_parameters.items()
        }
        rewards_support = np.mean(
            rewards_support_batch,
            axis=0
        )
        return rewards_support

    def train(self, dataloader_train, dataloader_val, writer):
        """Train the MAML.

        Consumes dataloader_train to optimize MAML meta-parameters
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query = (
                self._outer_step(task_batch, train=True)
            )
            outer_loss.backward()
            self._optimizer.step()

            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracies_support[0]:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracies_support[-1]:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_query:.3f}'
                )
                #print(outer_loss.item())
                writer.add_scalar('loss/train', outer_loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/pre_adapt_support',
                    accuracies_support[0],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_support',
                    accuracies_support[-1],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_query',
                    accuracy_query,
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for val_task_batch in dataloader_val:
                    outer_loss, accuracies_support, accuracy_query = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(
                    accuracies_pre_adapt_support
                )
                accuracy_post_adapt_support = np.mean(
                    accuracies_post_adapt_support
                )
                accuracy_post_adapt_query = np.mean(
                    accuracies_post_adapt_query
                )
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracy_pre_adapt_support:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracy_post_adapt_support:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_post_adapt_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/pre_adapt_support',
                    accuracy_pre_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_support',
                    accuracy_post_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_query',
                    accuracy_post_adapt_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            _, _, accuracy_query = self._outer_step(task_batch, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._meta_parameters = state['meta_parameters']
            self._inner_lrs = state['inner_lrs']
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/maml/omniglot.way:{args.num_way}.support:{args.num_support}.query:{args.num_query}.inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}.learn_inner_lrs:{args.learn_inner_lrs}.outer_lr:{args.outer_lr}.batch_size:{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    maml = MAML(
        args.num_way,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_train = omniglot.get_omniglot_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks
        )
        dataloader_val = omniglot.get_omniglot_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4
        )
        maml.train(
            dataloader_train,
            dataloader_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS
        )
        maml.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    main_args = parser.parse_args()
    main(main_args)
