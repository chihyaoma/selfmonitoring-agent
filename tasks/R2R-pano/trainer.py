import time
import math
import numpy as np

import torch
from utils import AverageMeter, load_datasets


class PanoSeq2SeqTrainer():
    """Trainer for training and validation process"""
    def __init__(self, opts, agent, optimizer, train_iters_epoch=100):
        self.opts = opts
        self.agent = agent
        self.optimizer = optimizer
        self.train_iters_epoch = train_iters_epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, train_env, tb_logger=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        dists = AverageMeter()
        movements = AverageMeter()
        val_losses = AverageMeter()
        val_acces = AverageMeter()

        print('Training on {} env ...'.format(train_env.splits[0]))
        # switch to train mode
        self.agent.env = train_env
        self.agent.encoder.train()
        self.agent.model.train()

        if self.opts.second_training:
            self.agent.model.first_stage_model.training = False

        self.agent.feedback = self.opts.feedback_training
        self.agent.value_loss = None
        self.agent.val_acc = None

        # load dataset path for computing ground truth distance
        self.agent.gt = {}
        for item in load_datasets(train_env.splits, self.opts):
            self.agent.gt[item['path_id']] = item

        end = time.time()
        for iter in range(1, self.train_iters_epoch + 1):
            # rollout the agent
            if self.opts.arch == 'self-monitoring':
                loss, traj = self.agent.rollout_monitor()
            elif self.opts.arch == 'speaker-baseline':
                loss, traj = self.agent.rollout()
            else:
                raise NotImplementedError()

            dist_from_goal = np.mean(self.agent.dist_from_goal)
            movement = np.mean(self.agent.traj_length)

            losses.update(loss.item(), self.opts.batch_size)
            dists.update(dist_from_goal, self.opts.batch_size)
            movements.update(movement, self.opts.batch_size)

            if self.agent.value_loss is not None:
                val_losses.update(self.agent.value_loss.item(), self.opts.batch_size)

            if self.agent.val_acc is not None:
                val_acces.update(np.mean(self.agent.val_acc), self.opts.batch_size)

            # zero the gradients before backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if tb_logger and iter % 10 == 0:
                current_iter = iter + (epoch - 1) * self.train_iters_epoch
                tb_logger.add_scalar('train/loss_train', loss, current_iter)
                tb_logger.add_scalar('train/dist_from_goal', dist_from_goal, current_iter)
                tb_logger.add_scalar('train/movements', movement, current_iter)
                if self.agent.value_loss is not None:
                    tb_logger.add_scalar('train/value_loss', self.agent.value_loss, current_iter)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, iter, self.train_iters_epoch, batch_time=batch_time,
                loss=losses))

        if tb_logger:
            tb_logger.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            tb_logger.add_scalar('epoch/train/loss', losses.avg, epoch)
            tb_logger.add_scalar('epoch/train/dist_from_goal', dists.avg, epoch)
            tb_logger.add_scalar('epoch/train/movements', movements.avg, epoch)
            if self.agent.value_loss is not None:
                tb_logger.add_scalar('epoch/train/val_loss', val_losses.avg, epoch)
            if self.agent.val_acc is not None:
                tb_logger.add_scalar('epoch/train/val_acc', val_acces.avg, epoch)


    def eval(self, epoch, val_env, tb_logger=None):
        batch_time = AverageMeter()
        losses = AverageMeter()
        dists = AverageMeter()
        movements = AverageMeter()
        val_losses = AverageMeter()
        val_acces = AverageMeter()

        env_name, (env, evaluator) = val_env

        print('Evaluating on {} env ...'.format(env_name))

        self.agent.env = env
        self.agent.env.reset_epoch()
        self.agent.model.eval()
        self.agent.encoder.eval()
        self.agent.feedback = self.opts.feedback
        self.agent.value_loss = None
        self.agent.val_acc = None

        # load dataset path for computing ground truth distance
        self.agent.gt = {}
        for item in load_datasets([env_name]):
            self.agent.gt[item['path_id']] = item
        val_iters_epoch = math.ceil(len(env.data) / self.opts.batch_size)
        self.agent.results = {}
        looped = False
        iter = 1

        with torch.no_grad():
            end = time.time()
            while True:

                if self.opts.progress_inference:
                    traj = self.agent.sample_progress_inference(self.opts.beam_size)
                elif self.opts.eval_beam:
                    traj = self.agent.sample_beam(self.opts.beam_size)
                else:
                    # rollout the agent
                    if self.opts.arch == 'self-monitoring':
                        loss, traj = self.agent.rollout_monitor()
                    elif self.opts.arch == 'speaker-baseline':
                        loss, traj = self.agent.rollout()
                    else:
                        raise NotImplementedError()

                    dist_from_goal = np.mean(self.agent.dist_from_goal)
                    movement = np.mean(self.agent.traj_length)

                    losses.update(loss.item(), self.opts.batch_size)
                    dists.update(dist_from_goal, self.opts.batch_size)
                    movements.update(movement, self.opts.batch_size)
                    if self.agent.value_loss is not None:
                        val_losses.update(self.agent.value_loss.item(), self.opts.batch_size)
                    if self.agent.val_acc is not None:
                        val_acces.update(np.mean(self.agent.val_acc), self.opts.batch_size)

                    if tb_logger and iter % 10 == 0:
                        current_iter = iter + (epoch - 1) * val_iters_epoch
                        tb_logger.add_scalar('{}/loss'.format(env_name), loss, current_iter)
                        tb_logger.add_scalar('{}/dist_from_goal'.format(env_name), dist_from_goal, current_iter)
                        tb_logger.add_scalar('{}/movements'.format(env_name), movement, current_iter)
                        if self.agent.value_loss is not None:
                            tb_logger.add_scalar('{}/val_loss'.format(env_name), self.agent.value_loss, current_iter)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, iter, val_iters_epoch, batch_time=batch_time,
                    loss=losses))

                # write into results
                for traj_ in traj:
                    if traj_['instr_id'] in self.agent.results:
                        looped = True
                    else:
                        result = {
                            'path': traj_['path'],
                            'distance': traj_['distance'],
                            'img_attn': traj_['img_attn'],
                            'ctx_attn': traj_['ctx_attn'],
                            'value': traj_['value'],
                            'viewpoint_idx': traj_['viewpoint_idx'],
                            'navigable_idx': traj_['navigable_idx']
                        }
                        self.agent.results[traj_['instr_id']] = result
                if looped:
                    break
                iter += 1

        if tb_logger:
            tb_logger.add_scalar('epoch/{}/loss'.format(env_name), losses.avg, epoch)
            tb_logger.add_scalar('epoch/{}/dist_from_goal'.format(env_name), dists.avg, epoch)
            tb_logger.add_scalar('epoch/{}/movements'.format(env_name), movements.avg, epoch)
            if self.agent.value_loss is not None:
                tb_logger.add_scalar('epoch/{}/val_loss'.format(env_name), val_losses.avg, epoch)
            if self.agent.val_acc is not None:
                tb_logger.add_scalar('epoch/{}/val_acc'.format(env_name), val_acces.avg, epoch)

        # dump into JSON file
        if self.opts.eval_beam:
            self.agent.results_path = '{}{}-beam_{}_{}_epoch_{}.json'.format(self.opts.results_dir, self.opts.exp_name,
                                                                             self.opts.beam_size, env_name, epoch)
        else:
            self.agent.results_path = '{}{}_{}_epoch_{}.json'.format(self.opts.results_dir, self.opts.exp_name,
                                                                     env_name, epoch)
        self.agent.write_results()
        score_summary, _ = evaluator.score(self.agent.results_path)
        result_str = ''
        success_rate = 0.0
        for metric, val in score_summary.items():
            result_str += '| {}: {} '.format(metric, val)
            if metric in ['success_rate']:
                success_rate = val
            if tb_logger:
                tb_logger.add_scalar('score/{}/{}'.format(env_name, metric), val, epoch)
        print(result_str)

        return success_rate