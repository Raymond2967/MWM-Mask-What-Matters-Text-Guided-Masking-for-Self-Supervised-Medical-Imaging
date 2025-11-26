import datetime
import time

import torch
import torch.distributed as tdist
from timm.utils import ModelEmaV2
from torch.utils.tensorboard import SummaryWriter

from arg import get_args, FineTuneArgs
from models import ConvNeXt, ResNet
__for_timm_registration = ConvNeXt, ResNet
from lr_decay import lr_wd_annealing
from util import init_distributed_environ, create_model_opt, load_checkpoint, save_checkpoint
from data import create_classification_dataset

# 引入 scikit-learn 用于计算 AUPRC, AUROC, Precision, Recall, F-scores
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import numpy as np
import sys



def main_ft():
    world_size, global_rank, local_rank, device = init_distributed_environ()
    args: FineTuneArgs = get_args(world_size, global_rank, local_rank, device)
    # print(f'初始参数:\n{str(args)}') # 暂时不打印初始参数，太冗长

    # Initial log_epoch call. At this point, only environment/setup info will be logged.
    # Metric fields will likely be default values (0.0 or N/A) until first evaluation.
    args.log_epoch()
    
    criterion, mixup_fn, model_without_ddp, model, model_ema, optimizer = create_model_opt(args)
    
    # 修改 load_checkpoint，使其能返回加载信息
    ep_start, performance_desc = 0, None
    if args.resume_from:
        print(f"[*] 检查点加载中... 路径: {args.resume_from}")
        ep_start, performance_desc, load_info = load_checkpoint(args.resume_from, model_without_ddp, model_ema, optimizer)
        print(f"[*] 检查点加载完成。加载了 {load_info['loaded_keys']} 个键，缺失 {load_info['missing_keys']} 个键。")
        if load_info['missing_keys']:
            print(f"[*] 缺失的键包括: {load_info['missing_keys']}")
        if load_info['unexpected_keys']:
            print(f"[*] 意外的键包括: {load_info['unexpected_keys']}")

    if ep_start >= args.ep: # load from a complete checkpoint file if args.resume_from:
        print(f'[*] [微调已完成] 最大/上次准确率: {performance_desc}')
    else:
        tb_lg = SummaryWriter(args.tb_lg_dir) if args.is_master else None
        
        print(f"[*] 数据集创建中... 数据路径: {args.data_path}")
        loader_train, iters_train, iterator_val, iters_val, num_classes, total_train_samples, total_val_samples = create_classification_dataset(
            args.data_path, args.img_size, args.rep_aug,
            args.dataloader_workers, args.batch_size_per_gpu, args.world_size, args.global_rank
        )
        # 打印数据读取情况
        print(f"[*] 数据读取完成。训练集样本数: {total_train_samples}，验证集样本数: {total_val_samples}。类别数: {num_classes}。")
        
        # Train & eval - initial evaluation
        # 修改 evaluate 函数的调用以获取原始预测结果
        all_targets_val, all_outputs_val, tot_pred_val, last_acc = evaluate(args.device, iterator_val, iters_val, model, num_classes)
        
        # 初始的最大准确率
        max_acc = last_acc 

        # 计算并打印初始的 AUPRC 和 AUROC
        initial_metrics = calculate_metrics(all_targets_val, all_outputs_val, num_classes)
        
        # *** NEW *** 初始化并更新 args 中与 AUPRC, AUROC 和 F-Scores 相关的最佳/当前值
        max_macro_auprc = initial_metrics["macro_auprc"]
        max_macro_auroc = initial_metrics["macro_auroc"]
        # F-Scores
        max_macro_f0_5 = initial_metrics["macro_f0_5_score"]
        max_macro_f1 = initial_metrics["macro_f1_score"]
        max_macro_f2 = initial_metrics["macro_f2_score"]

        args.best_macro_auprc = max_macro_auprc
        args.best_macro_auroc = max_macro_auroc
        args.current_macro_auprc = initial_metrics["macro_auprc"]
        args.current_macro_auroc = initial_metrics["macro_auroc"]
        
        # F-Scores
        args.best_macro_f0_5 = max_macro_f0_5
        args.best_macro_f1 = max_macro_f1
        args.best_macro_f2 = max_macro_f2
        args.current_macro_f0_5 = initial_metrics["macro_f0_5_score"]
        args.current_macro_f1 = initial_metrics["macro_f1_score"]
        args.current_macro_f2 = initial_metrics["macro_f2_score"]


        # *** NEW *** 初始化最佳每类分数
        args.best_class_auprcs = initial_metrics["class_auprcs"]
        args.best_class_auroc_scores = initial_metrics["class_auroc_scores"]
        # F-Scores
        args.best_class_f0_5_scores = initial_metrics["class_f0_5_scores"]
        args.best_class_f1_scores = initial_metrics["class_f1_scores"]
        args.best_class_f2_scores = initial_metrics["class_f2_scores"]


        # 设置cur_ep，以便 log_epoch在初始评估后记录正确的epoch信息
        args.cur_ep = f'{ep_start}/{args.ep}'
        args.train_loss, args.train_acc = 0.0, 0.0 # 初始时训练损失和准确率未知或为0
        args.best_val_acc = max_acc
        args.s_finish_time = time.strftime("%m-%d %H:%M", time.localtime()) # 初始化一下
        args.remain_time = '-' # 初始化一下
        args.log_epoch() # Log initial evaluation metrics

        print(f'[微调] 初始准确率: {last_acc:.2f}%')
        print(f'[微调] 初始 AUPRC (宏平均): {initial_metrics["macro_auprc"]:.4f}')
        print(f'[微调] 初始 AUROC (宏平均): {initial_metrics["macro_auroc"]:.4f}')
        print(f'[微调] 初始 F0.5-score (宏平均): {initial_metrics["macro_f0_5_score"]:.4f}')
        print(f'[微调] 初始 F1-score (宏平均): {initial_metrics["macro_f1_score"]:.4f}')
        print(f'[微调] 初始 F2-score (宏平均): {initial_metrics["macro_f2_score"]:.4f}')
        print(f'[微调] 初始各类别 AUPRC: {initial_metrics["class_auprcs"]}')
        print(f'[微调] 初始各类别 AUROC: {initial_metrics["class_auroc_scores"]}')
        print(f'[微调] 初始各类别 F0.5-score: {initial_metrics["class_f0_5_scores"]}')
        print(f'[微调] 初始各类别 F1-score: {initial_metrics["class_f1_scores"]}')
        print(f'[微调] 初始各类别 F2-score: {initial_metrics["class_f2_scores"]}')
        
        ep_eval = set(range(0, args.ep//3, 5)) | set(range(args.ep//3, args.ep))
        print(f'[微调开始] 评估周期: {sorted(ep_eval)}')
        print(f'[微调开始] 从 Epoch {ep_start} 开始')
        
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
        ft_start_time = time.time()
        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            if hasattr(loader_train, 'sampler') and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(ep)
                if 0 <= ep <= 3:
                    print(f'[训练加载器采样器设置 Epoch {ep}]')
            
            train_loss, train_acc = fine_tune_one_epoch(ep, args, tb_lg, loader_train, iters_train, criterion, mixup_fn, model, model_ema, optimizer, params_req_grad)
            
            # *** NEW *** 更新 args 中的训练损失和准确率
            args.train_loss = train_loss
            args.train_acc = train_acc

            if ep in ep_eval:
                eval_start_time = time.time()
                
                # 获取 evaluate 的完整返回 (只针对当前模型)
                all_targets, all_outputs, tot_pred, last_acc = evaluate(args.device, iterator_val, iters_val, model, num_classes)
                eval_cost = round(time.time() - eval_start_time, 2)
                
                # 计算本轮的 AUPRC, AUROC 和 F-Scores (只针对当前模型)
                current_metrics = calculate_metrics(all_targets, all_outputs, num_classes)

                # 更新最大准确率
                if last_acc > max_acc:
                    max_acc = last_acc
                
                # *** NEW *** 更新最大AUPRC、AUROC和F-Scores，并同时更新最佳每类分数
                # 我们以宏平均AUPRC作为决定“最佳时刻”的主要指标。
                should_update_best_metrics = False
                if current_metrics["macro_auprc"] > max_macro_auprc:
                    max_macro_auprc = current_metrics["macro_auprc"]
                    should_update_best_metrics = True
                
                # 你可以根据实际情况选择：
                # 1. 独立更新每个最佳指标，但这样“最佳模型”可能很难定义。
                # 2. 选择一个主要指标（例如macro_auprc）来定义“最佳模型”，并在这个时刻更新所有“最佳”相关的每类指标。
                # 这里我选择了第二种方式，以 macro_auprc 为主。如果 macro_auroc 达到新高，也更新它的最佳值。
                if current_metrics["macro_auroc"] > max_macro_auroc:
                    max_macro_auroc = current_metrics["macro_auroc"]
                    # should_update_best_metrics = True # 也可以在这里设为True，让AUROC也作为更新所有最佳每类指标的条件

                if current_metrics["macro_f0_5_score"] > max_macro_f0_5:
                    max_macro_f0_5 = current_metrics["macro_f0_5_score"]
                if current_metrics["macro_f1_score"] > max_macro_f1:
                    max_macro_f1 = current_metrics["macro_f1_score"]
                if current_metrics["macro_f2_score"] > max_macro_f2:
                    max_macro_f2 = current_metrics["macro_f2_score"]
                
                if should_update_best_metrics: # 只有当主要指标AUPRC创新高时，才更新最佳每类分数
                    args.best_class_auprcs = current_metrics["class_auprcs"]
                    args.best_class_auroc_scores = current_metrics["class_auroc_scores"]
                    args.best_class_f0_5_scores = current_metrics["class_f0_5_scores"]
                    args.best_class_f1_scores = current_metrics["class_f1_scores"]
                    args.best_class_f2_scores = current_metrics["class_f2_scores"]


                # *** NEW *** 更新 args 中当前轮次的指标和最佳指标
                args.current_macro_auprc = current_metrics["macro_auprc"]
                args.current_macro_auroc = current_metrics["macro_auroc"]
                args.current_macro_f0_5 = current_metrics["macro_f0_5_score"]
                args.current_macro_f1 = current_metrics["macro_f1_score"]
                args.current_macro_f2 = current_metrics["macro_f2_score"]

                args.best_macro_auprc = max_macro_auprc # 确保 args 总是持有当前最佳
                args.best_macro_auroc = max_macro_auroc # 确保 args 总是持有当前最佳
                args.best_macro_f0_5 = max_macro_f0_5
                args.best_macro_f1 = max_macro_f1
                args.best_macro_f2 = max_macro_f2

                args.best_val_acc = max_acc # 更新 args 中的最佳验证准确率
                
                # 构建打印字符串
                # 调整为只打印当前模型的最佳准确率，以及当前 epoch 的准确率
                performance_desc = (
                    f'最佳准确率: {max_acc:.2f}% (当前: {last_acc:.2f}% / {tot_pred})'
                )
                
                metrics_desc = (
                    f'AUPRC (宏平均): {current_metrics["macro_auprc"]:.4f}, '
                    f'AUROC (宏平均): {current_metrics["macro_auroc"]:.4f}, '
                    f'F0.5 (宏平均): {current_metrics["macro_f0_5_score"]:.4f}, '
                    f'F1 (宏平均): {current_metrics["macro_f1_score"]:.4f}, '
                    f'F2 (宏平均): {current_metrics["macro_f2_score"]:.4f}'
                )

                states = model_without_ddp.state_dict(), model_ema.module.state_dict(), optimizer.state_dict()
                
                # 保存最佳模型只基于非EMA模型的准确率
                # 注意：这里 max_acc 已经被更新为当前轮次的准确率，所以条件应该是 if last_acc == max_acc
                # 或者是检查 current_metrics["macro_auprc"] > args.best_macro_auprc 来判断是否基于AUPRC保存
                # 为保持原逻辑，依然基于 max_acc。
                # 如果你想根据 AUPRC 或 AUROC 决定“最佳”模型，你需要修改这里的条件。
                if last_acc == max_acc: # 只有当当前准确率达到新的最高点时才保存"best"模型
                    # 同时将当前最佳指标也加入到性能描述中，以便保存到检查点
                    performance_desc_for_save = (
                        f'最佳准确率: {max_acc:.2f}%, '
                        f'最佳AUPRC: {max_macro_auprc:.4f}, '
                        f'最佳AUROC: {max_macro_auroc:.4f}, '
                        f'最佳F0.5: {max_macro_f0_5:.4f}, '
                        f'最佳F1: {max_macro_f1:.4f}, '
                        f'最佳F2: {max_macro_f2:.4f}, '
                        f'每类AUPRC: {args.best_class_auprcs}, '
                        f'每类AUROC: {args.best_class_auroc_scores}, '
                        f'每类F0.5: {args.best_class_f0_5_scores}, '
                        f'每类F1: {args.best_class_f1_scores}, '
                        f'每类F2: {args.best_class_f2_scores}'
                    )
                    save_checkpoint(f'{args.model}_1kfinetuned_best.pth', args, ep, performance_desc_for_save, *states)
                
                # 始终保存最新的模型，其性能描述应反映当前轮次的准确率和指标
                performance_desc_last = (
                    f'当前准确率: {last_acc:.2f}%, '
                    f'当前AUPRC: {current_metrics["macro_auprc"]:.4f}, '
                    f'当前AUROC: {current_metrics["macro_auroc"]:.4f}, '
                    f'当前F0.5: {current_metrics["macro_f0_5_score"]:.4f}, '
                    f'当前F1: {current_metrics["macro_f1_score"]:.4f}, '
                    f'当前F2: {current_metrics["macro_f2_score"]:.4f}'
                )
                save_checkpoint(f'{args.model}_1kfinetuned_last.pth', args, ep, performance_desc_last, *states)
            else:
                eval_cost = '-'
            
            ep_cost = round(time.time() - ep_start_time, 2) + 1    # +1s: approximate the following logging cost
            remain_secs = (args.ep-1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            s_finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs)) # Changed variable name to avoid conflict with args.finish_time in log_epoch
            
            # 打印当前轮次的结果，只包含非EMA模型的信息
            # 这一行是确保宏平均指标打印出来的关键
            print(f'Epoch {ep+1}/{args.ep}] {performance_desc if ep in ep_eval else "N/A"} {metrics_desc if ep in ep_eval else ""} 本轮耗时: {ep_cost}s, 评估耗时: {eval_cost}, 剩余时间: {remain_time}, 预计完成于: {s_finish_time}')
            
            # --- START OF MODIFICATION ---
            # 确保在评估发生的每个 epoch 都打印详细的各类别指标
            if ep in ep_eval: # Only print detailed per-class metrics when evaluation happens
                # 使用 datetime.datetime.now().strftime('%m-%d %H:%M:%S') 确保时间戳格式与您的日志匹配
                print(f"[{datetime.datetime.now().strftime('%m-%d %H:%M:%S')}] (main.py , line {sys._getframe().f_lineno - 1})=> [Epoch {ep+1}/{args.ep}] 各类别 AUPRC: {current_metrics['class_auprcs']}")
                print(f"[{datetime.datetime.now().strftime('%m-%d %H:%M:%S')}] (main.py , line {sys._getframe().f_lineno - 1})=> [Epoch {ep+1}/{args.ep}] 各类别 AUROC: {current_metrics['class_auroc_scores']}")
                print(f"[{datetime.datetime.now().strftime('%m-%d %H:%M:%S')}] (main.py , line {sys._getframe().f_lineno - 1})=> [Epoch {ep+1}/{args.ep}] 各类别 F0.5-score: {current_metrics['class_f0_5_scores']}")
                print(f"[{datetime.datetime.now().strftime('%m-%d %H:%M:%S')}] (main.py , line {sys._getframe().f_lineno - 1})=> [Epoch {ep+1}/{args.ep}] 各类别 F1-score: {current_metrics['class_f1_scores']}")
                print(f"[{datetime.datetime.now().strftime('%m-%d %H:%M:%S')}] (main.py , line {sys._getframe().f_lineno - 1})=> [Epoch {ep+1}/{args.ep}] 各类别 F2-score: {current_metrics['class_f2_scores']}")
            # --- END OF MODIFICATION ---

            args.cur_ep = f'{ep + 1}/{args.ep}'
            args.remain_time = str(remain_time)
            args.s_finish_time = str(s_finish_time) # Update the string representation of finish time
            # args.train_loss, args.train_acc, args.best_val_acc 已经在前面更新了
            args.log_epoch() # 调用 log_epoch 写入日志文件
            
            if args.is_master:
                tb_lg.add_scalar(f'ft_train/ep_loss', train_loss, ep)
                tb_lg.add_scalar(f'ft_eval/max_acc', max_acc, ep)
                tb_lg.add_scalar(f'ft_eval/last_acc', last_acc, ep)
                # 不再记录 EMA 相关的准确率到 TensorBoard
                tb_lg.add_scalar(f'ft_eval/macro_auprc', current_metrics["macro_auprc"], ep)
                tb_lg.add_scalar(f'ft_eval/macro_auroc', current_metrics["macro_auroc"], ep)
                # *** NEW *** TensorBoard 记录 F-Scores
                tb_lg.add_scalar(f'ft_eval/macro_f0_5_score', current_metrics["macro_f0_5_score"], ep)
                tb_lg.add_scalar(f'ft_eval/macro_f1_score', current_metrics["macro_f1_score"], ep)
                tb_lg.add_scalar(f'ft_eval/macro_f2_score', current_metrics["macro_f2_score"], ep)

                tb_lg.add_scalar(f'ft_z_burnout/rest_hours', round(remain_secs/60/60, 2), ep)
                tb_lg.flush()
        
        # finish fine-tuning
        result_acc = max_acc # 最终结果就是最佳非EMA模型的准确率
        final_metrics = calculate_metrics(all_targets, all_outputs, num_classes) # Final metrics for non-EMA model

        # *** NEW *** 更新 args 中的最终指标
        args.final_macro_auprc = final_metrics["macro_auprc"]
        args.final_macro_auroc = final_metrics["macro_auroc"]
        args.final_macro_f0_5 = final_metrics["macro_f0_5_score"]
        args.final_macro_f1 = final_metrics["macro_f1_score"]
        args.final_macro_f2 = final_metrics["macro_f2_score"]


        if args.is_master:
            tb_lg.add_scalar('ft_result/result_acc', result_acc, ep_start)
            tb_lg.add_scalar('ft_result/result_acc', result_acc, args.ep)
            tb_lg.add_scalar('ft_result/final_macro_auprc', final_metrics["macro_auprc"], args.ep)
            tb_lg.add_scalar('ft_result/final_macro_auroc', final_metrics["macro_auroc"], args.ep)
            # *** NEW *** TensorBoard 记录最终 F-Scores
            tb_lg.add_scalar('ft_result/final_macro_f0_5_score', final_metrics["macro_f0_5_score"], args.ep)
            tb_lg.add_scalar('ft_result/final_macro_f1_score', final_metrics["macro_f1_score"], args.ep)
            tb_lg.add_scalar('ft_result/final_macro_f2_score', final_metrics["macro_f2_score"], args.ep)

            tb_lg.flush()
            tb_lg.close()
        
        # 最终总结打印 (只打印非EMA模型的最终指标)
        # 重新设置 cur_ep 以标记为最终状态，这会触发 log_epoch 写入最终指标
        args.cur_ep = f'{args.ep}/{args.ep}' 
        args.remain_time, args.s_finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
        args.log_epoch() # Log final metrics
        
        print(f'最终参数:\n{str(args)}')
        print('\n\n')
        print(f'[*] [微调完成] 最佳准确率: {result_acc:.2f}%。总耗时: {(time.time() - ft_start_time) / 60 / 60:.1f}小时\n')
        print(f'[*] [微调完成] 最终 AUPRC (宏平均): {final_metrics["macro_auprc"]:.4f}, 最终 AUROC (宏平均): {final_metrics["macro_auroc"]:.4f}')
        print(f'[*] [微调完成] 最终 F0.5-score (宏平均): {final_metrics["macro_f0_5_score"]:.4f}')
        print(f'[*] [微调完成] 最终 F1-score (宏平均): {final_metrics["macro_f1_score"]:.4f}')
        print(f'[*] [微调完成] 最终 F2-score (宏平均): {final_metrics["macro_f2_score"]:.4f}')

        # *** NEW *** 打印最终最佳的每类分数
        print(f'[*] [微调完成] 最佳每类 AUPRC: {args.best_class_auprcs}')
        print(f'[*] [微调完成] 最佳每类 AUROC: {args.best_class_auroc_scores}')
        print(f'[*] [微调完成] 最佳每类 F0.5-score: {args.best_class_f0_5_scores}')
        print(f'[*] [微调完成] 最佳每类 F1-score: {args.best_class_f1_scores}')
        print(f'[*] [微调完成] 最佳每类 F2-score: {args.best_class_f2_scores}')
        print('\n\n')
        time.sleep(10)
    
    # 这一段在 if-else 块外，确保无论是否微调完成都记录最终状态
    args.remain_time, args.s_finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    # 由于上面的 else 块已经调用了 final log_epoch，这里可以根据需要保留或删除
    # args.log_epoch() # 避免重复记录最终状态


def fine_tune_one_epoch(ep, args: FineTuneArgs, tb_lg: SummaryWriter, loader_train, iters_train, criterion, mixup_fn, model, model_ema: ModelEmaV2, optimizer, params_req_grad):
    model.train()
    tot_loss = tot_acc = 0.0
    # Adjusted log_freq to print at 3, halfway, and end of epoch for better visibility
    log_freq_milestones = {3, iters_train // 2, iters_train - 1} 
    ep_start_time = time.time()
    for it, (inp, tar) in enumerate(loader_train):
        # adjust lr and wd
        cur_it = it + ep * iters_train
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, cur_it, args.wp_ep * iters_train, args.ep * iters_train)
        
        # forward
        inp = inp.to(args.device, non_blocking=True)
        raw_tar = tar = tar.to(args.device, non_blocking=True)
        if mixup_fn is not None:
            inp, tar, raw_tar = mixup_fn(inp, tar)
        oup = model(inp)
        pred = oup.data.argmax(dim=1)
        if mixup_fn is None:
            acc = pred.eq(tar).float().mean().item() * 100
            tot_acc += acc
        else:
            acc = (pred.eq(raw_tar) | pred.eq(raw_tar.flip(0))).float().mean().item() * 100
            tot_acc += acc
        
        # backward
        optimizer.zero_grad()
        loss = criterion(oup, tar)
        loss.backward()
        loss = loss.item()
        tot_loss += loss
        if args.clip > 0:
            orig_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
        else:
            orig_norm = None
        optimizer.step()
        model_ema.update(model)
        torch.cuda.synchronize()
        
        # log
        if args.is_master and cur_it % (iters_train // 10 if iters_train >= 10 else 1) == 0: # Log more frequently for TensorBoard
            tb_lg.add_scalar(f'ft_train/it_loss', loss, cur_it)
            tb_lg.add_scalar(f'ft_train/it_acc', acc, cur_it)
            tb_lg.add_scalar(f'ft_hp/min_lr', min_lr, cur_it), tb_lg.add_scalar(f'ft_hp/max_lr', max_lr, cur_it)
            tb_lg.add_scalar(f'ft_hp/min_wd', min_wd, cur_it), tb_lg.add_scalar(f'ft_hp/max_wd', max_wd, cur_it)
            if orig_norm is not None:
                tb_lg.add_scalar(f'ft_hp/orig_norm', orig_norm, cur_it)
        
        # Changed to use log_freq_milestones for specific iteration prints
        if it in log_freq_milestones:
            remain_secs = (iters_train-1 - it) * (time.time() - ep_start_time) / (it + 1)
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            print(f'[Epoch {ep} 迭代 {it:3d}/{iters_train}] 损失: {loss:.4f} 准确率: {acc:.2f} 学习率: {min_lr:.1e}~{max_lr:.1e} 剩余时间: {remain_time}')
        
    return tot_loss / iters_train, tot_acc / iters_train


@torch.no_grad()
def evaluate(dev, iterator_val, iters_val, model, num_classes):
    training = model.training
    model.train(False)
    
    all_targets = []
    all_outputs = [] # Store raw logits or probabilities
    tot_pred, tot_correct = 0., 0.

    for _ in range(iters_val):
        inp, tar = next(iterator_val)
        tot_pred += tar.shape[0]
        inp = inp.to(dev, non_blocking=True)
        tar = tar.to(dev, non_blocking=True)
        oup = model(inp) # OUP will be logits

        all_targets.append(tar.cpu().numpy())
        all_outputs.append(oup.cpu().numpy()) # Store logits for AUPRC/AUROC

        tot_correct += oup.argmax(dim=1).eq(tar).sum().item()
        
    model.train(training)
    
    # gather results from all ranks
    t = torch.tensor([tot_pred, tot_correct]).to(dev)
    tdist.all_reduce(t)
    
    # Also gather all_targets and all_outputs from all ranks for metric calculation
    # This might consume significant memory for very large validation sets.
    # For distributed setup, you'd typically need a custom all_gather for lists/arrays.
    # If it is DDP, you need to use tdist.all_gather_object for lists or concatenate then all_reduce if tensors.
    
    if tdist.is_initialized(): # Check if DDP is initialized
        gathered_targets = [None for _ in range(tdist.get_world_size())]
        gathered_outputs = [None for _ in range(tdist.get_world_size())]
        
        # Need to concatenate first, then gather, as all_gather_object does not concatenate automatically
        current_rank_targets = np.concatenate(all_targets, axis=0)
        current_rank_outputs = np.concatenate(all_outputs, axis=0)

        tdist.all_gather_object(gathered_targets, current_rank_targets)
        tdist.all_gather_object(gathered_outputs, current_rank_outputs)
        
        all_targets_combined = np.concatenate(gathered_targets, axis=0)
        all_outputs_combined = np.concatenate(gathered_outputs, axis=0)
    else:
        all_targets_combined = np.concatenate(all_targets, axis=0)
        all_outputs_combined = np.concatenate(all_outputs, axis=0)

    return all_targets_combined, all_outputs_combined, t[0].item(), (t[1] / t[0]).item() * 100.


def calculate_metrics(targets, outputs, num_classes):
    """
    计算每个类别的 AUPRC, AUROC, F0.5, F1, F2-score，以及它们的宏平均。
    targets: 真实的类别标签 (N,)
    outputs: 模型的预测分数/logits (N, num_classes)
    num_classes: 类别总数
    """
    class_auprcs = {}
    class_auroc_scores = {}
    class_f0_5_scores = {}
    class_f1_scores = {}
    class_f2_scores = {}
    
    targets_np = np.asarray(targets)
    outputs_np = np.asarray(outputs)

    # Convert logits to probabilities for F-scores (using softmax)
    # For AUPRC/AUROC, raw logits often work fine, but probabilities are standard
    probabilities = torch.nn.functional.softmax(torch.from_numpy(outputs_np), dim=1).numpy()
    
    # For F-scores, we need hard predictions (0 or 1)
    # Using argmax for predictions for F-score calculation
    predicted_labels = np.argmax(outputs_np, axis=1)

    macro_auprcs = []
    macro_auroc_scores = []
    macro_f0_5s = []
    macro_f1s = []
    macro_f2s = []

    for i in range(num_classes):
        # Create binary target for current class (one-vs-rest)
        binary_targets = (targets_np == i).astype(int)
        
        # For F-scores, create binary predictions for current class (one-vs-rest)
        binary_preds = (predicted_labels == i).astype(int)

        # Ensure there's at least one positive and one negative sample for valid AUPRC/AUROC calculation
        # For F-scores, precision_recall_fscore_support can handle cases where no true/predicted samples for a class
        if len(np.unique(binary_targets)) < 2:
            # print(f"警告: 类别 {i} 在验证集中样本不足以计算AUPRC/AUROC。") # Debug warning
            class_auprcs[f'class_{i}'] = np.nan
            class_auroc_scores[f'class_{i}'] = np.nan
            # F-scores might still be calculable (e.g., all zeros), but we'll mark as NaN for consistency if no true samples
            class_f0_5_scores[f'class_{i}'] = np.nan
            class_f1_scores[f'class_{i}'] = np.nan
            class_f2_scores[f'class_{i}'] = np.nan
            continue

        try:
            # AUPRC (Average Precision Score)
            ap = average_precision_score(binary_targets, probabilities[:, i])
            class_auprcs[f'class_{i}'] = ap
            macro_auprcs.append(ap)
        except ValueError:
            class_auprcs[f'class_{i}'] = np.nan
            # print(f"警告: 类别 {i} 计算AUPRC时出现ValueError。") # Debug warning
        
        try:
            # AUROC (ROC AUC Score)
            roc_auc = roc_auc_score(binary_targets, probabilities[:, i])
            class_auroc_scores[f'class_{i}'] = roc_auc
            macro_auroc_scores.append(roc_auc)
        except ValueError:
            class_auroc_scores[f'class_{i}'] = np.nan
            # print(f"警告: 类别 {i} 计算AUROC时出现ValueError。") # Debug warning

        # F-scores for current class
        # Zero_division='warn' or '0' or '1' or 'nan'
        # '0': sets f-score to 0.0 when there are no true samples or no predicted samples for a class
        # 'nan': sets f-score to np.nan
        # We use '0' as it's common to treat a class with no true positives as having 0 precision/recall/f-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_targets, binary_preds, average=None, labels=[1], zero_division=0
        )
        
        # precision, recall are arrays, take the first element (for label 1)
        precision = precision[0] if len(precision) > 0 else 0.0
        recall = recall[0] if len(recall) > 0 else 0.0

        # Calculate F0.5, F1, F2 using the formula
        f0_5_score = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall) if (0.5**2 * precision) + recall > 0 else 0.0
        f1_score = (1 + 1**2) * (precision * recall) / ((1**2 * precision) + recall) if (1**2 * precision) + recall > 0 else 0.0
        f2_score = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall) if (2**2 * precision) + recall > 0 else 0.0

        class_f0_5_scores[f'class_{i}'] = f0_5_score
        class_f1_scores[f'class_{i}'] = f1_score
        class_f2_scores[f'class_{i}'] = f2_score
        
        macro_f0_5s.append(f0_5_score)
        macro_f1s.append(f1_score)
        macro_f2s.append(f2_score)

    # Calculate Macro Average, ignore NaN values for AUPRC/AUROC if some classes had no samples
    macro_auprc = np.nanmean(macro_auprcs) if macro_auprcs else np.nan
    macro_auroc = np.nanmean(macro_auroc_scores) if macro_auroc_scores else np.nan
    
    # For F-scores, if a class didn't appear, its score was 0.0, so np.mean is fine.
    macro_f0_5_score = np.mean(macro_f0_5s) if macro_f0_5s else np.nan
    macro_f1_score = np.mean(macro_f1s) if macro_f1s else np.nan
    macro_f2_score = np.mean(macro_f2s) if macro_f2s else np.nan
    
    return {
        "class_auprcs": class_auprcs,
        "class_auroc_scores": class_auroc_scores,
        "class_f0_5_scores": class_f0_5_scores,
        "class_f1_scores": class_f1_scores,
        "class_f2_scores": class_f2_scores,
        "macro_auprc": macro_auprc,
        "macro_auroc": macro_auroc,
        "macro_f0_5_score": macro_f0_5_score,
        "macro_f1_score": macro_f1_score,
        "macro_f2_score": macro_f2_score,
    }


if __name__ == '__main__':
    main_ft()