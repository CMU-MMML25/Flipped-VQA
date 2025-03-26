import torch
import math
import sys
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
import time
import json
def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 4)
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        vqa_loss, vaq_loss, qav_loss = model(data)

        loss = vqa_loss + vaq_loss + qav_loss
        loss_value = loss.item()
        vqa_loss_value = vqa_loss.item()
        vaq_loss_value = vaq_loss.item()
        qav_loss_value = qav_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(vqa_loss=vqa_loss_value)
        metric_logger.update(vaq_loss=vaq_loss_value)
        metric_logger.update(qav_loss=qav_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 4)

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        answer = data['answer'].cuda()
        bsz = answer.shape[0]

        with torch.no_grad():
            logits = model(data, inference=True)
        
        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)

        eval = (answer == prediction)
        acc = eval.sum().item() / bsz
        
        misc.log_qtype(data, eval, metric_logger, args)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(n=bsz, acc=acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def val_one_epoch_output_json(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None, max_samples=500, output_json_path=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 4)
    
    results = []  # Store results for JSON output
    samples_processed = 0
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch_start_time = time.time()
        
        # Get video_ids
        video_ids = data['vid']
        question_ids = data['question_id']
        
        # # Get frame indices - assuming video_len gives us usable frames
        # frame_ids_list = []
        # for i, length in enumerate(data['video_len']):
        #     frame_ids_list.append(list(range(length.item())))
        
        answer = data['answer'].cuda()
        bsz = answer.shape[0]
        
        with torch.no_grad():
            inference_start = time.time()
            logits = model(data, inference=True)
            inference_time = time.time() - inference_start
            
            count = (logits != 0).sum(-1)
            avg_logits = logits.sum(-1) / count
            prediction = avg_logits.argmin(-1)
            
            # Calculate confidence scores (using negative since lower is better in this model)
            neg_avg_logits = -avg_logits
            confidence_scores = torch.nn.functional.softmax(neg_avg_logits, dim=-1)
        
        # Total processing time
        total_time = time.time() - batch_start_time
        
        # Store prediction results
        for i in range(bsz):
            # Create a unique question_id
            question_text = data['text'][i]['q_text'] if 'text' in data else f"question_{len(results)}"
            # question_id = f"{video_ids[i]}_{hash(question_text) % 10000}"
            
            result = {
                # "question_id": question_id,
                "pred_ans_idx": prediction[i].item(),
                "correct_ans_idx": answer[i].item(),
                "confidence": confidence_scores[i].tolist(),
                # "frame_ids": frame_ids_list[i],
                "inference_time": inference_time,
                "question_id": question_ids[i],
                "video_id": video_ids[i]
            }
            results.append(result)
        samples_processed += bsz
        # Original evaluation code
        eval = (answer == prediction)
        acc = eval.sum().item() / bsz
        misc.log_qtype(data, eval, metric_logger, args)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(n=bsz, acc=acc)
        # Check if we've reached max_samples
        if max_samples is not None and samples_processed >= max_samples:
            print(f"Reached maximum samples ({max_samples}). Stopping evaluation.")
            break
    
    # Save results to JSON if path is provided
    if output_json_path and misc.is_main_process():
        print(f"Saving {len(results)} predictions to {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
