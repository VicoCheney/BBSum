import os
import re

import torch
from torch.optim.lr_scheduler import _LRScheduler
from buffer import Buffer


def find_lastest_checkpoint(checkpoints_dir, epoch=False):
    lastest = (-1, '')
    if os.path.exists(checkpoints_dir):
        for shortname in os.listdir(checkpoints_dir):
            m = re.match(r'epoch=(\d+).+', shortname)
            if m is not None and int(m.group(1)) > lastest[0]:
                lastest = (int(m.group(1)), shortname)
    return os.path.join(checkpoints_dir, lastest[-1]) if not epoch else lastest[0]


class WarmupLinearLR(_LRScheduler):
    def __init__(self, optimizer, step_size, peak_percentage=0.1, min_lr=1e-5, last_epoch=-1):
        self.step_size = step_size
        self.peak_step = peak_percentage * step_size
        self.min_lr = min_lr
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ret = []
        for base_lr in self.base_lrs:
            if self._step_count <= self.peak_step:
                ret.append(self.min_lr + (base_lr - self.min_lr) * self._step_count / self.peak_step)
            else:
                ret.append(self.min_lr + max(0, (base_lr - self.min_lr) * (self.step_size - self._step_count) / (
                        self.step_size - self.peak_step)))
        return ret


def compress(compresser, data_buf, times, device, batch_size_inference):
    times = [int(x) for x in times.split(',')]
    inputs = torch.zeros(3, batch_size_inference, 512, dtype=torch.long, device=device)
    B_set = []  # the poses of B blks in qbuf
    compressed_buf = Buffer(data_buf.summary)
    for k, inc in enumerate(times):
        num_to_keep = len(compressed_buf) + inc
        estimations = torch.zeros(len(data_buf), device='cpu')
        bufs, t = compressed_buf.fill(data_buf), 0
        for i in range((len(bufs) - 1) // batch_size_inference + 1):
            l, r = batch_size_inference * i, min(len(bufs), batch_size_inference * (i + 1))
            for j, buf in enumerate(bufs[l:r]):
                buf.export(out=(inputs[0, j], inputs[1, j]), device=device)
            logits = compresser(*inputs[:, :r - l]).sigmoid_()
            for j, buf in enumerate(bufs[l:r]):
                estimation = blk_scorer(buf, logits[j])[len(compressed_buf):]
                estimations[t: t + len(estimation)] = estimation
                t += len(estimation)
        assert t == len(data_buf)

        indices = estimations.argsort(descending=True)

        compressed_size = compressed_buf.calc_size()
        for idx in indices:
            if compressed_size + len(data_buf[idx]) > 512:
                break
            if data_buf[idx] in B_set:
                continue
            compressed_size += len(data_buf[idx])
            compressed_buf.insert(data_buf[idx])

        relevance_token = torch.sigmoid(compresser(*inputs[:, :1]).view(-1))
        relevance_blk = blk_scorer(compressed_buf, relevance_token)
        keeped_indices = relevance_blk.argsort(descending=True)
        if len(keeped_indices) > num_to_keep and k < len(times) - 1:
            keeped_indices = keeped_indices[:num_to_keep]
        else:
            return compressed_buf
        filtered_buf = Buffer(data_buf.summary)
        for i, blk in enumerate(compressed_buf):
            if i in keeped_indices:
                filtered_buf.blocks.append(blk)
        compressed_buf = filtered_buf
        B_set = [blk for blk in compressed_buf]

    estimations = torch.zeros(len(data_buf), device='cpu')
    bufs, t = compressed_buf.fill(data_buf), 0
    for i in range((len(bufs) - 1) // batch_size_inference + 1):
        l, r = batch_size_inference * i, min(len(bufs), batch_size_inference * (i + 1))
        for j, buf in enumerate(bufs[l:r]):
            buf.export(out=(inputs[0, j], inputs[1, j]), device=device)
        logits = compresser(*inputs[:, :r - l]).sigmoid_()
        for j, buf in enumerate(bufs[l:r]):
            estimation = blk_scorer(buf, logits[j])[len(compressed_buf):]
            estimations[t: t + len(estimation)] = estimation
            t += len(estimation)
    assert t == len(data_buf)
    indices = estimations.argsort(descending=True)
    compressed_size = compressed_buf.calc_size()
    for idx in indices:
        if compressed_size + len(data_buf[idx]) > 1024:
            break
        if data_buf[idx] in B_set:
            continue
        compressed_size += len(data_buf[idx])
        compressed_buf.insert(data_buf[idx])
    return compressed_buf

def blk_scorer(buf, relevance_token):
    ends = buf.block_ends()
    relevance_blk = torch.zeros(len(ends), dtype=torch.float, device='cpu')
    for i in range(len(ends)):
        if i == 0:
            relevance_blk[i] = (relevance_token[0:ends[i]]).mean()
        else:
            relevance_blk[i] = (relevance_token[ends[i-1]:ends[i]]).mean()
    return relevance_blk
