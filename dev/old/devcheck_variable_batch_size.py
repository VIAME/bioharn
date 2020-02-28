import ubelt as ub
import torch
import kwarray
import torch.utils.data.sampler as torch_sampler
import torch.utils.data as torch_data


class MyDataset(torch_data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 211

    def __getitem__(self, spec):
        if isinstance(spec, int):
            index = spec
            input_size = [256, 256]
        elif isinstance(spec, dict):
            index = spec['index']
            input_size = spec['input_size']
        else:
            raise TypeError(spec)
        assert isinstance(index, int)

        item = {
            'index': torch.LongTensor([index]),
            'input_size': torch.FloatTensor([input_size]),
        }
        return item


class MyBatchSampler(torch_sampler.BatchSampler, ub.NiceRepr):

    def __nice__(self):
        return str(self.num_batches)

    def __init__(self, sampler):
        self.sampler = sampler
        self.drop_last = False

        self.batch_dynamics = [
            {'batch_size': 2, 'input_size': [1024, 1024]},
            {'batch_size': 4, 'input_size': [512, 512]},
            {'batch_size': 4, 'input_size': [256, 256]},
            {'batch_size': 6, 'input_size': [128, 128]},
        ]
        self.num_batches = None
        self.resample_interval = 4
        self._dynamic_schedule = ub.odict()
        self.rng = kwarray.ensure_rng(None)
        self._build_dynamic_schedule()

    def _build_dynamic_schedule(self):
        self._dynamic_schedule = ub.odict()
        total = len(self.sampler)
        remain = total

        # Always end on the native dynamic
        native_dynamic = self.batch_dynamics[0]
        final_native = self.resample_interval * 2

        num_final = final_native * native_dynamic['batch_size']

        bx = 0
        while remain > 0:
            if remain <= num_final:
                current = native_dynamic.copy()
                current['remain'] = remain
                self._dynamic_schedule[bx] = current
            elif bx % self.resample_interval == 0:
                dyn_idx = self.rng.randint(len(self.batch_dynamics))
                current = self.batch_dynamics[dyn_idx]
                current = current.copy()
                if remain < 0:
                    current['batch_size'] += remain
                current['remain'] = remain
                self._dynamic_schedule[bx] = current

                if remain < num_final:
                    # Ensure there are enough items for final batches
                    current['remain'] = remain
                    current['batch_size'] -= (num_final - remain)
                    self._dynamic_schedule[bx] = current

            if remain <= current['batch_size']:
                current['batch_size'] = remain
                current['remain'] = remain
                current = current.copy()
                self._dynamic_schedule[bx] = current

            bx += 1
            remain = remain - current['batch_size']

        final_bx, final_dynamic = list(self._dynamic_schedule.items())[-1]

        last = final_dynamic['remain'] // final_dynamic['batch_size']
        num_batches = final_bx + last

        self.num_batches = num_batches

        # schedule_items = list(self._dynamic_schedule)
        # for a, b in ub.iter_window(schedule_items, 2):
        #     num = b[0] - a[0]
        #     # a[1]['batch_size']
        # print('self._dynamic_schedule = {}'.format(ub.repr2(self._dynamic_schedule, nl=1)))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Start first batch
        bx = 0
        batch = []
        if bx in self._dynamic_schedule:
            current_dynamic = self._dynamic_schedule[bx]

        for idx in self.sampler:
            # Specify dynamic information to the dataset
            index = {
                'index': idx,
                'input_size': current_dynamic['input_size'],
            }
            batch.append(index)
            if len(batch) == current_dynamic['batch_size']:
                yield batch

                # Start next batch
                bx += 1
                batch = []
                if bx in self._dynamic_schedule:
                    current_dynamic = self._dynamic_schedule[bx]

        if len(batch) > 0 and not self.drop_last:
            yield batch


def main():

    dataset = MyDataset()

    shuffle = False
    if shuffle:
        sampler = torch_sampler.RandomSampler(dataset)
    else:
        sampler = torch_sampler.SequentialSampler(dataset)

    batch_sampler = MyBatchSampler(sampler)

    print(len(batch_sampler))

    # torch.utils.data.sampler.WeightedRandomSampler

    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler,
        num_workers=0, pin_memory=True)

    for batch in ub.ProgIter(loader, total=len(loader), verbose=3):
        print('batch = {!r}'.format(batch))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/dev/devcheck_variable_batch_size.py
    """
    main()
