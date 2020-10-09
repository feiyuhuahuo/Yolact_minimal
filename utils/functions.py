class ProgressBar:
    def __init__(self, length, max_val):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0

        self.cur_num_bars = -1
        self.update_str()

    def update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = '█' * num_bars + '░' * (self.length - num_bars)

    def get_bar(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        self.update_str()
        return self.string