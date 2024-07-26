class FactorScheduler:
    def __init__(self, optimizer, factor=1, stop_factor_lr=1e-6, base_lr=0.1, total_iterations=250, max_segments=3, warmup_steps=50, max_epochs=250):
        self.optimizer = optimizer
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr
        self.mem_lr = self.base_lr
        self.max_lr = self.base_lr
        self.new_segment = False
        self.segments_count = 0
        self.iteration = 0
        self.total_iterations = total_iterations
        self.max_segments = max_segments
        self.current_iteration = 0
        self.warmup_steps = warmup_steps
        self.warmup_counter = 0
        self.max_epochs = max_epochs
        self.flag = False

    def step(self):
        if self.warmup_counter < self.warmup_steps:
            if self.flag == False:
                self.base_lr = 0
                self.flag = True
            else:
                self.base_lr = (self.warmup_counter+1) / (self.warmup_steps) * self.mem_lr

            self.warmup_counter += 1
            self.current_iteration = 0
        else:
            self.flag = False
            self.decay_per_iteration = self.base_lr * (self.factor - self.stop_factor_lr) * self.max_segments / (self.total_iterations)# * self.segments_count
            self.base_lr = self.base_lr - self.decay_per_iteration
            self.mem_lr = (1 - self.segments_count / self.max_segments) * self.max_lr
            
            self.current_iteration += 1

        self.optimizer.param_groups[0]['lr'] = self.base_lr