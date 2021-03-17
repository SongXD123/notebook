考虑的几个方面：

* 最明显的是，学习率的大小很重要。如果它太大，优化就会发散，如果它太小，需要太长时间来训练，或者我们最终得到一个次优的结果。我们之前看到问题的条件数很重要（病态：一些方向上的进展比其他方向慢得多，就像一个狭窄的峡谷）。直觉上条件数是**最不敏感方向的变化量与最敏感方向的变化量之比**。
* 其次，衰减率同样重要。如果学习率仍然很高，最终会在最小值附近跳动，不会到达最优点。总之，我们希望学习率逐渐衰减，对于凸问题来说，衰减速率$O(t_{\frac{1}{2}})$是最好的。
* 另一个重要方面是初始化。这既涉及参数最初是如何设置的，也涉及它们最初是如何更新的。这被称为warm-up，也就是说，最开始向解决方案移动的速度有多快。**开始时大的步伐不一定有益**，特别是参数是随机初始化的。**最初的更新方向也可能毫无意义**
* 最后，有一些优化变量遵循周期性学习率调整。

### Schedulers

```python
if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

```



* 一种调整学习率的方法是在每一步显式调整

  ```python
  lr = 0.1
  trainer.param_groups[0]["lr"] = lr
  print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
  
  #如
  # Using custom defined scheduler
  for param_group in trainer.param_groups:
      param_group['lr'] = scheduler(epoch)
  
  ```

  

* 或者定义scheduler，如将学习率设为$\eta=\eta_0(t+1)^{-\frac{1}{2}}$

  ```python
  class SquareRootScheduler:
      def __init__(self, lr=0.1):
          self.lr = lr
  
      def __call__(self, num_update):
          return self.lr * pow(num_update + 1.0, -0.5)
  ```

  

* 通过调整学习率曲线更加平滑，减少过拟合。然而，为何一些策略会减少过拟合仍然是未解决的问题。有人认为，步长越小，参数越接近零，因此越简单。然而，这并不能完全解释这一现象，因为我们并没有真的提前停止，而是只是稍微地降低学习速度。

### Policies

几种调整策略：常见的选择是多项式衰减和分段常数调度。除此之外，余弦学习率表已经被发现在一些问题上有很好的经验效果。最后，在一些问题上，在使用大的学习率之前对优化器进行预热是有益的。

* Factor Schedular：多项式衰减的一种替代方法乘法衰减：$\eta_{t+1}\leftarrow\eta_{t}\alpha\quad for\quad \alpha\in(0,1)$。为防止学习率衰减超过合理下限，通常修改为$\eta_{t+1}\leftarrow max(\eta_{min},\eta_t\alpha)$

  ```python
  class FactorScheduler:
      def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
          self.factor = factor
          self.stop_factor_lr = stop_factor_lr
          self.base_lr = base_lr
  
      def __call__(self, num_update):
          self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
          return self.base_lr
  #或内置torch.optim.lr_schedular没有FactorSchedular
  
  ```

* Multi Factor Schedular：保持学习率分段不变，每隔一定时间降低一定量。给定一组降低速率的时间，如$s=\{5,10,20\}$，通过$\eta_{t+1}\leftarrow\eta_{t}\alpha$，当$t\in{s}$时。

  ```python
  net = net_fn()
  trainer = torch.optim.SGD(net.parameters(), lr=0.5)
  scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)#每次衰减一半
  
  def get_lr(trainer, scheduler):
      lr = scheduler.get_last_lr()[0]
      trainer.step()
      scheduler.step()
      return lr
  
  d2l.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler)
                                    for t in range(num_epochs)])
  ```

  * 分段尝试的思想在于使得优化继续，直到根据权重向量分布达到一个稳定点。每当进度停滞时，分段降低学习率时有效的

* Cosine Scheduler：如果不希望在开始时大幅降低学习率，而且希望最终通过非常小的学习率来改进解，于是就有
  $$
  \eta_t=\eta_T+\frac{\eta_0-\eta_T}{2}(1+cos(\pi t/T)),t\in[0,T]
  $$
  其中，$\eta_0$是初始学习率，$\eta_{T}$是$T$时目标学习率。另外，对于$t>T$，可以将值固定在$\eta_T$

  ```python
  class CosineScheduler:
      def __init__(self, max_update, base_lr=0.01, final_lr=0,
                 warmup_steps=0, warmup_begin_lr=0):
          self.base_lr_orig = base_lr
          self.max_update = max_update
          self.final_lr = final_lr
          self.warmup_steps = warmup_steps
          self.warmup_begin_lr = warmup_begin_lr
          self.max_steps = self.max_update - self.warmup_steps
  
      def get_warmup_lr(self, epoch):
          increase = (self.base_lr_orig - self.warmup_begin_lr) \
                         * float(epoch) / float(self.warmup_steps)
          return self.warmup_begin_lr + increase
  
      def __call__(self, epoch):
          if epoch < self.warmup_steps:
              return self.get_warmup_lr(epoch)
          if epoch <= self.max_update:
              self.base_lr = self.final_lr + (
                  self.base_lr_orig - self.final_lr) * (1 + math.cos(
                  math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
          return self.base_lr
  
  scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
  d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
  #或
  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
  # 以余弦函数为周期，并在每个周期最大值时重新设置学习率。以初始学习率为最大学习率，以 2∗Tmax2*Tmax2∗Tmax 为周期，在一个周期内先下降，后上升。
  # T_max：余弦函数半周期
  ```

  在计算机视觉中，该方法可以提高结果

* Warm-up：

  在某些情况下，初始化参数不足以保证好的解决方案。这对于一些可能导致不稳定优化问题的高级网络设计来说尤其是一个问题。我们可以通过选择足够小的学习率来解决这个问题，以防止一开始就出现发散。不幸的是，这意味着进展缓慢。相反，大的学习率最初会导致发散。

  一个简单的解决方法时是使用**warm-up**，使得学习率增加到初始最大值，然后再减小直到优化过程结束

  warm-up可以应用于任何schedular

  ```python
  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
  ```

  

* 

