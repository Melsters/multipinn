import torch
import torch.distributed as dist

from .basic import BasicLosses


class NormalLosses(BasicLosses):
    def __init__(self, alpha: float = 0.9):
        super().__init__()
        self.lambda_regularization = None
        self.alpha = alpha

    def __call__(self, trainer):
        losses = trainer.pinn.calculate_loss()
        if len(trainer.pinn.conditions) != len(losses):
            raise Exception("Regularization does not supported system ODE")
        weighted_losses = [losses[0]]
        for ind, loss in enumerate(losses[1:]):
            weighted_losses.append(loss * self.lambda_regularization[ind])
        return (
            torch.sum(torch.stack(weighted_losses)),
            torch.stack(losses).detach(),
        )

    def prepare_for_batches(self, trainer):
        last_layer = list(list(trainer.pinn.model.children())[-1].children())[-1]
        weight = last_layer.weight
        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
            and hasattr(trainer, "rank")
        )
        batch_indices = (
            (trainer.rank,) if distributed else range(trainer.num_batches)
        )
        divisor = dist.get_world_size() if distributed else trainer.num_batches
        gradient_sums = None

        for batch_index in batch_indices:
            trainer.pinn.select_batch(batch_index)
            losses = trainer.pinn.calculate_loss()
            if len(trainer.pinn.conditions) != len(losses):
                raise Exception("Regularization does not supported system ODE")
            if gradient_sums is None:
                gradient_sums = [torch.zeros_like(weight) for _ in losses]
            for loss_index, loss in enumerate(losses):
                gradient = torch.autograd.grad(
                    loss, weight, retain_graph=True
                )[0]
                gradient_sums[loss_index].add_(gradient.detach())

        if distributed:
            for gradient_sum in gradient_sums:
                dist.all_reduce(gradient_sum, op=dist.ReduceOp.SUM)

        mean_gradients = [gradient_sum / divisor for gradient_sum in gradient_sums]
        var_f = torch.std(mean_gradients[0])
        proposed = torch.stack(
            [var_f / torch.std(gradient) for gradient in mean_gradients[1:]]
        )
        if self.lambda_regularization is None:
            self.lambda_regularization = torch.ones_like(proposed)
        self.lambda_regularization = (
            (1 - self.alpha) * self.lambda_regularization
            + self.alpha * proposed
        ).detach()
