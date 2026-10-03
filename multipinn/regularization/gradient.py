import torch
import torch.distributed as dist

from .basic import BasicLosses


class GradientLosses(BasicLosses):
    """
    Class with regularization UNDERSTANDING AND MITIGATING GRADIENT FLOW PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS
    github authors https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/tree/master

    Args:
        alpha (float): hyperparametr, authors recommended value = 0.9.
    """

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
        params = [p for p in trainer.pinn.model.parameters() if p.requires_grad]
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
        used_masks = None

        for batch_index in batch_indices:
            trainer.pinn.select_batch(batch_index)
            losses = trainer.pinn.calculate_loss()
            if len(trainer.pinn.conditions) != len(losses):
                raise Exception("Regularization does not supported system ODE")
            if gradient_sums is None:
                flat_size = sum(parameter.numel() for parameter in params)
                gradient_sums = [
                    torch.zeros(
                        flat_size,
                        device=losses[0].device,
                        dtype=losses[0].dtype,
                    )
                    for _ in losses
                ]
                used_masks = [
                    torch.zeros(
                        len(params), device=losses[0].device, dtype=torch.int32
                    )
                    for _ in losses
                ]
            for loss_index, loss in enumerate(losses):
                gradients = torch.autograd.grad(
                    loss, params, retain_graph=True, allow_unused=True
                )
                flat_gradient = torch.cat(
                    [
                        (
                            gradient
                            if gradient is not None
                            else torch.zeros_like(parameter)
                        ).reshape(-1)
                        for gradient, parameter in zip(gradients, params)
                    ]
                )
                gradient_sums[loss_index].add_(flat_gradient.detach())
                for parameter_index, gradient in enumerate(gradients):
                    if gradient is not None:
                        used_masks[loss_index][parameter_index] = 1

        if distributed:
            for gradient_sum, used_mask in zip(gradient_sums, used_masks):
                dist.all_reduce(gradient_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(used_mask, op=dist.ReduceOp.MAX)

        mean_gradients = [gradient_sum / divisor for gradient_sum in gradient_sums]
        max_f = mean_gradients[0].abs().max()
        offsets = []
        offset = 0
        for parameter in params:
            offsets.append((offset, offset + parameter.numel()))
            offset += parameter.numel()

        proposed = []
        for gradient, used_mask in zip(mean_gradients[1:], used_masks[1:]):
            parameter_means = [
                gradient[start:end].abs().mean()
                for parameter_index, (start, end) in enumerate(offsets)
                if used_mask[parameter_index]
            ]
            proposed.append(max_f / torch.stack(parameter_means).mean())

        proposed = torch.stack(proposed)
        if self.lambda_regularization is None:
            self.lambda_regularization = torch.ones_like(proposed)
        self.lambda_regularization = (
            (1 - self.alpha) * self.lambda_regularization
            + self.alpha * proposed
        ).detach()
