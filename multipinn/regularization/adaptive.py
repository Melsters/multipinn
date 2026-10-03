import torch
import torch.distributed as dist
from torch import nn

from .basic import BasicLosses


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class AdaptiveWeightSum(nn.Module):
    """
    A module that adaptively learns to weight summed inputs using trainable parameters.

    The weights are initialized with given initial weights and adjusted during training
    via gradient ascent to maximize the weighted sum of inputs. The weights are normalized
    using softmax to ensure they sum to 1.

    Args:
        initial_weights (array-like): Initial values for the weights. These are log-transformed
                                      and used as trainable parameters.
        lr (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-3.

    Attributes:
        weight (torch.Tensor): The current softmax-normalized weights, detached from the computation graph.
        train_weight (torch.Tensor): Log-transformed initial weights used as trainable parameters.
        optimizer (torch.optim.Adam): Optimizer that adjusts train_weight to maximize the weighted sum.
    """

    def __init__(
        self,
        initial_weights,
        lr: float = 1e-3,
        device=None,
        sync_across_ranks: bool = False,
    ):
        super().__init__()
        self.lr = lr
        self.sync_across_ranks = sync_across_ranks
        weight = torch.as_tensor(initial_weights, dtype=torch.float32)
        if device is not None:
            weight = weight.to(device)
        self.weight = weight.detach()
        self.train_weight = nn.Parameter(torch.log(self.weight.clamp(min=1e-12)))
        self.optimizer = torch.optim.Adam(
            params=(self.train_weight,), lr=self.lr, maximize=True, weight_decay=1e-6
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass that computes the weighted sum.

        Args:
            x (torch.Tensor): Input tensor containing individual loss terms to be weighted.

        Returns:
            torch.Tensor: The weighted sum of the input tensor using the current adaptive weights.
        """
        if self.train_weight.device != x.device:
            self.to(x.device)
        return self.__weight_sum(x)

    def __weight_sum(self, x: torch.Tensor):
        """
        Computes the weighted sum of inputs using the current softmax-normalized weights.

        Args:
            x (torch.Tensor): Input tensor to be weighted and summed.

        Returns:
            torch.Tensor: Result of the weighted sum.
        """
        self.weight = torch.softmax(self.train_weight, dim=0)
        return torch.sum(x * self.weight)

    def step(self, scaler=None):
        if self.train_weight.grad is None:
            return
        if self.sync_across_ranks:
            self._all_reduce_grad()
        if scaler is None:
            self.optimizer.step()
        else:
            scaler.step(self.optimizer)
        self.optimizer.zero_grad()
        if self.sync_across_ranks:
            self._broadcast_train_weight()
        with torch.no_grad():
            self.weight = torch.softmax(self.train_weight, dim=0)

    def _all_reduce_grad(self) -> None:
        if not _ddp_active() or self.train_weight.grad is None:
            return
        dist.all_reduce(self.train_weight.grad, op=dist.ReduceOp.SUM)
        self.train_weight.grad /= dist.get_world_size()

    def _broadcast_train_weight(self) -> None:
        if not _ddp_active():
            return
        with torch.no_grad():
            dist.broadcast(self.train_weight, src=0)


class AdaptiveConditionsLosses(BasicLosses):
    """
    Implements adaptive loss weighting for multiple loss terms in PINN training.

    This class automatically adjusts the weights of different loss components during training
    using a gradient-based optimization approach. The weights are learned by maximizing
    the weighted sum of losses, encouraging balanced training across different conditions.

    Args:
        lr (float, optional): Learning rate for the weight optimizer. Defaults to 1e-3.

    Attributes:
        lr (float): Learning rate for the weight optimizer.
        adaptive_cond (AdaptiveWeightSum): Class that store train_weight and use them.
    """

    def __init__(
        self,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.lr = lr
        self.adaptive_cond = None

    def __call__(self, trainer: "Trainer"):
        """
        Computes the total adaptive loss for the current training state.

        Args:
            trainer (Trainer): The PINN trainer instance.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Total loss and detached individual mean losses.
        """
        if self.adaptive_cond is None:
            return self.__first_launch(trainer)
        return self.__calc_losses(trainer)

    def __calc_losses(self, trainer: "Trainer"):
        """
        Computes the mean losses and applies adaptive weighting.

        Args:
            trainer (Trainer): The PINN trainer instance.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Total loss and detached individual losses.
        """
        mean_losses = trainer.pinn.calculate_loss()
        return self.weight_sum(mean_losses)

    def __first_launch(self, trainer: "Trainer"):
        """
        Initializes adaptive weights during the first call.

        Args:
            trainer (Trainer): The PINN trainer instance.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Total loss and detached individual losses.
        """
        mean_losses, counts = trainer.pinn.calculate_loss_and_count()
        device = mean_losses[0].device
        self.adaptive_cond = AdaptiveWeightSum(
            counts, self.lr, device=device, sync_across_ranks=True
        )
        return self.weight_sum(mean_losses)

    def weight_sum(self, mean_losses):
        """
        Applies adaptive weighting to the mean losses and returns the total loss.

        Args:
            mean_losses (list[torch.Tensor]): List of mean loss values for each condition.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Total loss and detached individual losses.
        """

        total_loss = self.adaptive_cond(torch.stack(mean_losses))
        return total_loss, torch.stack(mean_losses).detach()

    def step(self, scaler=None):
        if self.adaptive_cond is not None:
            self.adaptive_cond.step(scaler)


class AdaptiveConditionsAndPointsLosses(AdaptiveConditionsLosses):
    """
    Extends adaptive loss weighting to handle both condition and point-wise losses.

    This class manages adaptive weights for both different loss conditions and individual
    points within those conditions. Weights are updated periodically based on the training epoch.

    Args:
        lr (float, optional): Learning rate for both condition and point weight optimizers. Defaults to 1e-3.

    Attributes:
        adaptive_points (tuple[AdaptiveWeightSum]): Tuple of AdaptiveWeightSum instances for point-wise weighting.
    """

    def __init__(
        self,
        lr: float = 1e-3,
    ):
        super().__init__(lr)
        self.adaptive_points = tuple()
        self._adaptive_points_by_batch = {}
        self._points_epoch = None

    def __call__(self, trainer: "Trainer"):
        losses = []
        for cond in trainer.pinn.conditions:
            losses += trainer.pinn.condition_loss(cond)
        if (
            trainer.current_epoch % trainer.update_grid_every == 0
            and self._points_epoch != trainer.current_epoch
        ):
            self._adaptive_points_by_batch = {}
            self._points_epoch = trainer.current_epoch
        batch_index = getattr(trainer, "rank", trainer.current_batch)
        if batch_index not in self._adaptive_points_by_batch:
            self._adaptive_points_by_batch[batch_index] = tuple(
                AdaptiveWeightSum(torch.ones_like(loss), self.lr, device=loss.device)
                for loss in losses
            )
        self.adaptive_points = self._adaptive_points_by_batch[batch_index]
        if self.adaptive_cond is None:
            counts = [len(x) for x in losses]
            device = losses[0].device
            self.adaptive_cond = AdaptiveWeightSum(
                counts, self.lr, device=device, sync_across_ranks=True
            )
        return self.__weight_points(losses)

    def __weight_points(self, losses):
        """
        Applies point-wise adaptive weighting to each loss component before condition weighting.

        Args:
            losses (list[torch.Tensor]): List of loss tensors for each condition.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Total loss and detached individual losses.
        """
        mean_losses = [self.adaptive_points[i](loss) for i, loss in enumerate(losses)]
        return self.weight_sum(mean_losses)

    def step(self, scaler=None):
        for adaptive_points in self._adaptive_points_by_batch.values():
            for adaptive_point in adaptive_points:
                adaptive_point.step(scaler)
        super().step(scaler)
