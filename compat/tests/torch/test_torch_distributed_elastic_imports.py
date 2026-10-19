"""Import surface used when DeepSpeed is launched by torchrun."""

import pytest
import torch
import torch.distributed as distributed


def test_torchrun_elastic_imports_are_explicitly_non_executable():
    from torch.distributed.elastic.agent.server.local_elastic_agent import (
        LocalElasticAgent,
    )
    from torch.distributed.elastic.agent.server.api import WorkerSpec

    assert distributed.elastic.agent.server.api.WorkerSpec is WorkerSpec
    assert isinstance(LocalElasticAgent, type)
    with pytest.raises(NotImplementedError, match="launch distributed workers with torchrun"):
        LocalElasticAgent(WorkerSpec())
