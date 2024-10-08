from dataclasses import dataclass
@dataclass
class CFG():
    epochs: int = 3

@dataclass
class TrainConfig(CFG):
    batch_size: int = 10

@dataclass
class TestConfig(CFG):
    batch_size: int = 10

trainarg = TrainConfig()
testarg = TestConfig()