from pathlib import Path

data_paths = {name: Path(__file__).parent.parent.joinpath('data', name) for name in (
    'hawks',
    'noisy_xor',
    'penguins',
    'pima'
)}
