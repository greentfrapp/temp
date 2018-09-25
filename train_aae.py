from modules.generator import GeneratorToy1
from modules.generator import GeneratorToy2
from modules.generator import GeneratorToy3
from modules.generator import GeneratorToy4
from modules.generator import GeneratorToy5
from modules.generator import GeneratorToy6
from modules.generator import GeneratorToy7
from modules.generator import GeneratorToy8
from modules.aae_labeled import LabeledAdversarialAutoencoder
from modules.aae_util import BasicAdversarialAutoencoder
import numpy as np


def main():
	generator = GeneratorToy4()
	dataset = generator.generate()
	aae = LabeledAdversarialAutoencoder(data_dim=3, z_dim=2)
	aae.train(data=dataset, n_epochs=2000)

if __name__ == "__main__":
	main()
