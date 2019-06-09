import logging

from torch import nn, optim

from models.CharacterRNN import CharacterRNN
from WordTrainer import WordTrainer

logging.basicConfig(datefmt='%d-%b %H:%M:%S',
                    format='%(asctime)s - [%(levelname)s]: %(message)s',
                    level=logging.INFO)


if __name__ == '__main__':
    trainer = WordTrainer(train_file=r'C:\Python\projects\PyLT3\pylt3\ml\data\char-test\partitions\names.train.txt',
                          valid_file=r'C:\Python\projects\PyLT3\pylt3\ml\data\char-test\partitions\names.dev.txt',
                          test_file=r'C:\Python\projects\PyLT3\pylt3\ml\data\char-test\partitions\names.test.txt')
    trainer.model = CharacterRNN(trainer.n_letters, 256, trainer.n_categories, dropout=0.2)
    trainer.criterion = nn.CrossEntropyLoss()
    trainer.optimizer = optim.Adam([p for p in trainer.model.parameters() if p.requires_grad], lr=0.0002)

    # trainer.train(epochs=150, patience=20)
    # trainer.test()
    trainer.predict()
