import torch

from model import HarryPotterTransformer
from dataset import Vocabulary


def generate(model, device, seed_words, input_length, output_length, vocab, temperature=1.0, strategy='sampling'):
    """
    Arguments:
        model: HarryPotterTransformer
        device: string, presents tensor's device
        seed_words: string
        input_length: int, model's max input length
        output_length: int, max output length during generation
        vocab: Vocabulary, defined in dataset.py
        temperature: float, temperature during model inference
        strategy: string, should be either 'sampling' based on model output probabilities 
                  or 'greedy' by choosing the maximum probability. 
    """
    model.eval()

    with torch.no_grad():

        output_arr = torch.LongTensor([-1]).to(device)

        ################################################################################
        # TODO: generate a paragraph from seed_words                                   #
        # To complete this task, you need to review the input and output formats       #
        # of various functions and models in other code file, and integrate them to    #
        # achieve our goal.                                                            #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        output_arr = vocab.words_to_array(seed_words).to(device)
        length=len(output_arr)
        while 1:
            p = model.inference(output_arr,temperature)
            #print(p)
            if strategy=='sampling':
                p /= p.sum()
                new_word = torch.multinomial(p, num_samples=1).item()
                new_word = torch.tensor([new_word]).to(device)
            elif strategy=="greedy":
                new_word = p.argmax()
                new_word = torch.tensor([new_word]).to(device)
            #print(new_word)
            #print(output_arr)
            output_arr=torch.cat((output_arr,new_word),dim=0)
            length=length+1
            if new_word == '#' or length>=output_length:
                break

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################

        return vocab.array_to_words(output_arr[:-1]), len(output_arr)-1

# hyper-parameters, shouble be same as in main.py
SEQUENCE_LENGTH = 100
FEATURE_SIZE = 512
NUM_HEADS = 8
USE_CUDA = True

MODEL_TYPE = 'transformer'
DATA_PATH = '/home/sensestudy/work/current/data/'
EXP_PATH = f'/home/sensestudy/work/current/exp/'

if __name__=="__main__":
    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    # build vocab
    vocab = Vocabulary(DATA_PATH + 'harry_potter_chars_train.pkl')

    # build model
    if MODEL_TYPE == 'transformer':
        model = HarryPotterTransformer(len(vocab), FEATURE_SIZE, NUM_HEADS).to(device)
        model.load_last_model(EXP_PATH + 'checkpoints/')
    else:
        raise NotImplementedError

    # generate
    # Try different seed_words. Find interesting ones.
    seed_words = "Don't worry, Harry "
    # Experiment with different temperatures to observe their impact on diversity and stability.
    temperature = 1.0
    # Experiment with different strategies.
    strategy = 'sampling'

    generated_sentence, array_length = generate(model, device, seed_words, SEQUENCE_LENGTH, 2000, vocab, temperature, strategy)
    print("Generated:\n", generated_sentence)
    print("Generated Length:", array_length)

    