# miao miao miao
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:  # pad the sentence
            word = ["<pad>"] * (n_step + 1 - len(word)) + word

        for word_index in range(len(word) - n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index + n_step]]  # create (1~n-1) as input
            # print(input)
            target = word2number_dict[
                word[word_index + n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch  # (batch num, batch size, n_step) (batch num, batch size)


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))  # set to list 并集
    # print(word_list)
    word2number_dict = {w: i + 4 for i, w in enumerate(word_list)}
    # print(word2number_dict)
    number2word_dict = {i + 4: w for i, w in enumerate(word_list)}

    # add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"
    # print(word2number_dict)
    # exit()
    return word2number_dict, number2word_dict


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(n_class, emb_size)  # 词嵌入矩阵
        # i_t
        self.W_ii = nn.Parameter(torch.Tensor(emb_size, n_hidden))  # [256,128]
        self.b_ii = nn.Parameter(torch.Tensor(n_hidden))
        self.W_hi = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_hi = nn.Parameter(torch.Tensor(n_hidden))
        # f_t
        self.W_if = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.b_if = nn.Parameter(torch.Tensor(n_hidden))
        self.W_hf = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_hf = nn.Parameter(torch.Tensor(n_hidden))
        # g_t
        self.W_ig = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.b_ig = nn.Parameter(torch.Tensor(n_hidden))
        self.W_hg = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_hg = nn.Parameter(torch.Tensor(n_hidden))

        # o_t
        self.W_io = nn.Parameter(torch.Tensor(emb_size, n_hidden))
        self.b_io = nn.Parameter(torch.Tensor(n_hidden))
        self.W_ho = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.b_ho = nn.Parameter(torch.Tensor(n_hidden))

        self.W = nn.Linear(n_hidden, n_class, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, X, hidden, cell):
        h_t_1 = hidden
        c_t_1 = cell
        X = self.embedding(X)
        # print(X.shape)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embeding size]
        for x_t in X:  # x_t[128,256]
            i_t = self.sigmoid(torch.matmul(x_t, self.W_ii) + self.b_ii + torch.matmul(h_t_1, self.W_hi) + self.b_hi)
            f_t = self.sigmoid(torch.matmul(x_t, self.W_if) + self.b_if + torch.matmul(h_t_1, self.W_hf) + self.b_hf)
            g_t = self.tanh(torch.matmul(x_t, self.W_ig) + self.b_ig + torch.matmul(h_t_1, self.W_hg) + self.b_hg)
            o_t = self.sigmoid(torch.matmul(x_t, self.W_io) + self.b_io + torch.matmul(h_t_1, self.W_ho) + self.b_ho)
            c_t = f_t * c_t_1 + i_t * g_t
            h_t = o_t * self.tanh(c_t)
            h_t_1 = h_t
            c_t_1 = c_t
        # reshape hidden_seq p/ retornar
        outputs = h_t[-1]  # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs)  # model : [batch_size, n_class]
        # print('model: ', model.shape)
        return model


def train_LSTMlm():
    model = TextLSTM()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()
            hidden_state = torch.zeros(1, batch_size,
                                       n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
            cell_state = torch.zeros(1, batch_size, n_hidden)  # [num_layers(=1) * num_directions(=1), b
            # print(input_batch.size())
            # print(hidden_state.size())
            # print(cell_state.size())
            # exit()
            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch, hidden_state, cell_state)
            # print(output.size())
            # exit()
            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
              'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target) * 128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                hidden_state = torch.zeros(1, batch_size, n_hidden)
                cell_state = torch.zeros(1, batch_size, n_hidden)
                valid_output = model(valid_batch, hidden_state, cell_state)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch + 1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/LSTMlm_model_epoch{epoch + 1}.ckpt')


def test_LSTMlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  # load the selected model
    model.to(device)

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target) * 128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        hidden_state = torch.zeros(1, batch_size, n_hidden)
        cell_state = torch.zeros(1, batch_size, n_hidden)
        test_output = model(test_batch, hidden_state, cell_state)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))


if __name__ == '__main__':
    n_step = 5  # number of cells(= number of Step)
    n_hidden = 128  # number of hidden units in one cell
    batch_size = 128  # batch size
    learn_rate = 0.0005
    all_epoch = 5  # the all epoch for training
    emb_size = 256  # embeding size
    save_checkpoint_epoch = 5  # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt')  # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  # n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]

    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    print("\nTrain the LSTMLM……………………")
    train_LSTMlm()

    print("\nTest the LSTMLM……………………")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    test_LSTMlm(select_model_path)

