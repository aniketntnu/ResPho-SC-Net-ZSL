import torch

import pandas as pd
import torch.nn as nn

from typing import Iterable
from modules.loss import PHOSCLoss

from modules.utils import get_map_dict
from modules.datasets import CharacterCounterDataset


from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: PHOSCLoss,
                    dataloader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,promptModel,args):

    #model.train(True)

    if args.prompts ==1:
        #model.eval()
        promptModel.train(True)
    else:
        model.train(True)

    n_batches = len(dataloader)
    batch = 1
    loss_over_epoch = 0

    pbar = tqdm(dataloader)

    for batch in pbar:
        # Putting images and targets on given device
        
        #print("\n\t img size:",batch['image'].shape,"\t 1st img:",batch['image'][0].shape)


        if args.prompts ==0:
            batch['image'] = batch['image'].to(device, non_blocking=True)
            model.zero_grad()
        
        elif args.prompts ==1:

            promptModel.zero_grad()
            batch['image'] = promptModel(batch['image'].to(device, non_blocking=True))
            #promptModel.zero_grad()
        
        batch['y_vectors']['phos'] = batch['y_vectors']['phos'].to(device, non_blocking=True)
        batch['y_vectors']['phoc'] = batch['y_vectors']['phoc'].to(device, non_blocking=True)
        batch['y_vectors']['phosc'] = batch['y_vectors']['phosc'].to(device, non_blocking=True)

        # zeroing gradients before next pass through
        #model.zero_grad()

        # passing images in batch through model
        outputs = model(batch['image'])

        # calculating loss and backpropagation the loss through the network
        loss = criterion(outputs, batch['y_vectors'])
        loss.backward()

        # adjusting weight according to backpropagation
        optimizer.step()

        # accumulating loss over complete epoch
        loss_over_epoch += loss.item()

        pbar.set_description(f'loss: {loss}')

    # mean loss for the epoch
    mean_loss = loss_over_epoch / n_batches

    return mean_loss


# tensorflow accuracy function, modified for pytorch
@torch.no_grad()
def zslAccuracyTest(model, dataloader: Iterable, device: torch.device,promptModel,promptFlag):
    # set in model in training mode

    if promptFlag ==1:
        #model.eval()
        promptModel.eval()
    else:
        model.eval()


    #model.eval()

    # gets the dataframe with all images, words and word vectors
    df = dataloader.dataset.df_all

    # gets the word map dictionary
    word_map = get_map_dict(list(set(df['Word'])))

    # number of correct predicted
    n_correct = 0
    no_of_images = len(df)

    # accuracy per word length
    acc_by_len = dict()

    # number of words per word length
    word_count_by_len = dict()

    # fills up the 2 described dictionaries over
    for w in df['Word'].tolist():
        acc_by_len[len(w)] = 0
        word_count_by_len[len(w)] = 0

    # Predictions list
    Predictions = []

    # this will not work with the current dataloader
    for batch in tqdm(dataloader):

        #batch['image'] = batch['image'].to(device, non_blocking=True)

        if promptFlag ==0:
            batch['image'] = batch['image'].to(device, non_blocking=True)

        elif promptFlag ==1:
            batch['image'] = promptModel(batch['image'].to(device, non_blocking=True))

        batch['y_vectors']['phos'] = batch['y_vectors']['phos'].to(device, non_blocking=True)
        batch['y_vectors']['phoc'] = batch['y_vectors']['phoc'].to(device, non_blocking=True)
        batch['y_vectors']['phosc'] = batch['y_vectors']['phosc'].to(device, non_blocking=True)

        vector_dict = model(batch['image'])
        vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)

        phosc_size = vectors.shape[1]

        for i in range(len(batch['word'])):
            target_word = batch['word'][i]
            pred_vector = vectors[i].view(-1, phosc_size)
            mx = -1

            for w in word_map:
                temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w]).to(device))
                if temp > mx:
                    mx = temp
                    pred_word = w

            Predictions.append((batch['image'][i], target_word, pred_word))

            if pred_word == target_word:
                n_correct += 1
                acc_by_len[len(target_word)] += 1

            word_count_by_len[len(target_word)] += 1

    for w in acc_by_len:
        if acc_by_len[w] != 0:
            acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100

    df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])

    acc = n_correct / no_of_images

    print('\n\t ZSL acc:', acc)

    return acc, df, acc_by_len


# tensorflow accuracy function, modified for pytorch
# dataloader_main contains the samples which will be tested
# dataloader_secondary contains the samples which won't be tested, but which words should also be added to the search space
# example: _main = seen_words_dataloader, _secondary = unseen_words_dataloader. For seen gzsl acc. Reverse for unseen gzsl acc.
@torch.no_grad()
def gzslAccuracyTest(model, dataloader_main: Iterable, dataloader_secondary: Iterable, device: torch.device,promptFlag,promptModel, words_list=None):
    # set in model in training mode
    #model.eval()

    if promptFlag ==1:
        promptModel.eval()
    else:
        model.eval()

    # gets the dataframe with all images, words and word vectors
    df_seen = dataloader_main.dataset.df_all
    df_unseen = dataloader_secondary.dataset.df_all

    # create list for seen and unseen words
    if words_list is None:
        seen_words = list(set(df_seen['Word']))
        unseen_words = list(set(df_unseen['Word']))
        words = list(set(seen_words + unseen_words))

        print('size seen map', len(seen_words))
        print('size unseen map',len(unseen_words))
    else:
        words = list(set(words_list))

    
    print('size total map',len(words))

    # gets the word map dictionary
    word_map = get_map_dict(words)

    # number of correct predicted
    n_correct = 0
    no_of_images = len(df_seen)

    # accuracy per word length
    acc_by_len = dict()

    # number of words per word length
    word_count_by_len = dict()

    # fills up the 2 described dictionaries over
    for w in df_seen['Word'].tolist():
        acc_by_len[len(w)] = 0
        word_count_by_len[len(w)] = 0

    # Predictions list
    Predictions = []

    # this will not work with the current dataloader
    for batch in tqdm(dataloader_main):

        #batch['image'] = batch['image'].to(device, non_blocking=True)

        if promptFlag ==0:
            batch['image'] = batch['image'].to(device, non_blocking=True)
        elif promptFlag ==1:
            batch['image'] = promptModel(batch['image'].to(device, non_blocking=True))

        batch['y_vectors']['phos'] = batch['y_vectors']['phos'].to(device, non_blocking=True)
        batch['y_vectors']['phoc'] = batch['y_vectors']['phoc'].to(device, non_blocking=True)
        batch['y_vectors']['phosc'] = batch['y_vectors']['phosc'].to(device, non_blocking=True)

        vector_dict = model(batch['image'])
        vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)

        phosc_size = vectors.shape[1]

        for i in range(len(batch['word'])):
            target_word = batch['word'][i]
            pred_vector = vectors[i].view(-1, phosc_size)
            mx = -1

            for w in word_map:
                temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w]).to(device))
                if temp > mx:
                    mx = temp
                    pred_word = w

            Predictions.append((batch['image'][i], target_word, pred_word))

            if pred_word == target_word:
                n_correct += 1
                acc_by_len[len(target_word)] += 1

            word_count_by_len[len(target_word)] += 1

    for w in acc_by_len:
        if acc_by_len[w] != 0:
            acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100

    df_seen = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])

    acc = (n_correct / no_of_images) *100 

    print('\n\t gzsl acc:', acc,"\t n_correct:",n_correct,"\t no_of_images:",no_of_images)

    return acc, df_seen, acc_by_len


# tensorflow accuracy function, modified for pytorch
@torch.no_grad()
def gzslAccuracyTestAni(model, allWords, dataloader: Iterable, device: torch.device, lenEstimation, promptFlag,promptModel):
    # set in model in training mode
    #model.eval()

    if promptFlag ==1:
        promptModel.eval()
    else:
        model.eval()

    # gets the dataframe with all images, words and word vectors
    df = dataloader.dataset.df_all

    # gets the word map dictionary
    word_map = get_map_dict(list(set(df['Word'])))
    word_mapGzsl = get_map_dict(allWords)

    #print("\n\t original no of words:",len(df["Word"]))
    #print("\n\t new no of words:",len(allWords))


    # number of correct predicted
    n_correct = 0
    n_correctGzsl = 0

    no_of_images = len(allWords)

    # accuracy per word length
    acc_by_len = dict()
    acc_by_lenGzsl = dict()

    # number of words per word length
    word_count_by_len = dict()
    word_count_by_lenGzsl = dict()


    # fills up the 2 described dictionaries over

    #allWords = df['Word'].tolist()

    #print("\n\t allWords:",allWords)

    for w in df['Word'].tolist():
        acc_by_len[len(w)] = 0
        word_count_by_len[len(w)] = 0

    for w in allWords:
        acc_by_lenGzsl[len(w)] = 0
        word_count_by_lenGzsl[len(w)] = 0


    # Predictions list
    Predictions = []
    lengthAccuracy = 0
    fuzzyAccuracy = 0

    PredictionsGzsl = []
    lengthAccuracyGzsl = 0
    fuzzyAccuracyGzsl = 0


    threshold = torch.tensor([0.5]).cuda()
    
    #print("\n\t *****inside engine ZSL Accuracy lenEstimation:",lenEstimation,"\t is True:",lenEstimation == True," \t is True2:",lenEstimation == "True","\t lenEstimation ==",lenEstimation ==2)

    # this will not work with the current dataloader
    for batch in tqdm(dataloader):

        #batch['image'] = batch['image'].to(device, non_blocking=True)

        if promptFlag ==0:
            batch['image'] = batch['image'].to(device, non_blocking=True)
        elif promptFlag ==1:
            batch['image'] = promptModel(batch['image'].to(device, non_blocking=True))

        batch['y_vectors']['phos'] = batch['y_vectors']['phos'].to(device, non_blocking=True)
        batch['y_vectors']['phoc'] = batch['y_vectors']['phoc'].to(device, non_blocking=True)
        batch['y_vectors']['phosc'] = batch['y_vectors']['phosc'].to(device, non_blocking=True)
        
        """
        if lenEstimation:
            batch['y_vectors']["length_embeddings"] = batch['y_vectors']["length_embeddings"].to(device, non_blocking=True)
        """

        vector_dict = model(batch['image'])
        vectors = torch.cat((vector_dict['phos'], vector_dict['phoc']), dim=1)


        # y['len_vec_sigmoid'] 
        
        if lenEstimation == "True":
            
            batch['y_vectors']["length_embeddings"] = batch['y_vectors']["length_embeddings"].to(device, non_blocking=True)

            predLenVector = vector_dict['len_vec_sigmoid']
            #predLenVector = ( predLenVector > threshold ).float()*1
        
            #print("\n\t predLenVector =",predLenVector[0])
            #print("\n\t predLenVector len =",predLenVector[0].shape)

        phosc_size = vectors.shape[1]

        for i in range(len(batch['word'])):
            target_word = batch['word'][i]
            pred_vector = vectors[i].view(-1, phosc_size)
            

            if lenEstimation == "True":
                lenVect = predLenVector[i]
                realLength = len(target_word)
            
                """
                    pred Length
                """
        
                lenVect = ( lenVect > threshold ).float()*1
                lenPred = sum(lenVect).cpu().detach().numpy()

            #print("\n\t predicted length:",sum(lenVect).cpu().detach().numpy()," \t realLength:",realLength)
            #print("\n\t predicted len vect:",lenVect)


            mx = -1

            for w in word_map:
                temp = torch.cosine_similarity(pred_vector, torch.tensor(word_map[w]).to(device))
                if temp > mx:
                    mx = temp
                    pred_word = w

            Predictions.append((batch['image'][i], target_word, pred_word))

            if pred_word == target_word:
                n_correct += 1
                acc_by_len[len(target_word)] += 1
            

            mx = -1

            for w in word_mapGzsl:
                temp = torch.cosine_similarity(pred_vector, torch.tensor(word_mapGzsl[w]).to(device))
                if temp > mx:
                    mx = temp
                    pred_word = w

            PredictionsGzsl.append((batch['image'][i], target_word, pred_word))

            if pred_word == target_word:
                n_correctGzsl += 1
                acc_by_lenGzsl[len(target_word)] += 1

            """
            lengthAccuracy = 0
            fuzzyAccuracy = 0

            """

            if lenEstimation == "True":
                
                if lenPred == realLength:
                    lengthAccuracy +=1
                
                elif (realLength-1) <= lenPred <= (realLength+1):
                    fuzzyAccuracy +=1
            else:
                lengthAccuracy = 0
                fuzzyAccuracy = 0


            word_count_by_len[len(target_word)] += 1
            word_count_by_lenGzsl[len(target_word)] += 1

    for w in acc_by_len:
        if acc_by_len[w] != 0:
            acc_by_len[w] = acc_by_len[w] / word_count_by_len[w] * 100

    df = pd.DataFrame(Predictions, columns=["Image", "True Label", "Predicted Label"])

    acc = (n_correct / no_of_images) * 100
    accGzsl = (n_correctGzsl / no_of_images) * 100


    print('\n\t phosc ZSL acc:', acc) #," \t lengthAccuracy:",lengthAccuracy / no_of_images ," \t fuzzyAccuracy:",fuzzyAccuracy / no_of_images)
    print("\n\t n_correct ZSL:",n_correct,"\t no_of_images:",no_of_images)

    print('\n\t phosc Generalised ZSL acc:', accGzsl) #," \t lengthAccuracy:",lengthAccuracy / no_of_images ," \t fuzzyAccuracy:",fuzzyAccuracy / no_of_images)
    print("\n\t n_correct ZSL:",n_correctGzsl,"\t no_of_images:",no_of_images)


    return acc, df, acc_by_len,lengthAccuracy,fuzzyAccuracy, accGzsl



# reserved for testing the character counter model
@torch.no_grad()
def test_accuracy(model: torch.nn.Module, dataloader: CharacterCounterDataset, device: torch.device):
    # set model in evaluation mode. turns of dropout layers and other layers which only are used for training. same
    # as .train(False)
    model.eval()

    # how many correct classified images
    cnt = 0

    for samples, targets, _ in tqdm(dataloader):
        # puts tensors onto devices
        samples = samples.to(device)
        targets = targets.to(device)

        # pass image and get output vector
        output = model(samples)

        # get argmax for output and target
        argmax_output = torch.argmax(output, dim=1)
        argmax_target = torch.argmax(targets, dim=1)

        for i in range(len(argmax_output)):
            if argmax_output[i] == argmax_target[i]:
                cnt += 1
                
                # print(argmax_output[i], argmax_target[i])
                # print(output[i], targets[i])
        
    # print(cnt)
    # print(len(dataloader.dataset))
    # print(cnt / len(dataloader.dataset))
    # number of correct predicted / total number of samples
    return cnt / len(dataloader.dataset)
