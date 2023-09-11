import argparse
import torch
import os

import pandas as pd

from modules import models, modelPaper , residualmodels, residualmodels_1

from timm import create_model
from torchsummary import summary
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modules.datasets import phosc_dataset
from modules.engine import train_one_epoch, zslAccuracyTest, gzslAccuracyTest, gzslAccuracyTestAni
from modules.loss import PHOSCLoss

import torch.nn as nn

# function for defining all the commandline parameters
def get_args_parser():
    parser = argparse.ArgumentParser('Main', add_help=False)

    # Model mode:
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'pass'], required=True,
                        help='train or test a model')

    # Testing method gzsl or zsl
    parser.add_argument('--testing_mode', type=str, choices=['zsl', 'gzsl', 'gzslAni'], default="gzsl",required=False,
                        help='zsl or gzsl testing method')
    parser.add_argument('--words_list', default=None, required=False,
                        help='zsl or gzsl testing method')             
    
    # Model settings
    parser.add_argument('--name', type=str, help='Name of run')
    parser.add_argument('--model', type=str, help='Name of model')
    parser.add_argument('--pretrained_weights', type=str, help='the path to pretrained weights file')

    # Dataset folder paths
    parser.add_argument('--train_csv', type=str, help='The train csv')
    parser.add_argument('--train_folder', type=str, help='The train root folder')
    parser.add_argument('--valid_csv', type=str, help='The valid csv')
    parser.add_argument('--valid_folder', type=str, help='The valid root folder')

    parser.add_argument('--test_csv_seen', type=str, help='The seen test csv')
    parser.add_argument('--test_folder_seen', type=str, help='The seen test root folder')

    parser.add_argument('--test_csv_unseen', type=str, help='The unseen test csv')
    parser.add_argument('--test_folder_unseen', type=str, help='The unseen test root folder')

    # Dataloader settings
    parser.add_argument('--batch_size', type=int, default=32, help='number of samples per iteration in the epoch')
    parser.add_argument('--num_workers', default=5, type=int)

    # optimizer settings
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')

    # trainng related parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for')

    parser.add_argument('--stopCode', type=int, default=9999, help='stopping code')
    parser.add_argument('--flagFile', type=str, default="flag.txt", help='flagFile')

    # model related
    parser.add_argument('--phos_size', type=int, default=165, help='Phos representation size')
    parser.add_argument('--phoc_size', type=int, default=604, help='Phoc representation size')
    parser.add_argument('--language', type=str, default='eng', choices=['eng', 'nor', 'gw'], help='language which help decide which phosc representation to use')
    parser.add_argument("--prompts", type =int, default = 0)
    #parser.add_argument("--promptsWeight", type =int, default = 0)
    parser.add_argument('--promptsWeight', type=str, help='the path to pretrained weights file')


    return parser

def epoch_saving(epoch, model, promptModel, optimizer, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'prompt_model_state_dict': promptModel.state_dict(),
                    # 'prompter_state_dict': prompter.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename) #just change to your preferred folder/filename


def loadPrompt():
    if args.promptsWeight:
        if os.path.isfile(args.promptsWeight):
            print(("=> loading checkpoint '{}'".format(args.promptsWeight)))
            checkpoint = torch.load(args.promptsWeight)
            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            prompt_model.load_state_dict(checkpoint['prompt_model_state_dict'],strict=False)
            # prompter.load_state_dict(checkpoint['prompter_state_dict'],strict=False)
            
        else:
            print(("=> no checkpoint found at '{}'".format(args.promptsWeight)))



def main(args):
    print('Creating dataset...')

    print("\n\t model name:",args.model,"\t args.batch_size:",args.batch_size)

    print("\n\t arguments are:",args)



    if args.mode == 'train' or args.mode == "test":
        dataset_train = phosc_dataset(args.train_csv,
                                      args.train_folder,
                                      args.language,
                                      transforms.ToTensor())

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True
        )

        validate_model = False

        if 1:#args.valid_csv is not None or args.valid_folder is not None:
            validate_model = True

            dataset_valid = phosc_dataset(args.valid_csv,
                                          args.valid_folder,
                                          args.language,
                                          transforms.ToTensor())

            data_loader_valid = torch.utils.data.DataLoader(
                dataset_valid,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                drop_last=False,
                shuffle=True
            )

    if args.mode == 'train' or args.mode == 'test':
        dataset_test_seen = phosc_dataset(args.test_csv_seen,
                                     args.test_folder_seen,
                                     args.language,
                                     transforms.ToTensor())

        data_loader_test_seen = torch.utils.data.DataLoader(
            dataset_test_seen,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True
        )

        dataset_test_unseen = phosc_dataset(args.test_csv_unseen,
                                     args.test_folder_unseen,
                                     args.language,
                                     transforms.ToTensor())

        data_loader_test_unseen = torch.utils.data.DataLoader(
            dataset_test_unseen,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True
        )
    


    try:

        # data_loader_train , data_loader_valid, data_loader_test_seen, data_loader_test_unseen,

        print("\n\t data_loader_train len:",len(data_loader_train))
        dfTrain = data_loader_train.dataset.df_all
        dfValid = data_loader_valid.dataset.df_all
        dfTestSeen  = data_loader_test_seen.dataset.df_all
        dfTestUnSeen = data_loader_test_unseen.dataset.df_all

        allWords = list(set(dfTrain['Word'])) + list(set(dfValid['Word'])) + list(set(dfTestSeen['Word'])) + list(set(dfTestUnSeen['Word']))

        print("\n\t 0.total number of words:",len(allWords))

        allWords = list(set(allWords))

        print("\n\t 1.total number of words:",len(allWords))
        print("\n\t dfTrain.shape:",dfTrain.shape)
        print("\n\t dfValid.shape:",dfValid.shape)
        print("\n\t dfTest:",dfTestSeen.shape)
        print("\n\t dfTest:",dfTestSeen.shape)
        #     word_map = get_map_dict(list(set(df['Word'])))

    except Exception as e:
        print("\n\t exception in df!!!",e)

        import sys
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\n\t Line number:", exc_tb.tb_lineno)
        pass


    # setting the device to do stuff on
    print('Training on GPU:', torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: add phos and phoc size to create_model
    model = create_model(args.model, phos_size=args.phos_size, phoc_size=args.phoc_size).to(device)

    if args.prompts ==0:
        model.eval()
        print("\n\t original model put on eval mode!!!")
        promptModel = create_model("FixedPatchPrompter").to(device)


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # print summary of model
    #summary(model, (3, 50, 250))

    def training():
        if not os.path.exists(f'{args.name}/'):
            os.mkdir(args.name)
        """
        try:
            
            allModelSummary =     summary(model, (3, 50, 250))
            
            with open(args.name + '/' + 'summary.txt', 'a') as f:
                print(allModelSummary, file=f)

        except Exception as e:
            pass
        """

        try:
            if os.path.isfile(args.name+"//epoch.pt"):
                #model.load_state_dict(torch.load(args.pretrained_weights))
                model.load_state_dict(torch.load(args.name+"//epoch.pt"),strict=False)
                print("\n\t loadeding weights ",args.name+"//epoch.pt","\t complete")

            elif os.path.isfile(args.pretrained_weights):
                model.load_state_dict(torch.load(args.pretrained_weights),strict = True)
                print("\n\t loadeding weights ",args.pretrained_weights,"\t complete")

        except Exception as e:
            pass


        with open(args.name + '/' + 'log.csv', 'a') as f:
            f.write('epoch,loss,acc,lr\n')

        #opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)

        if args.prompts ==0:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
        elif args.prompts ==1:

            print("\n\t training prompter part!!!")
            opt = torch.optim.AdamW(promptModel.parameters(), lr=args.lr, weight_decay=5e-5)


        scheduler = ReduceLROnPlateau(opt, 'max', factor=0.25, patience=5, verbose=True, threshold=0.0001, cooldown=2,
                                      min_lr=1e-12)

        criterion = PHOSCLoss()

        mx_acc = 0
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):

            with open(args.flagFile, "r") as f:
                flag = f.read().strip()

            if flag == args.stopCode:
                epoch = args.epochs-1
                break

            if int(flag) == int(args.stopCode):
                epoch = args.epochs-1
                break

            if args.prompts == 0:
                mean_loss = train_one_epoch(model, criterion, data_loader_train, opt, device, epoch)
            elif args.prompts == 1: 
                mean_loss = train_one_epoch(model, criterion, data_loader_train, opt, device, epoch,promptModel,args)


            acc = -1
            if validate_model:
                acc, _, __ = zslAccuracyTest(model, data_loader_valid, device,promptModel,args)

                if acc > mx_acc:
                    mx_acc = acc
                    # removes previous best weights
                    if args.prompts == 0 and os.path.exists(f'{args.name}/epoch{best_epoch}.pt'):
                        os.remove(f'{args.name}/epoch{best_epoch}.pt')

                    best_epoch = epoch

                    #torch.save(model.state_dict(), f'{args.name}/epoch.pt')
                    
                    if args.prompts == 0:
                        torch.save(model.state_dict(), f'{args.name}/epoch.pt')
                    elif args.prompts == 1:
                        epoch_saving(epoch, model, promptModel, opt,f'{args.name}/epochPrompt.pt')


                scheduler.step(acc)

            with open(args.name + '/' + 'log.csv', 'a') as f:
                f.write(f'{epoch},{mean_loss},{acc},{opt.param_groups[0]["lr"]}\n')

            if (epoch%10)-1 == 0:

                #print("\n\t calling test")/zslAccuracyTest

                try:
                    testing(epoch)
                except Exception as e:
                    pass            
        
        if args.prompts == 0:
            torch.save(model.state_dict(), f'{args.name}/epoch.pt')
        elif args.prompts == 1:
            epoch_saving(epoch, model, promptModel, opt,f'{args.name}/epochPrompt.pt')
    


    def testing(epoch = 1000):

        try:
            if os.path.isfile(args.name+"//epoch.pt"):
                #model.load_state_dict(torch.load(args.pretrained_weights))
                model.load_state_dict(torch.load(args.name+"//epoch.pt"),strict=False)
            elif os.path.isfile(args.pretrained_weights):
                model.load_state_dict(torch.load(args.pretrained_weights),strict = False)
                
            if args.prompts == 0:
                torch.save(model.state_dict(), f'{args.name}/epoch.pt')
            elif args.prompts == 1:

                print("\n\t prompt model loaded for testing ")
                epoch_saving(epoch, model, promptModel, opt,f'{args.name}/epochPrompt.pt')

        except Exception as e:
            pass


        print(f'Testing {args.testing_mode}')

        if args.testing_mode == 'zsl' or args.testing_mode == 'gzsl':
            
            print("\n\t1.  ======================================================================================")
            print("\n\t WITH PROMPT ACCURACY:")
            acc_seen, _, __ = zslAccuracyTest(model, data_loader_test_seen, device,promptModel,1)
            acc_unseen, _, __ = zslAccuracyTest(model, data_loader_test_unseen, device,promptModel,1)
        
            print("\n\t\t WITH PROMPT ZSL ACCURACY NUMBERS:")
            print("\n\t\t 1. acc_seen:",acc_seen,"\t acc_unseen:",acc_unseen)


            print("\n\t ORIGINAL MODEL ACCURACY:")
            acc_seen1, _, __ = zslAccuracyTest(model, data_loader_test_seen, device,promptModel,0)
            acc_unseen1, _, __ = zslAccuracyTest(model, data_loader_test_unseen, device,promptModel,0)
            print("\n\t\t WITH ORIGINAL MODEL ZSL ACCURACY NUMBERS:")
            print("\n\t\t 11. acc_seen:",acc_seen1,"\t acc_unseen:",acc_unseen1)




        if args.testing_mode == 'zsl' or args.testing_mode == 'gzsl':
            if args.words_list is not None:
                df_words = pd.read_csv(args.words_list)
                args.words_list = list(df_words['Word'])
                
            print("\n\t2.  ======================================================================================")
            print("\n\t WITH PROMPT ACCURACY:")

            acc_seenGzsl, _, __ = gzslAccuracyTest(model, data_loader_test_seen, data_loader_test_unseen, device,1,promptModel, words_list=allWords)
            acc_unseenGzsl, _, __ = gzslAccuracyTest(model, data_loader_test_unseen, data_loader_test_seen, device,1,promptModel, words_list=allWords)
        
            print("\n\t\t WITH PROMPT GZSL ACCURACY NUMBERS:")
            print("\n\t\t 2.acc_seenGzsl,:",acc_seenGzsl,"\t acc_unseenGzsl,:",acc_unseenGzsl)

            print("\n\t ORIGINAL MODEL GZSL ACCURACY")
            acc_seenGzsl2, _, __ = gzslAccuracyTest(model, data_loader_test_seen, data_loader_test_unseen, device,0,promptModel, words_list=allWords)
            acc_unseenGzsl2, _, __ = gzslAccuracyTest(model, data_loader_test_unseen, data_loader_test_seen, device,0,promptModel, words_list=allWords)

            print("\n\t\t WITH PROMPT GZSL ACCURACY NUMBERS:")
            print("\n\t\t 22.acc_seenGzsl,:",acc_seenGzsl2,"\t acc_unseenGzsl,:",acc_unseenGzsl2)

        
        elif args.testing_mode == 'gzslAni':
            df_words = pd.read_csv(args.words_list)

            words = list(df_words['Word'])
            
            print("\n\t3.  ======================================================================================")

            print("\n\t WITH PROMPT ACCURACY:")

            _, _, _, _, _, acc_seen = gzslAccuracyTestAni(model, words, data_loader_test_seen, device,1,promptModel)
            _, _, _, _, _, acc_unseen = gzslAccuracyTestAni(model, words, data_loader_test_unseen, device, 1, args,promptModel)

            print("\n\t\t WITH PROMPT GZSL ACCURACY ANI NUMBERS:")
            print("\n\t\t 3. acc_seen:",acc_seen,"\t acc_unseen:",acc_unseen)


            print("\n\t ORIGINAL MODEL ACCURACY")
            _, _, _, _, _, acc_seen3 = gzslAccuracyTestAni(model, words, data_loader_test_seen, device,0,promptModel)
            _, _, _, _, _, acc_unseen3 = gzslAccuracyTestAni(model, words, data_loader_test_unseen, device, 0, args,promptModel)
            print("\n\t\t WITH ORIGINAL MODEL GZSL ACCURACY ANI NUMBERS:")
            print("\n\t\t 33. acc_seen:",acc_seen3,"\t acc_unseen:",acc_unseen3)



        with open(args.name + '/' + 'testresults.txt', 'a') as f:

            f.write(f'\n\t epoch: {epoch}\n')
            f.write(f'{args.model} {args.testing_mode} test results\n')
            f.write(f'Seen acc: {acc_seen}\n')
            f.write(f'Unseen acc: {acc_unseen}\n')
            
            f.write(f' \n Generalised ZSL')

            f.write(f'Seen acc: {acc_seenGzsl}\n')
            f.write(f'Unseen acc: {acc_unseenGzsl}\n')



        print(f'{args.testing_mode} accuracies of model: {args.model}')
        print('Seen accuracies:', acc_seen)
        print('Unseen accuracies:', acc_unseen)

    if args.mode == 'train':
        training()
    elif args.mode == 'test':
        testing()


# program start
if __name__ == '__main__':
    # creates commandline parser
    arg_parser = argparse.ArgumentParser('train ', parents=[get_args_parser()])
    args = arg_parser.parse_args()

    # passes the commandline argument to the main function
    main(args)
