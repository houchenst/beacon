import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys, argparse, math, glob, gc, traceback
import numpy as np

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '.'))

import utils

def RestrictedFloat_N10_100(x):
    x = float(x)
    MinMax = [-10.0, 100.0]
    if x < MinMax[0] or x > MinMax[1]:
        raise argparse.ArgumentTypeError('{} not in range [{}, {}]'.format(x, MinMax[0], MinMax[1]))
    return x

class netMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        loss = torch.mean((output - target) ** 2)
        return loss


class SuperLoss(nn.Module):
    def __init__(self, Losses=[], Weights=[], Names=[], CustomLosses=[]):
        super().__init__()
        if not Losses and not CustomLosses: # empty list
            self.Losses = [netMSELoss()]
            self.Weights = [1.0]
            self.Names = ['Default MSE Loss']
        else:
            if len(Losses) + len(CustomLosses) != len(Weights):
                raise RuntimeError('SuperLoss() given Losses and Weights don''t match.')

            self.Losses = Losses
            self.CustomLosses = CustomLosses
            self.Weights = Weights
            self.Names = ['Subloss ' + str(i).zfill(2) for i in range(len(self.Losses)+len(self.CustomLosses))]
            for Ctr, n in enumerate(Names, 0):
                self.Names[Ctr] = n
            self.cleanUp()

    def __len__(self):
        return len(self.Losses) + len(self.CustomLosses)

    def getItems(self, withoutWeights=False):
        RetLossValsFloat = []
        if withoutWeights:
            for v in self.LossVals:
                RetLossValsFloat.append(v.item())
        else:
            for v in self.LossValsWeighted:
                RetLossValsFloat.append(v.item())

        return RetLossValsFloat

    def cleanUp(self):
        self.LossVals = [0.0] * (len(self.Losses) + len(self.CustomLosses))
        self.LossValsWeighted = [0.0] * (len(self.Losses) + len(self.CustomLosses))

    def forward(self, output, target, otherInputs={}):
        self.cleanUp()
        return self.computeLoss(output, target, otherInputs=otherInputs)

    def computeLoss(self, output, target, otherInputs={}):
        TotalLossVal = 0.0

        for Ctr, (l, w, custom) in enumerate(zip(self.Losses + self.CustomLosses, self.Weights, [False]*len(self.Losses) + [True]*len(self.CustomLosses)), 0):
            if not custom:
                LossVal = l.forward(output, target, otherInputs={})
            else:
                LossVal = l.forward(output, target, otherInputs=otherInputs)
            self.LossVals[Ctr] = LossVal
            self.LossValsWeighted[Ctr] = w * LossVal
            TotalLossVal += self.LossValsWeighted[Ctr]

        return TotalLossVal


class SuperNetExptConfig():
    def __init__(self, InputArgs=None, isPrint=True):
        self.Parser = argparse.ArgumentParser(description='Parse arguments for a PyTorch neural network.', fromfile_prefix_chars='@')

        # Search params
        self.Parser.add_argument('--learning-rate', help='Choose the learning rate.', required=False, default=0.001,
                            type=RestrictedFloat_N10_100)
        self.Parser.add_argument('--batch-size', help='Choose mini-batch size.', required=False, default=16, type=int)

        # Machine-specific params
        self.Parser.add_argument('--expt-name', help='Provide a name for this experiment.')
        self.Parser.add_argument('--input-dir', help='Provide the input directory where datasets are stored.')
        # -----
        self.Parser.add_argument('--output-dir',
                            help='Provide the *absolute* output directory where checkpoints, logs, and other output will be stored (under expt_name).')
        self.Parser.add_argument('--rel-output-dir',
                            help='Provide the *relative* (pwd or config file) output directory where checkpoints, logs, and other output will be stored (under expt_name).')
        # -----
        self.Parser.add_argument('--epochs', help='Choose number of epochs.', required=False, default=10, type=int)
        self.Parser.add_argument('--save-freq', help='Choose epoch frequency to save checkpoints. Zero (0) will only at the end of training [not recommended].', choices=range(0, 10000), metavar='0..10000',
                            required=False, default=5, type=int)

        # -----
        self.Parser.add_argument('--no-save', help='Disable checkpoint saving after every epoch. CAUTION: Use this for debugging only.', action='store_true', required=False)
        self.Parser.set_defaults(no_save=False)

        self.Args, _ = self.Parser.parse_known_args(InputArgs)

        if self.Args.expt_name is None:
            raise RuntimeError('No experiment name (--expt-name) provided.')

        if self.Args.rel_output_dir is None and self.Args.output_dir is None:
            raise RuntimeError('One or both of --output-dir or --rel-output-dir is required.')

        if self.Args.rel_output_dir is not None: # Relative path takes precedence
            if self.Args.output_dir is not None:
                print('[ INFO ]: Relative path taking precedence to absolute path.')
            DirPath = os.getcwd() # os.path.dirname(os.path.realpath(__file__))
            for Arg in InputArgs:
                if '@' in Arg: # Config file is passed, path should be relative to config file
                    DirPath = os.path.abspath(os.path.dirname(utils.expandTilde(Arg[1:]))) # Abs directory path of config file
                    break
            self.Args.output_dir = os.path.join(DirPath, self.Args.rel_output_dir)
            print('[ INFO ]: Converted relative path {} to absolute path {}'.format(self.Args.rel_output_dir, self.Args.output_dir))

        # Logging directory and file
        self.ExptDirPath = ''
        self.ExptDirPath = os.path.join(utils.expandTilde(self.Args.output_dir), self.Args.expt_name)
        if os.path.exists(self.ExptDirPath) == False:
            os.makedirs(self.ExptDirPath)

        self.ExptLogFile = os.path.join(self.ExptDirPath, self.Args.expt_name + '_' + utils.getTimeString('humanlocal') + '.log')
        # if os.path.exists(self.ExptLogFile) == False:
        with open(self.ExptLogFile, 'w+', newline='') as f:
            os.utime(self.ExptLogFile, None)

        sys.stdout = utils.beaconLogger(sys.stdout, self.ExptLogFile)
        sys.stderr = utils.beaconLogger(sys.stderr, self.ExptLogFile)

        if isPrint:
            print('-'*60)
            ArgsDict = vars(self.Args)
            for Arg in ArgsDict:
                if ArgsDict[Arg] is not None:
                    print('{:<15}:   {:<50}'.format(Arg, ArgsDict[Arg]))
                else:
                    print('{:<15}:   {:<50}'.format(Arg, 'NOT DEFINED'))
            print('-'*60)

    def getHelp(self):
        self.Parser.print_help()

    def serialize(self, FilePath, isAppend=True):
        utils.configSerialize(self.Args, FilePath, isAppend)


class SuperNet(nn.Module):
    def __init__(self, Args=None):
        super().__init__()

        self.Config = SuperNetExptConfig(InputArgs=Args)

        # Defaults
        self.StartEpoch = 0
        self.ExptDirPath = self.Config.ExptDirPath
        self.SaveFrequency = self.Config.Args.save_freq if self.Config.Args.save_freq > 0 else self.Config.Args.epochs
        self.LossHistory = []
        self.ValLossHistory = []
        self.SeparateLossesHistory = []
        self.Optimizer = None

    def loadCheckpoint(self, Path=None, Device='cpu', OtherParameters=[], OtherParameterNames=[]):
        if Path is None:
            self.ExptDirPath = os.path.join(utils.expandTilde(self.Config.Args.output_dir), self.Config.Args.expt_name)
            print('[ INFO ]: Loading from latest checkpoint.')
            CheckpointDict = utils.loadLatestPyTorchCheckpoint(self.ExptDirPath, map_location=Device)
        else: # Load latest
            print('[ INFO ]: Loading from checkpoint {}'.format(Path))
            CheckpointDict = utils.loadPyTorchCheckpoint(Path)

        self.load_state_dict(CheckpointDict['ModelStateDict'])

        for i, param_name in enumerate(OtherParameterNames):
            if param_name in CheckpointDict:
                OtherParameters[i].load_state_dict(CheckpointDict[param_name])
            else:
                print(f"[ INFO ]: Could not find parameter '{param_name}' in checkpoint")
        

    def setupCheckpoint(self, TrainDevice, OtherParameters=[], OtherParameterNames=[]):
        LatestCheckpointDict = None
        AllCheckpoints = glob.glob(os.path.join(self.ExptDirPath, '*.tar'))
        if len(AllCheckpoints) > 0:
            LatestCheckpointDict = utils.loadLatestPyTorchCheckpoint(self.ExptDirPath, map_location=TrainDevice)
            print('[ INFO ]: Loading from last checkpoint.')

        if LatestCheckpointDict is not None:
            # Make sure experiment names match
            if self.Config.Args.expt_name == LatestCheckpointDict['Name']:
                self.load_state_dict(LatestCheckpointDict['ModelStateDict'])
                self.StartEpoch = LatestCheckpointDict['Epoch']
                if self.Optimizer is not None:
                    self.Optimizer.load_state_dict(LatestCheckpointDict['OptimizerStateDict'])
                self.LossHistory = LatestCheckpointDict['LossHistory']
                if 'ValLossHistory' in LatestCheckpointDict:
                    self.ValLossHistory = LatestCheckpointDict['ValLossHistory']
                else:
                    self.ValLossHistory = self.LossHistory
                if 'SeparateLossesHistory' in LatestCheckpointDict:
                    self.SeparateLossesHistory = LatestCheckpointDict['SeparateLossesHistory']
                else:
                    self.SeparateLossesHistory = self.LossHistory

                for i, param_name in enumerate(OtherParameterNames):
                    if param_name in LatestCheckpointDict:
                        OtherParameters[i].load_state_dict(LatestCheckpointDict[param_name])
                    else:
                        print(f"[ INFO ]: Parameter '{param_name}' not found in last checkpoint. Using reinitialized values.")

                # Move optimizer state to GPU if needed. See https://github.com/pytorch/pytorch/issues/2830
                if TrainDevice != 'cpu' and self.Optimizer is not None:
                    for state in self.Optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(TrainDevice)
            else:
                print('[ INFO ]: Experiment names do not match. Training from scratch.')

    def validate(self, ValDataLoader, Objective, Device='cpu', OtherParameterDict={}):
        self.eval()         #switch to evaluation mode
        ValLosses = []
        Tic = utils.getCurrentEpochTime()
        # print('Val length:', len(ValDataLoader))
        for i, (Data, Targets) in enumerate(ValDataLoader, 0):  # Get each batch
            DataTD = utils.sendToDevice(Data, Device)
            TargetsTD = utils.sendToDevice(Targets, Device)

            Output = self.forward(DataTD, OtherParameterDict)
            Loss = Objective(Output, TargetsTD, otherInputs={"data": DataTD, "model": self})
            ValLosses.append(Loss.item())

            # Print stats
            Toc = utils.getCurrentEpochTime()
            Elapsed = math.floor((Toc - Tic) * 1e-6)
            done = int(50 * (i+1) / len(ValDataLoader))
            sys.stdout.write(('\r[{}>{}] val loss - {:.8f}, elapsed - {}')
                             .format('+' * done, '-' * (50 - done), np.mean(np.asarray(ValLosses)), utils.getTimeDur(Elapsed)))
            sys.stdout.flush()
        sys.stdout.write('\n')
        self.train()     #switch back to train mode

        return ValLosses

    def fit(self, TrainDataLoader, Optimizer=None, Objective=nn.MSELoss(), TrainDevice='cpu', ValDataLoader=None, OtherParameters=[], OtherParameterNames=[]):
        if len(OtherParameters) != len(OtherParameterNames):
                raise RuntimeError('fit() OtherParameters and OtherParameterNames don''t match.')
        for i, param in enumerate(OtherParameters):
            if not issubclass(type(param), nn.Module):
                raise RuntimeError(f"fit() OtherParameter '{OtherParameters[i]}' must be an instance of nn.Module")

        if self.Optimizer is None:
            self.Optimizer = optim.Adam(self.parameters(), lr=self.Config.Args.learning_rate, weight_decay=1e-5)  # PARAM
            # TODO: Do we want to include the other parameters in the optimizer by default?
        if Optimizer is not None:
            self.Optimizer = Optimizer

        ObjectiveFunc = Objective
        if isinstance(ObjectiveFunc, SuperLoss) == False:
            ObjectiveFunc = SuperLoss(Losses=[ObjectiveFunc], Weights=[1.0])  # Cast to SuperLoss

        self.setupCheckpoint(TrainDevice, OtherParameters, OtherParameterNames)

        print('[ INFO ]: Training on {}'.format(TrainDevice))
        self.to(TrainDevice)

        #Move other parameters to GPU if needed
        if TrainDevice != 'cpu':
            GPUParameters = []
            for param in OtherParameters:
                GPUParameters.append(param.to(TrainDevice))
            OtherParameters = GPUParameters
        
        OtherParameterDict = {OtherParameterNames[i]: OtherParameters[i] for i in range(len(OtherParameterNames))}

        CurrLegend = ['Train loss', *ObjectiveFunc.Names]

        AllTic = utils.getCurrentEpochTime()
        for Epoch in range(self.Config.Args.epochs):
            try:
                EpochLosses = [] # For all batches in an epoch
                EpochSeparateLosses = []  # For all batches in an epoch
                Tic = utils.getCurrentEpochTime()
                for i, Examples in enumerate(TrainDataLoader, 0):  # Get each batch
                    Data, Targets = Examples
                            
                    DataTD = utils.sendToDevice(Data, TrainDevice)
                    TargetsTD = utils.sendToDevice(Targets, TrainDevice)

                    self.Optimizer.zero_grad()

                    # Forward, backward, optimize
                    Output = self.forward(DataTD, OtherParameterDict)
                    Loss = ObjectiveFunc(Output, TargetsTD, otherInputs={"data": DataTD, "model": self})
                    Loss.backward()

                    # print(type(Data))
                    # print(Data)
                    # print(f"lat vec: {OtherParameterDict['Latent Vectors'].weight}")
                    # print(f"lat vec grad: {OtherParameterDict['Latent Vectors'].weight.grad}")

                    self.Optimizer.step()
                    EpochLosses.append(Loss.item())
                    EpochSeparateLosses.append(ObjectiveFunc.getItems())

                    gc.collect() # Collect garbage after each batch

                    # Terminate early if loss is nan
                    isTerminateEarly = False
                    if math.isnan(EpochLosses[-1]):
                        print('[ WARN ]: NaN loss encountered. Terminating training and saving current model checkpoint (might be junk).')
                        isTerminateEarly = True
                        break

                    # Print stats
                    Toc = utils.getCurrentEpochTime()
                    Elapsed = math.floor((Toc - Tic) * 1e-6)
                    TotalElapsed = math.floor((Toc - AllTic) * 1e-6)
                    # Compute ETA
                    TimePerBatch = (Toc - AllTic) / ((Epoch * len(TrainDataLoader)) + (i+1)) # Time per batch
                    ETA = math.floor(TimePerBatch * self.Config.Args.epochs * len(TrainDataLoader) * 1e-6)
                    done = int(50 * (i+1) / len(TrainDataLoader))
                    ProgressStr = ('\r[{}>{}] epoch - {}/{}, train loss - {:.8f} | epoch - {}, total - {} ETA - {} |').format('=' * done, '-' * (50 - done), self.StartEpoch + Epoch + 1, self.StartEpoch + self.Config.Args.epochs
                                                                                                                              , np.mean(np.asarray(EpochLosses)), utils.getTimeDur(Elapsed), utils.getTimeDur(TotalElapsed), utils.getTimeDur(ETA - TotalElapsed))
                    sys.stdout.write(ProgressStr.ljust(150))
                    sys.stdout.flush()
                sys.stdout.write('\n')

                self.LossHistory.append(np.mean(np.asarray(EpochLosses)))

                # Transpose and sum: https://stackoverflow.com/questions/47114706/python-sum-first-element-of-a-list-of-lists
                SepMeans = list(map(sum, zip(*EpochSeparateLosses)))
                SepMeans[:] = [x / len(EpochLosses) for x in SepMeans]
                self.SeparateLossesHistory.append(SepMeans)
                if ValDataLoader is not None:
                    ValLosses = self.validate(ValDataLoader, Objective, TrainDevice, OtherParameterDict)
                    self.ValLossHistory.append(np.mean(np.asarray(ValLosses)))
                    # print('Last epoch val loss - {:.16f}'.format(self.ValLossHistory[-1]))
                    CurrLegend = ['Train loss', 'Val loss', *ObjectiveFunc.Names]

                # Always save checkpoint after an epoch. Will be replaced each epoch. This is independent of requested checkpointing
                if self.Config.Args.no_save == False:
                    self.saveCheckpoint(Epoch, CurrLegend, OtherParameterDict, TimeString='eot', PrintStr='~'*3)

                isLastLoop = (Epoch == self.Config.Args.epochs-1) and (i == len(TrainDataLoader)-1)
                if (Epoch + 1) % self.SaveFrequency == 0 or isTerminateEarly or isLastLoop:
                    self.saveCheckpoint(Epoch, CurrLegend, OtherParameterDict)
                    if isTerminateEarly:
                        break
            except (KeyboardInterrupt, SystemExit):
                print('\n[ INFO ]: KeyboardInterrupt detected. Saving checkpoint.')
                self.saveCheckpoint(Epoch, CurrLegend, OtherParameterDict, TimeString='eot', PrintStr='$'*3)
                break
            except Exception as e:
                print(traceback.format_exc())
                print('\n[ WARN ]: Exception detected. *NOT* saving checkpoint. {}'.format(e))
                # self.saveCheckpoint(Epoch, CurrLegend, OtherParameterDict, TimeString='eot', PrintStr='$'*3)
                break

        AllToc = utils.getCurrentEpochTime()
        print('[ INFO ]: All done in {}.'.format(utils.getTimeDur((AllToc - AllTic) * 1e-6)))

    def saveCheckpoint(self, Epoch, CurrLegend, OtherParameterDict, TimeString='humanlocal', PrintStr='*'*3):
        CheckpointDict = {
            'Name': self.Config.Args.expt_name,
            'ModelStateDict': self.state_dict(),
            'OptimizerStateDict': self.Optimizer.state_dict(),
            'LossHistory': self.LossHistory,
            'ValLossHistory': self.ValLossHistory,
            'SeparateLossesHistory': self.SeparateLossesHistory,
            'Epoch': self.StartEpoch + Epoch + 1,
            'SavedTimeZ': utils.getZuluTimeString(),
        }
        
        for key in OtherParameterDict:
            CheckpointDict[key] = OtherParameterDict[key].state_dict()
        
        OutFilePath = utils.savePyTorchCheckpoint(CheckpointDict, self.ExptDirPath, TimeString=TimeString)
        # ptUtils.saveLossesCurve(self.LossHistory, self.ValLossHistory, out_path=os.path.splitext(OutFilePath)[0] + '.png',
        #                         xlim = [0, int(self.Config.Args.epochs + self.StartEpoch)], legend=CurrLegend, title=self.Config.Args.expt_name)
        TSLH = list(map(list, zip(*self.SeparateLossesHistory))) # Transposed list
        try:
            utils.saveLossesCurve(self.LossHistory, self.ValLossHistory, *TSLH, out_path=os.path.splitext(OutFilePath)[0] + '.png',
                                  xlim = [0, int(self.Config.Args.epochs + self.StartEpoch)], legend=CurrLegend, title=self.Config.Args.expt_name)
        except Exception as e:
            print('[ WARN ]: Failed to write loss curve. On some operating systems having the file open can cause write problems. Please close the file.')

        # print('[ INFO ]: Checkpoint saved.')
        print(PrintStr) # Checkpoint saved. 50 + 3 characters [>]

    def forward(self, x, otherParameters):
        print('[ WARN ]: This is an identity network. Override this in a derived class.')
        return x
