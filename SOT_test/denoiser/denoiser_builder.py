import torch
import torch.nn.functional as F
import numpy as np

from .CDT.model.CDT_arch import CDT

DENOISERS = {
          'CDT': CDT,
         }

class Denoiser():
    def __init__(self, model, args):
        super(Denoiser, self).__init__()
        
        self.model = model
        self.model.cuda()

        checkpoint = torch.load(args.d_weights)
        # for key in checkpoint.keys():
        #     print(key)
        # print(checkpoint['params'])
        self.model.load_state_dict(checkpoint['params'])
        model.eval()

        self.multiples = 8

    def denoise(self, img):

        input_ = torch.div(img, 255.)

        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+self.multiples)//self.multiples)*self.multiples, ((w+self.multiples)//self.multiples)*self.multiples
        padh = H-h if h%self.multiples!=0 else 0
        padw = W-w if w%self.multiples!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = self.model(input_)

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:h,:w]

        return torch.mul(restored, 255.)
    
    def single_denoise(self, img):
        img = (np.asarray(img)/255.0)
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)
        input_ = img.cuda().unsqueeze(0)

        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+self.multiples)//self.multiples)*self.multiples, ((w+self.multiples)//self.multiples)*self.multiples
        padh = H-h if h%self.multiples!=0 else 0
        padw = W-w if w%self.multiples!=0 else 0
        

        restored = self.model(F.pad(input_, (0,padw,0,padh), 'reflect'))

        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        return restored[:,:,:h,:w]


def build_denoiser(args):

    parameters = {'CDT':{'in_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[1,2,3,4], 'heads':[1,2,2,4], 'ffn_expansion_factor':2.67, 'bias':False, 'LayerNorm_type':'BiasFree'},
                  }

    model = DENOISERS[args.denoisername.split('-')[0]](**parameters[args.denoisername.split('-')[0]])

    return Denoiser(model, args)

