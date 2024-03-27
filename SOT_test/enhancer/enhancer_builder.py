import torch
import torch.nn.functional as F

from .SCT.model.SCT_model import SCT
from .DCE.model.DCE_model import enhance_net_nopool as DCE
from .LDEnhancer.model.enhance_model import enhancer as LDE
from .HighlightNet.model.HighlightNet_model import enhance_net_nopool as HighlightNet
from .DarkLighter.model.DarkLighter_model import enhancer as DarkLighter
ENHANCERS = {
          'SCT': SCT,
          'DCE': DCE,
          'LDE': LDE,
          'HighlightNet': HighlightNet,
          'DarkLighter': DarkLighter,
         }

class Enhancer():
    def __init__(self, args):
        super(Enhancer, self).__init__()
        self.args = args
        if args.enhancername.split('-')[0]=='SCT':
            self.model = SCT(img_size=128,embed_dim=32,win_size=4,token_embed='linear',token_mlp='resffn')
        elif args.enhancername.split('-')[0]=='DCE':
            self.model = DCE(scale_factor=12)
        elif args.enhancername.split('-')[0]=='LDE':
            self.model = LDE()
        elif args.enhancername.split('-')[0]=='DarkLighter':
            self.model = DarkLighter()
        elif args.enhancername.split('-')[0]=='HighlightNet':
            self.model = HighlightNet()
        
        self.model.load_state_dict(torch.load(args.e_weights))
        self.model.cuda().eval()
        
    def enhance(self, img):

        input_ = torch.div(img, 255.)
        if self.args.enhancername.split('-')[0]=='DCE':
            self.multiples = 12

            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+self.multiples)//self.multiples)*self.multiples, ((w+self.multiples)//self.multiples)*self.multiples
            padh = H-h if h%self.multiples!=0 else 0
            padw = W-w if w%self.multiples!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            enhanced,_ = self.model(input_)
            enhanced = enhanced[:,:,:h,:w]

        elif self.args.enhancername.split('-')[0] in ['LDE']:
            self.multiples = 8

            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+self.multiples)//self.multiples)*self.multiples, ((w+self.multiples)//self.multiples)*self.multiples
            padh = H-h if h%self.multiples!=0 else 0
            padw = W-w if w%self.multiples!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            enhanced = self.model(input_)[0]
            enhanced = enhanced[:,:,:h,:w]
        else:
            enhanced = self.model(input_)

        enhanced = torch.clamp(enhanced, 0, 1)

        return torch.mul(enhanced, 255.)


def build_enhancer(args):
    return Enhancer(args)

