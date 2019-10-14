from fastai.vision import *
from fastai.vision.gan import *
import torchvision
from torchvision.models import vgg16_bn
from fastai.callbacks import *
from src.config import *
import os
import sys

base_loss = F.l1_loss

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]
        

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

def save_pred(data, generator, path):
    """Generate images from mask, and saves them

    Parameters
    ----------
    data: list
        full path and name of all the masks
    generator: nn.Sequential
        The model that generates the images
    path: str
        path for where to save the generated images
    """
    if (os.path.isdir(path)==False): os.mkdir(path)
    for i in range(len(data)):
        if (data[i].endswith('.jpg') or data[i].endswith('.png')):
            img = open_image(data[i])
            _, pred_img, _ = generator.predict(img)
            s = "_"
            name = 'img_' + s.join(data[i].split('/')[-1].split('_')[1:])
            torchvision.utils.save_image(pred_img, path + '/' + name)
        sys.stdout.write("\r[Progress: %d/%d]"% (i, len(data)))

    
    
def label_data_critic (org_data, gen_data, path_org, path_gen, classes):
    """ Returns df with labeled data for critic

    Parameters
    ----------
    org_data: list
        filenames of original data (Real)
    gen:data: list
        filenames of generated data (Fake)
    path_org: str
        the path to the original data
    path_gen: str
        the path to the generated data

    Returns
    -------
    pd.Dataframe
        containes all data with filename and label
    """
    labeled_data = pd.DataFrame(columns=['Filenames', 'label'])
    for i in org_data:
        if (i.endswith('.jpg') or i.endswith('.png')):
            labeled_data = labeled_data.append({'Filenames': path_org + '/' + i, 'label': classes[0]}, ignore_index=True)
    for i in gen_data:
        if (i.endswith('.jpg') or i.endswith('.png')):
            labeled_data = labeled_data.append({'Filenames': path_gen + '/' + i, 'label': classes[1]}, ignore_index=True)
    
    return labeled_data
        